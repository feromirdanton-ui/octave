"""Step 2 — search over the field.

The field (the network) gives a **prior** over moves and a **value** for a position. PUCT turns
those into a tree. The guard sits between the two: a move whose concept-head derivative is an
outlier drop on a safety axis is never expanded, and the reason is kept in axis terms so the
search can say *why* a branch was discarded.

The prior at the root mixes the network policy (the field) with the retrieval prior from memory
(step 4) — "in positions that look like this one, the best move was that". The guard is a
*cut*, not a prior: it removes, it does not reweight.

Cost discipline: the full concept guard (a batch forward pass plus SEE for every candidate) runs
at the root and at depth 1, where the explanation is actually read. Deeper nodes use the cheap
material-only check, because a search that spends all its time in the guard is not a search.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from caissa2.core.encoding import legal_moves_by_index, move_to_index
from caissa2.guard.guard import ConceptGuard, worst_material_loss


@dataclass
class CutRecord:
    """A branch the search refused to expand, with its reason in axis terms."""

    move_uci: str
    reasons: list[str]
    deltas: dict[str, float]

    def explain(self) -> str:
        return f"{self.move_uci}: " + "; ".join(self.reasons)


@dataclass
class Node:
    board: chess.Board
    prior: float = 1.0
    move: chess.Move | None = None
    parent: "Node | None" = None
    children: dict[str, "Node"] = field(default_factory=dict)
    n: int = 0
    w: float = 0.0
    value: float = 0.0
    terminal: bool = False
    cuts: list[CutRecord] = field(default_factory=list)
    expanded: bool = False
    depth: int = 0
    proposals: list[str] = field(default_factory=list)

    @property
    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


class PuctSearch:
    """PUCT/MCTS with a network prior+value and a concept-derivative guard."""

    def __init__(
        self,
        model,
        guard: ConceptGuard | None = None,
        memory=None,
        c_puct: float = 1.6,
        simulations: int = 64,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.0,
        seed: int = 0,
        memory_weight: float = 0.5,
        guard_depth: int = 1,
        material_threshold: int = 2,
        memory_proposals: int = 3,
        proposal_floor: float = 0.08,
    ) -> None:
        self.model = model
        self.guard = guard if guard is not None else ConceptGuard(model)
        self.memory = memory
        self.c_puct = c_puct
        self.simulations = simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = np.random.default_rng(seed)
        self.memory_weight = memory_weight
        self.guard_depth = guard_depth
        self.material_threshold = material_threshold
        #: How many memory-proposed moves may be forced into the root's candidate set, and the
        #: prior floor they get. Without this the memory can only reweight moves the field
        #: already likes; with it, the memory can make the search *look* at a move the field
        #: would never have visited.
        self.memory_proposals = memory_proposals
        self.proposal_floor = proposal_floor

    # -- field -------------------------------------------------------------------
    @staticmethod
    def _mask(board: chess.Board) -> np.ndarray:
        from caissa2.core.encoding import legal_move_mask

        m, _ = legal_move_mask(board)
        return np.asarray(m, dtype=bool)[None, :]

    def _evaluate(self, board: chess.Board) -> tuple[dict[str, float], float]:
        """Network prior over legal moves (dict uci->p) and a value in [-1, 1]."""
        from caissa2.model.features import encode_batch

        x = encode_batch([board])
        idx_map = legal_moves_by_index(board)
        probs = self.model.policy(x, self._mask(board))[0]
        value = float(self.model.value(x)[0])
        prior = {mv.uci(): float(probs[move_to_index(mv)]) for mv in idx_map.values()}
        total = sum(prior.values())
        if total > 0:
            prior = {k: v / total for k, v in prior.items()}
        return prior, value

    def _prior(self, board: chess.Board, depth: int) -> tuple[dict[str, float], float]:
        prior, value = self._evaluate(board)
        if depth == 0 and self.memory is not None and prior:
            boost = self.memory.retrieval_prior(board, legal_ucis=set(prior))
            if boost:
                for uci, p in boost.items():
                    if uci in prior:
                        prior[uci] = (1.0 - self.memory_weight) * prior[uci] + self.memory_weight * p
                total = sum(prior.values())
                prior = {k: v / total for k, v in prior.items()}
        return prior, value

    # -- guard -------------------------------------------------------------------
    def _cut(self, board: chess.Board, moves: list[chess.Move], depth: int):
        """(kept, cut records) — full concept guard near the root, material-only deeper."""
        if depth <= self.guard_depth:
            return self.guard.filter(board, moves)
        kept: list[chess.Move] = []
        cut = []
        for mv in moves:
            loss, loss_move = worst_material_loss(board, mv)
            if loss >= self.material_threshold:
                cut.append(
                    _MaterialCut(mv.uci(), [f"material -{loss} to {loss_move} (SEE)"], {})
                )
            else:
                kept.append(mv)
        if not kept and cut:
            kept = [moves[0]]
            cut = cut[1:]
        return kept, cut

    # -- tree --------------------------------------------------------------------
    def _expand(self, node: Node) -> float:
        board = node.board
        if board.is_game_over(claim_draw=True):
            node.terminal = True
            node.expanded = True
            node.value = -1.0 if board.is_checkmate() else 0.0
            return node.value

        prior, value = self._prior(board, node.depth)
        node.value = value
        node.expanded = True

        moves = list(board.legal_moves)
        kept, cut = self._cut(board, moves, node.depth)
        node.cuts = [
            CutRecord(move_uci=v.move_uci, reasons=list(v.reasons), deltas=dict(v.deltas))
            for v in cut
        ]
        kept_ucis = {m.uci() for m in kept}
        kept_prior = {k: v for k, v in prior.items() if k in kept_ucis}

        # Memory as a *director of exploration*, not only a root reweighting. A move the field
        # ranks at ~0 can never be visited by PUCT no matter how the root is reweighted, so a
        # confident neighbourhood is allowed to inject its recommendation into the candidate set
        # with a prior floor. The guard still has the last word: a proposal that the guard cut is
        # not resurrected.
        proposed: list[str] = []
        if node.depth == 0 and self.memory is not None and self.memory_proposals > 0:
            legal_ucis = {m.uci() for m in kept}
            proposed = self.memory.propose(board, legal_ucis, top_n=self.memory_proposals)
            for uci in proposed:
                kept_prior[uci] = max(kept_prior.get(uci, 0.0), self.proposal_floor)
            node.proposals = list(proposed)

        total = sum(kept_prior.values())
        if total <= 0:
            kept_prior = {k: 1.0 / max(len(kept_ucis), 1) for k in kept_ucis}
        else:
            kept_prior = {k: v / total for k, v in kept_prior.items()}

        if self.dirichlet_eps > 0 and len(kept_prior) > 1:
            keys = list(kept_prior)
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(keys))
            for k, e in zip(keys, noise):
                kept_prior[k] = (1 - self.dirichlet_eps) * kept_prior[k] + self.dirichlet_eps * e

        for mv in kept:
            node.children[mv.uci()] = Node(
                board=board.copy(),
                prior=kept_prior.get(mv.uci(), 0.0),
                move=mv,
                parent=node,
                depth=node.depth + 1,
            )
        return value

    def _select(self, node: Node) -> Node:
        while node.children and node.expanded:
            best = None
            best_score = -1e18
            sqrt_n = np.sqrt(max(node.n, 1))
            for child in node.children.values():
                u = self.c_puct * child.prior * sqrt_n / (1 + child.n)
                score = child.q + u
                if score > best_score:
                    best_score = score
                    best = child
            assert best is not None
            node = best
            if not node.expanded:
                return node
        return node

    def _backup(self, node: Node, value: float) -> None:
        sign = 1.0
        cur: Node | None = node
        while cur is not None:
            cur.n += 1
            cur.w += sign * value
            sign = -sign
            cur = cur.parent

    def run(self, board: chess.Board) -> Node:
        root = Node(board=board.copy())
        self._expand(root)
        for _ in range(self.simulations):
            leaf = self._select(root)
            if leaf.terminal:
                value = leaf.value
            elif not leaf.expanded:
                value = self._expand(leaf)
            else:
                value = leaf.value
            self._backup(leaf, value)
        return root

    # -- answers -----------------------------------------------------------------
    def best_move(self, board: chess.Board) -> chess.Move | None:
        root = self.run(board)
        if not root.children:
            return None
        best = max(root.children.values(), key=lambda c: (c.n, c.q))
        return best.move

    def visit_policy(self, board: chess.Board, temperature: float = 0.0) -> dict[str, float]:
        root = self.run(board)
        if not root.children:
            return {}
        counts = {k: c.n for k, c in root.children.items()}
        total = sum(counts.values())
        if total == 0:
            return {k: 1.0 / len(counts) for k in counts}
        if temperature <= 0:
            best = max(counts, key=counts.get)
            return {k: (1.0 if k == best else 0.0) for k in counts}
        weights = {k: (v / total) ** (1.0 / temperature) for k, v in counts.items()}
        s = sum(weights.values())
        return {k: v / s for k, v in weights.items()}

    def explain(self, board: chess.Board) -> dict:
        """Why the search chose what it chose, and why it discarded what it discarded."""
        root = self.run(board)
        chosen = max(root.children.values(), key=lambda c: (c.n, c.q)) if root.children else None
        return {
            "chosen": chosen.move.uci() if chosen and chosen.move else None,
            "visits": {k: c.n for k, c in root.children.items()},
            "q": {k: round(c.q, 4) for k, c in root.children.items()},
            "discarded": [c.explain() for c in root.cuts],
            "discarded_axes": [
                {"move": c.move_uci, "axes": {k: round(v, 3) for k, v in c.deltas.items()}}
                for c in root.cuts
            ],
            "memory_proposals": list(root.proposals),
        }


@dataclass
class _MaterialCut:
    """A cut made by the cheap material check, shaped like a GuardVerdict."""

    move_uci: str
    reasons: list[str]
    deltas: dict[str, float]
