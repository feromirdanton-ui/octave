"""Step 3 — the lattice player.

A *lattice player* is the whole stack acting as one agent:

    field (network prior + value)  ->  search (PUCT)  ->  guard (concept derivative)
                                   ->  memory (retrieval prior)

It is the thing whose choices the network is trained to imitate. That is the inversion of
Caïssa-1: there, the network imitated human moves and the guard was a post-hoc filter. Here the
network imitates *the player*, and the player is stronger than the network alone — so each
distillation round moves the field toward the player, and the player moves with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from caissa2.guard.guard import ConceptGuard
from caissa2.memory.index import ConceptMemory
from caissa2.search.puct import PuctSearch


@dataclass
class LatticePlayer:
    """Policy + search + guard + memory, as one move-choosing agent."""

    model: object
    simulations: int = 32
    use_search: bool = True
    use_guard: bool = True
    use_memory: bool = True
    memory: ConceptMemory | None = None
    c_puct: float = 1.6
    seed: int = 0
    name: str = "lattice"
    controller: object | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.guard = ConceptGuard(self.model) if self.use_guard else None
        if self.memory is None and self.use_memory:
            self.memory = ConceptMemory()
        self.search = PuctSearch(
            self.model,
            guard=self.guard if self.guard is not None else ConceptGuard(self.model, use_head=False),
            memory=self.memory if self.use_memory else None,
            c_puct=self.c_puct,
            simulations=self.simulations,
            seed=self.seed,
        )

    def _budget(self, board: chess.Board) -> int:
        """Simulations for this position: fixed, or routed by the metacognitive controller."""
        if self.controller is None:
            return self.simulations
        sims, _ = self.controller.budget(board, self.model)
        return int(sims)

    # -- move choice -------------------------------------------------------------
    def policy_distribution(self, board: chess.Board, temperature: float = 0.0) -> dict[str, float]:
        """The player's own distribution over moves — the distillation target."""
        if self.use_search:
            sims = self._budget(board)
            if sims != self.search.simulations:
                # a routed budget is per-position, so the search object is retuned rather than
                # rebuilt: rebuilding would throw away the tree and the RNG stream
                self.search.simulations = sims
            return self.search.visit_policy(board, temperature=temperature)
        from caissa2.core.encoding import legal_moves_by_index, move_to_index
        from caissa2.model.features import encode_batch

        x = encode_batch([board])
        mask = PuctSearch._mask(board)
        probs = self.model.policy(x, mask)[0]
        out = {}
        for mv in legal_moves_by_index(board).values():
            out[mv.uci()] = float(probs[move_to_index(mv)])
        total = sum(out.values())
        return {k: v / total for k, v in out.items()} if total > 0 else out

    def choose(self, board: chess.Board, temperature: float = 0.0) -> chess.Move | None:
        dist = self.policy_distribution(board, temperature=temperature)
        if not dist:
            return None
        if temperature <= 0:
            uci = max(dist, key=dist.get)
        else:
            keys = list(dist)
            p = np.asarray([dist[k] for k in keys], dtype=float)
            p = p / p.sum()
            uci = str(self.rng.choice(keys, p=p))
        return chess.Move.from_uci(uci)

    def explain(self, board: chess.Board) -> dict:
        return self.search.explain(board)

    # -- learning from its own play ---------------------------------------------
    def remember(self, board: chess.Board, move: chess.Move, weight: float = 1.0) -> None:
        if self.memory is not None:
            self.memory.add(board, move, weight=weight)


@dataclass
class GameRecord:
    """One self-play game: the positions, the player's choices, and the result."""

    moves: list[str] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    result: float = 0.0  # +1 white wins, -1 black wins, 0 draw
    plies: int = 0


def play_game(
    white: LatticePlayer,
    black: LatticePlayer,
    max_plies: int = 120,
    temperature: float = 0.6,
    remember: bool = True,
) -> GameRecord:
    """Play one game between two players, recording every decision for distillation."""
    board = chess.Board()
    rec = GameRecord()
    while not board.is_game_over(claim_draw=True) and rec.plies < max_plies:
        player = white if board.turn == chess.WHITE else black
        dist = player.policy_distribution(board, temperature=temperature)
        if not dist:
            break
        keys = list(dist)
        p = np.asarray([dist[k] for k in keys], dtype=float)
        p = p / p.sum()
        uci = str(player.rng.choice(keys, p=p))
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        rec.samples.append(
            {
                "fen": board.fen(),
                "move_uci": uci,
                "dist": dict(dist),
                "turn": board.turn,
            }
        )
        if remember:
            player.remember(board, move)
        board.push(move)
        rec.moves.append(uci)
        rec.plies += 1

    if board.is_checkmate():
        rec.result = -1.0 if board.turn == chess.WHITE else 1.0
    else:
        rec.result = 0.0
    return rec
