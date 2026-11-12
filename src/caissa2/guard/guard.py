"""The guard — the *derivative of the concept head*, not a hand-written rulebook.

In Caïssa-1 the guard was a separate heuristic (Queens Guard v0) that enumerated threats with
python-chess. In Caïssa-2 the concept head is a differentiable measurement of the position, so
the guard can ask a sharper question: **along this move, which axes move, and by how much?**

The cut is *relative*, and that matters. An absolute threshold on a concept delta is meaningless
because the head's scale is not calibrated to pawns: measured on real positions, the median
safety-axis drop is ~0.7, so any fixed threshold either cuts everything or nothing. What is
meaningful is an **outlier**: a move whose safety-axis drop is far outside the distribution of
drops among its siblings in the same position. That is a derivative pointing downhill *relative
to the alternatives*, which is exactly the question a guard should answer.

A material signal (static exchange evaluation) is kept as a second, cheap, non-learned check,
because a concept head can be wrong and a hanging queen is not a matter of opinion.

Every cut carries its reason in axis terms, which is what step 2 needs in order to say *why* a
branch was discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from caissa2.core.concepts import concepts_after, extract_ground_truth_concepts
from caissa2.core.types import CONCEPT_NAMES, SAFETY_AXES, axis_deltas

PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 1000,
}

MATERIAL_THRESHOLD = 2


def _least_valuable_legal_capturer(board: chess.Board, square: int) -> chess.Move | None:
    best: tuple[int, chess.Move] | None = None
    for move in board.legal_moves:
        if move.to_square != square:
            continue
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue
        value = PIECE_VALUES[piece.piece_type]
        if best is None or value < best[0]:
            best = (value, move)
    return best[1] if best else None


def static_exchange_eval(board: chess.Board, move: chess.Move) -> int:
    """Net material the mover wins by initiating the exchange on ``move.to_square``."""
    target = board.piece_at(move.to_square)
    if target is None:
        return 0
    gain = [PIECE_VALUES[target.piece_type]]
    b = board.copy()
    b.push(move)
    while True:
        recapture = _least_valuable_legal_capturer(b, move.to_square)
        if recapture is None:
            break
        captured = b.piece_at(move.to_square)
        if captured is None:
            break
        gain.append(PIECE_VALUES[captured.piece_type] - gain[-1])
        b.push(recapture)
    d = len(gain) - 1
    while d > 0:
        gain[d - 1] = -max(-gain[d - 1], gain[d])
        d -= 1
    return gain[0]


def worst_material_loss(board: chess.Board, move: chess.Move) -> tuple[int, str]:
    """Best material the opponent wins after ``move`` (SEE), and the move that does it."""
    after = board.copy()
    after.push(move)
    worst = 0
    worst_uci = ""
    for opp in after.legal_moves:
        if after.piece_at(opp.to_square) is None:
            continue
        see = static_exchange_eval(after, opp)
        if see > worst:
            worst = see
            worst_uci = opp.uci()
    return worst, worst_uci


@dataclass
class GuardVerdict:
    """The guard's answer for one candidate move, in axis terms."""

    move_uci: str
    passed: bool
    deltas: dict[str, float] = field(default_factory=dict)
    material_loss: int = 0
    material_move: str = ""
    reasons: list[str] = field(default_factory=list)
    z_score: float = 0.0

    def explain(self) -> str:
        if self.passed:
            return f"PASS {self.move_uci}: no safety axis is an outlier drop"
        return f"CUT {self.move_uci}: " + "; ".join(self.reasons)


class ConceptGuard:
    """Cuts moves whose concept-head derivative is an outlier on a safety axis."""

    def __init__(
        self,
        model=None,
        z_threshold: float = 2.0,
        min_drop: float = 0.15,
        material_threshold: int = MATERIAL_THRESHOLD,
        use_head: bool = True,
    ) -> None:
        self.model = model
        self.z_threshold = z_threshold
        self.min_drop = min_drop
        self.material_threshold = material_threshold
        self.use_head = use_head and model is not None

    # -- concept measurement -----------------------------------------------------
    def _concepts(self, boards: list[chess.Board]) -> np.ndarray:
        if self.use_head:
            from caissa2.model.features import encode_batch

            return self.model.concepts(encode_batch(boards))
        return np.stack([extract_ground_truth_concepts(b).as_array() for b in boards])

    def _concepts_after(self, boards: list[chess.Board], moves: list[chess.Move]) -> np.ndarray:
        if self.use_head:
            from caissa2.model.features import encode_batch

            afters = []
            for b, mv in zip(boards, moves):
                a = b.copy()
                a.push(mv)
                afters.append(a)
            # the head is side-to-move relative; negate to read the pushed position as the mover
            return -self.model.concepts(encode_batch(afters))
        return np.stack([concepts_after(b, mv).as_array() for b, mv in zip(boards, moves)])

    # -- the cut -----------------------------------------------------------------
    def inspect_many(self, board: chess.Board, moves: list[chess.Move]) -> list[GuardVerdict]:
        """Inspect every candidate in one batch, then cut the outliers among them."""
        if not moves:
            return []
        legal = [mv for mv in moves if mv in board.legal_moves]
        verdicts: dict[str, GuardVerdict] = {}
        for mv in moves:
            if mv not in board.legal_moves:
                verdicts[mv.uci()] = GuardVerdict(
                    move_uci=mv.uci(), passed=False, reasons=["illegal move"]
                )
        if not legal:
            return [verdicts[mv.uci()] for mv in moves]

        before = self._concepts([board])[0]
        after = self._concepts_after([board] * len(legal), legal)
        deltas = after - before[None, :]

        safety_idx = [list(CONCEPT_NAMES).index(a) for a in SAFETY_AXES]
        drops = -deltas[:, safety_idx]  # positive = axis went down
        worst_drop = drops.max(axis=1)

        # relative outlier test: how far is this move's worst safety drop from its siblings?
        med = float(np.median(worst_drop))
        mad = float(np.median(np.abs(worst_drop - med))) * 1.4826
        scale = max(mad, 1e-3)
        z = (worst_drop - med) / scale

        for i, mv in enumerate(legal):
            v = GuardVerdict(move_uci=mv.uci(), passed=True)
            v.deltas = {name: float(deltas[i, j]) for j, name in enumerate(CONCEPT_NAMES)}
            v.z_score = float(z[i])
            if z[i] > self.z_threshold and worst_drop[i] > self.min_drop:
                axis = SAFETY_AXES[int(np.argmax(drops[i]))]
                v.passed = False
                v.reasons.append(
                    f"{axis} {v.deltas[axis]:+.2f} is an outlier drop "
                    f"(z={z[i]:.1f} vs siblings, median {med:+.2f})"
                )
            loss, loss_move = worst_material_loss(board, mv)
            v.material_loss = loss
            v.material_move = loss_move
            if loss >= self.material_threshold:
                v.passed = False
                v.reasons.append(f"material -{loss} to {loss_move} (SEE)")
            verdicts[mv.uci()] = v

        return [verdicts[mv.uci()] for mv in moves]

    def inspect(self, board: chess.Board, move: chess.Move) -> GuardVerdict:
        """Single-move inspection (no sibling distribution: only the material check is absolute)."""
        if move not in board.legal_moves:
            return GuardVerdict(move_uci=move.uci(), passed=False, reasons=["illegal move"])
        before = self._concepts([board])[0]
        after = self._concepts_after([board], [move])[0]
        v = GuardVerdict(move_uci=move.uci(), passed=True)
        v.deltas = axis_deltas(before, after)
        loss, loss_move = worst_material_loss(board, move)
        v.material_loss = loss
        v.material_move = loss_move
        if loss >= self.material_threshold:
            v.passed = False
            v.reasons.append(f"material -{loss} to {loss_move} (SEE)")
        return v

    def filter(
        self, board: chess.Board, moves: list[chess.Move]
    ) -> tuple[list[chess.Move], list[GuardVerdict]]:
        """Split candidates into (kept, cut verdicts). Never returns an empty kept list."""
        verdicts = self.inspect_many(board, moves)
        kept = [mv for mv, v in zip(moves, verdicts) if v.passed]
        cut = [v for v in verdicts if not v.passed]
        if not kept and cut:
            # The guard is a hygiene gate, not a suicide pact: if every move is cut, keep the
            # least-bad one and say so, rather than returning no move at all.
            best = min(cut, key=lambda v: (v.material_loss, v.z_score))
            kept = [m for m in moves if m.uci() == best.move_uci]
            cut = [v for v in cut if v.move_uci != best.move_uci]
        return kept, cut
