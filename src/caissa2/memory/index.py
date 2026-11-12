"""Step 4 — memory as geometry.

A position is a point in concept space. The index stores, for every position seen, its nine
floats (the eight axes plus a material balance) and the move that turned out best there. At
search time the retrieval prior asks: *in the positions nearest to this one, what was the best
move?* — and hands that answer to PUCT as a prior.

This is deliberately not a hash table of exact positions. Two positions that are far apart on
the board but close in concept space are the same *kind* of position, and that is the whole
point: the memory generalises along the axes the system already measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from caissa2.core.concepts import extract_ground_truth_concepts
from caissa2.core.types import N_CONCEPTS

#: The index is 9 floats: 8 concept axes + material balance in [-1, 1].
INDEX_DIM = N_CONCEPTS + 1

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def material_balance(board: chess.Board) -> float:
    """Material from the side-to-move's view, squashed into [-1, 1]."""
    us, them = board.turn, not board.turn
    mine = sum(
        len(board.pieces(pt, us)) * v for pt, v in PIECE_VALUES.items()
    )
    theirs = sum(
        len(board.pieces(pt, them)) * v for pt, v in PIECE_VALUES.items()
    )
    return float(np.tanh((mine - theirs) / 10.0))


def position_key(board: chess.Board) -> np.ndarray:
    """The 9-float concept index of a position."""
    c = extract_ground_truth_concepts(board).as_array()
    return np.concatenate([c, [material_balance(board)]]).astype(np.float32)


@dataclass
class MemoryEntry:
    key: np.ndarray
    move_uci: str
    weight: float = 1.0
    visits: int = 1


@dataclass
class ConceptMemory:
    """A concept-space index with a k-NN retrieval prior."""

    k: int = 8
    temperature: float = 0.15
    entries: list[MemoryEntry] = field(default_factory=list)
    _keys: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, board: chess.Board, move: chess.Move | str, weight: float = 1.0) -> None:
        uci = move.uci() if isinstance(move, chess.Move) else str(move)
        key = position_key(board)
        for e in self.entries:
            if e.move_uci == uci and float(np.linalg.norm(e.key - key)) < 1e-6:
                e.weight += weight
                e.visits += 1
                self._keys = None
                return
        self.entries.append(MemoryEntry(key=key, move_uci=uci, weight=weight))
        self._keys = None

    def _matrix(self) -> np.ndarray:
        if self._keys is None:
            self._keys = (
                np.stack([e.key for e in self.entries])
                if self.entries
                else np.zeros((0, INDEX_DIM), dtype=np.float32)
            )
        return self._keys

    def neighbours(self, board: chess.Board, k: int | None = None) -> list[tuple[float, MemoryEntry]]:
        if not self.entries:
            return []
        k = k or self.k
        q = position_key(board)
        d = np.linalg.norm(self._matrix() - q[None, :], axis=1)
        order = np.argsort(d)[:k]
        return [(float(d[i]), self.entries[i]) for i in order]

    def retrieval_prior(self, board: chess.Board, legal_ucis: set[str] | None = None) -> dict[str, float]:
        """A prior over moves from the nearest positions in concept space.

        Weight = exp(-distance / temperature) * entry weight. Returns an empty dict when the
        memory is empty or nothing retrieved is legal here.
        """
        near = self.neighbours(board)
        if not near:
            return {}
        scores: dict[str, float] = {}
        for dist, entry in near:
            if legal_ucis is not None and entry.move_uci not in legal_ucis:
                continue
            w = float(np.exp(-dist / max(self.temperature, 1e-6))) * entry.weight
            scores[entry.move_uci] = scores.get(entry.move_uci, 0.0) + w
        total = sum(scores.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in scores.items()}

    def best_move(self, board: chess.Board) -> str | None:
        prior = self.retrieval_prior(board)
        if not prior:
            return None
        return max(prior, key=prior.get)
