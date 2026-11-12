"""Feature plumbing: board -> 832-D vector, and a batched encoder."""

from __future__ import annotations

import chess
import numpy as np

from caissa2.core.encoding import BOARD_DIM, encode_board


def encode_batch(boards: list[chess.Board]) -> np.ndarray:
    if not boards:
        return np.zeros((0, BOARD_DIM), dtype=np.float32)
    return np.asarray([encode_board(b) for b in boards], dtype=np.float32)


def mask_batch(boards: list[chess.Board]) -> np.ndarray:
    from caissa2.core.encoding import legal_move_mask

    out = np.zeros((len(boards), 64 * 64), dtype=bool)
    for i, b in enumerate(boards):
        m, _ = legal_move_mask(b)
        out[i] = np.asarray(m, dtype=bool)
    return out
