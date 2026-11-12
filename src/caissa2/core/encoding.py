"""Board encoding (832D) and the legal-move mask (4096D).

Carried over from Caïssa-1 stage 1 unchanged in *meaning*: 12 piece planes of 64 squares plus
one 64-square "in check" plane, and a 64x64 from/to move space. Caïssa-2 keeps this contract
because the concept extractor, the guard and the memory index are all defined on it.
"""

from __future__ import annotations

from typing import List, Tuple

import chess

PIECE_ORDER = [
    (chess.PAWN, chess.WHITE),
    (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE),
    (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK),
    (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK),
    (chess.KING, chess.BLACK),
]

BOARD_DIM = 64 * 12 + 64
MOVE_DIM = 64 * 64


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, promotion: int | None = None) -> chess.Move:
    frm, to = divmod(int(index), 64)
    return chess.Move(frm, to, promotion=promotion)


def encode_board(board: chess.Board) -> List[float]:
    """832 floats: 12 piece planes (side-to-move first) + an in-check plane."""
    us = board.turn
    planes: List[float] = []
    order = PIECE_ORDER if us == chess.WHITE else PIECE_ORDER[6:] + PIECE_ORDER[:6]
    for p_type, p_color in order:
        mask = int(board.pieces(p_type, p_color))
        for sq in range(64):
            real_sq = sq if us == chess.WHITE else chess.square_mirror(sq)
            planes.append(1.0 if (mask >> real_sq) & 1 else 0.0)
    meta_val = 1.0 if board.is_check() else 0.0
    planes.extend([meta_val] * 64)
    return planes


def legal_move_mask(board: chess.Board) -> Tuple[List[bool], dict[int, str]]:
    """4096-entry boolean mask plus an index -> UCI map (queen promotion wins a collision)."""
    mask = [False] * MOVE_DIM
    idx_map: dict[int, str] = {}
    for mv in board.legal_moves:
        idx = move_to_index(mv)
        if idx not in idx_map or mv.promotion == chess.QUEEN:
            mask[idx] = True
            idx_map[idx] = mv.uci()
    return mask, idx_map


def legal_moves_by_index(board: chess.Board) -> dict[int, chess.Move]:
    """Index -> legal ``chess.Move`` (queen promotion wins a collision, as in the mask)."""
    out: dict[int, chess.Move] = {}
    for mv in board.legal_moves:
        idx = move_to_index(mv)
        if idx not in out or mv.promotion == chess.QUEEN:
            out[idx] = mv
    return out
