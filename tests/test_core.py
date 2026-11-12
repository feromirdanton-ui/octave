"""Core contract: 832-D board, 4096-D mask, 8 concept axes in [-1, 1]."""

from __future__ import annotations

import chess
import numpy as np

from caissa2.core.concepts import concepts_after, extract_ground_truth_concepts
from caissa2.core.encoding import (
    BOARD_DIM,
    MOVE_DIM,
    encode_board,
    legal_move_mask,
    legal_moves_by_index,
    move_to_index,
)
from caissa2.core.types import CONCEPT_NAMES, N_CONCEPTS


def test_board_encoding_dimensions() -> None:
    board = chess.Board()
    assert len(encode_board(board)) == BOARD_DIM == 832


def test_mask_dimensions_and_start_position() -> None:
    mask, idx_map = legal_move_mask(chess.Board())
    assert len(mask) == MOVE_DIM == 4096
    assert sum(mask) == 20
    assert len(idx_map) == 20


def test_mask_matches_legal_moves_on_random_positions() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        board = chess.Board()
        for _ in range(int(rng.integers(1, 30))):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
        mask, idx_map = legal_move_mask(board)
        by_index = legal_moves_by_index(board)
        assert sum(mask) == len(by_index)
        for idx, mv in by_index.items():
            assert mask[idx]
            assert idx_map[idx] == mv.uci()
            assert move_to_index(mv) == idx


def test_concepts_are_eight_and_bounded() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        board = chess.Board()
        for _ in range(int(rng.integers(0, 40))):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
        c = extract_ground_truth_concepts(board)
        assert len(c.as_tuple()) == N_CONCEPTS == 8
        assert tuple(c.values.keys()) == CONCEPT_NAMES
        assert all(-1.0 <= v <= 1.0 for v in c.as_tuple())


def test_concepts_after_is_mover_relative() -> None:
    board = chess.Board()
    mv = chess.Move.from_uci("g1f3")
    c = concepts_after(board, mv)
    assert len(c.as_tuple()) == 8
    assert all(-1.0 <= v <= 1.0 for v in c.as_tuple())
