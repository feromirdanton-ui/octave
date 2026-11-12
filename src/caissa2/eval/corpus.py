"""A deterministic corpus and a deterministic teacher — no engine, no network, no downloads.

The teacher is a 2-ply search over a material + concept evaluation. It is not Stockfish and it
does not pretend to be: it is a *fixed, reproducible* reference whose moves are good enough to
supervise a policy and to define "a blunder" objectively (a move that loses material by SEE).
Every number in the report is therefore reproducible from a seed alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from caissa2.core.concepts import extract_ground_truth_concepts
from caissa2.core.types import CONCEPT_NAMES
from caissa2.guard.guard import static_exchange_eval

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

#: How much each concept axis is worth, in pawns, to the teacher's evaluation.
CONCEPT_WEIGHTS = {
    "king_safety": 0.9,
    "center_control": 0.7,
    "piece_activity": 0.5,
    "pawn_structure": 0.6,
    "tactical_tension": 0.15,
    "passed_pawn_threat": 0.8,
    "open_file_dominance": 0.4,
    "endgame_simplification": 0.1,
}


def material(board: chess.Board) -> float:
    """Material from the side-to-move's view, in pawns."""
    us, them = board.turn, not board.turn
    mine = sum(len(board.pieces(pt, us)) * v for pt, v in PIECE_VALUES.items())
    theirs = sum(len(board.pieces(pt, them)) * v for pt, v in PIECE_VALUES.items())
    return float(mine - theirs)


def evaluate(board: chess.Board) -> float:
    """Static evaluation in pawns, from the side-to-move's view."""
    if board.is_checkmate():
        return -100.0
    if board.is_game_over(claim_draw=True):
        return 0.0
    c = extract_ground_truth_concepts(board)
    score = material(board)
    for name in CONCEPT_NAMES:
        score += CONCEPT_WEIGHTS[name] * c.get(name)
    return float(score)


def teacher_move(board: chess.Board, depth: int = 2) -> chess.Move:
    """Deterministic 2-ply search: maximise the worst-case static evaluation."""
    best_move = None
    best_score = -1e9
    for mv in board.legal_moves:
        after = board.copy()
        after.push(mv)
        if after.is_checkmate():
            return mv
        if depth <= 1 or after.is_game_over(claim_draw=True):
            score = -evaluate(after)
        else:
            worst = 1e9
            for reply in after.legal_moves:
                b2 = after.copy()
                b2.push(reply)
                worst = min(worst, -evaluate(b2))
            score = worst
        # deterministic tie-break: UCI string, so the corpus never depends on dict order
        if score > best_score + 1e-9 or (
            abs(score - best_score) <= 1e-9 and best_move is not None and mv.uci() < best_move.uci()
        ):
            best_score = score
            best_move = mv
    assert best_move is not None
    return best_move


@dataclass
class CorpusItem:
    fen: str
    move_uci: str
    concepts: np.ndarray


def generate_corpus(n_positions: int = 400, seed: int = 0, max_plies: int = 40) -> list[CorpusItem]:
    """Play seeded games with the teacher and record (position, teacher move, concepts)."""
    rng = np.random.default_rng(seed)
    items: list[CorpusItem] = []
    while len(items) < n_positions:
        board = chess.Board()
        for _ in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            if len(items) >= n_positions:
                break
            # a little opening noise so the corpus is not 400 copies of one line
            if board.ply() < 6 and rng.random() < 0.5:
                moves = list(board.legal_moves)
                mv = moves[int(rng.integers(len(moves)))]
            else:
                mv = teacher_move(board)
            items.append(
                CorpusItem(
                    fen=board.fen(),
                    move_uci=mv.uci(),
                    concepts=extract_ground_truth_concepts(board).as_array(),
                )
            )
            board.push(mv)
    return items


def generate_tactical_corpus(
    n_positions: int = 200, seed: int = 0, max_plies: int = 60
) -> list[CorpusItem]:
    """Positions where a blunder is *possible* — the only place a blunder rate means anything.

    A quiet opening position has no move that loses material, so every model scores 0% and the
    metric measures nothing. Here we keep only positions in which at least one legal move loses
    >= 2 pawns by SEE, and where the teacher's move does not. That is the population the guard
    and the search are supposed to protect.
    """
    rng = np.random.default_rng(seed)
    items: list[CorpusItem] = []
    attempts = 0
    while len(items) < n_positions and attempts < n_positions * 200:
        attempts += 1
        board = chess.Board()
        for _ in range(int(rng.integers(8, max_plies))):
            if board.is_game_over(claim_draw=True):
                break
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
        if board.is_game_over(claim_draw=True):
            continue
        legal = list(board.legal_moves)
        if not any(blunder_loss(board, mv) >= 2 for mv in legal):
            continue
        best = teacher_move(board)
        if blunder_loss(board, best) >= 2:
            continue
        items.append(
            CorpusItem(
                fen=board.fen(),
                move_uci=best.uci(),
                concepts=extract_ground_truth_concepts(board).as_array(),
            )
        )
    return items


def blunder_loss(board: chess.Board, move: chess.Move) -> int:
    """Material the opponent wins after ``move`` by SEE — the objective blunder definition."""
    after = board.copy()
    after.push(move)
    worst = 0
    for opp in after.legal_moves:
        if after.piece_at(opp.to_square) is None:
            continue
        worst = max(worst, static_exchange_eval(after, opp))
    return worst


def is_blunder(board: chess.Board, move: chess.Move, threshold: int = 2) -> bool:
    return blunder_loss(board, move) >= threshold
