"""Ground-truth concept extractor — the teacher for the concept head.

Eight axes, each in [-1, 1], computed from python-chess only. This is the *measurement*
function: in Caïssa-2 it no longer feeds the policy, it supervises the concept head and it
defines the axes in which the guard and the search explain themselves.
"""

from __future__ import annotations

import chess

from caissa2.core.types import CONCEPT_NAMES, ConceptVector

CENTER_SQ = {chess.D4, chess.E4, chess.D5, chess.E5}


def _clamp(val: float) -> float:
    return round(max(-1.0, min(1.0, val)), 4)


def extract_ground_truth_concepts(board: chess.Board) -> ConceptVector:
    us, them = board.turn, not board.turn

    # 1. King Safety
    my_king, opp_king = board.king(us), board.king(them)
    ks_score = 0.0
    if my_king is not None and opp_king is not None:
        my_att = sum(
            len(board.attackers(them, sq)) for sq in chess.SquareSet(chess.BB_KING_ATTACKS[my_king])
        )
        opp_att = sum(
            len(board.attackers(us, sq)) for sq in chess.SquareSet(chess.BB_KING_ATTACKS[opp_king])
        )
        ks_score = (opp_att - my_att) * 0.15 - (0.4 if board.is_check() else 0.0)

    # 2. Center Control
    cc_us = sum(
        len(board.attackers(us, sq)) + (1 if board.color_at(sq) == us else 0) for sq in CENTER_SQ
    )
    cc_them = sum(
        len(board.attackers(them, sq)) + (1 if board.color_at(sq) == them else 0) for sq in CENTER_SQ
    )
    center_score = (cc_us - cc_them) * 0.12

    # 3. Piece Activity
    activity_score = (board.legal_moves.count() - 20) / 25.0

    # 4. Pawn Structure (doubled pawns penalty)
    my_pawns, opp_pawns = board.pieces(chess.PAWN, us), board.pieces(chess.PAWN, them)
    my_dbl = sum(1 for f in range(8) if len(my_pawns & chess.BB_FILES[f]) > 1)
    opp_dbl = sum(1 for f in range(8) if len(opp_pawns & chess.BB_FILES[f]) > 1)
    structure_score = (opp_dbl - my_dbl) * 0.35

    # 5. Tactical Tension
    captures = sum(1 for m in board.legal_moves if board.is_capture(m))
    checks = sum(1 for m in board.legal_moves if m and board.gives_check(m))
    tension_score = (captures * 0.18 + checks * 0.25) - 0.3

    # 6. Passed Pawn Threat
    adv = (
        (chess.BB_RANK_5 | chess.BB_RANK_6 | chess.BB_RANK_7)
        if us == chess.WHITE
        else (chess.BB_RANK_4 | chess.BB_RANK_3 | chess.BB_RANK_2)
    )
    passed_score = (len(my_pawns & adv) - len(opp_pawns & adv)) * 0.4

    # 7. Open File Dominance
    all_pawns = my_pawns | opp_pawns
    my_heavy = board.pieces(chess.ROOK, us) | board.pieces(chess.QUEEN, us)
    open_files = sum(
        1 for sq in my_heavy if not (all_pawns & chess.BB_FILES[chess.square_file(sq)])
    )
    open_file_score = open_files * 0.45

    # 8. Endgame Simplification
    non_pawns = sum(
        len(board.pieces(pt, c))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for c in (chess.WHITE, chess.BLACK)
    )
    endgame_score = ((14 - non_pawns) / 14.0) * 2.0 - 0.5

    values = {
        "king_safety": _clamp(ks_score),
        "center_control": _clamp(center_score),
        "piece_activity": _clamp(activity_score),
        "pawn_structure": _clamp(structure_score),
        "tactical_tension": _clamp(tension_score),
        "passed_pawn_threat": _clamp(passed_score),
        "open_file_dominance": _clamp(open_file_score),
        "endgame_simplification": _clamp(endgame_score),
    }
    assert tuple(values.keys()) == CONCEPT_NAMES
    return ConceptVector(values)


def concepts_after(board: chess.Board, move: chess.Move) -> ConceptVector:
    """Concepts of the position *after* ``move``, from the mover's point of view.

    The extractor is side-to-move relative, so the pushed position measures the opponent's
    view; every axis is a "us minus them" difference, so negating gives the mover's view.
    """
    after = board.copy()
    after.push(move)
    raw = extract_ground_truth_concepts(after)
    return ConceptVector({name: -raw.get(name) for name in CONCEPT_NAMES})
