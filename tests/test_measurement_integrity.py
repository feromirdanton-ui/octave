"""Measurement integrity — the checks that caught the guard cutting good moves.

Three defects were found by auditing the arena rather than by reading it:

1. The guard's material check and the blunder metric are the *same function*
   (``worst_material_loss == blunder_loss``), so a guarded player's blunder rate is partly a
   statement about its own definition. ``eval_drop`` is the independent number.
2. The concept head, allowed to cut on its own, cut moves that were *better* than the
   alternatives. It is now corroborated by material.
3. ``head_to_head`` reported 0.5/0.5 for six drawn games, which reads like a measurement and is
   really "nothing was decided". The W/L/D split is now reported alongside.

These tests pin all three.
"""

from __future__ import annotations

import chess
import numpy as np

from caissa2.eval.arena import (
    best_move_value,
    eval_drop,
    evaluate_guarded_model,
    guard_cut_audit,
    head_to_head,
    top_k_legal_moves,
)
from caissa2.eval.corpus import blunder_loss, generate_tactical_corpus
from caissa2.guard.guard import ConceptGuard, worst_material_loss
from caissa2.model.features import encode_batch
from caissa2.model.nets import OctaveNet
from caissa2.search.puct import PuctSearch


def _net() -> OctaveNet:
    return OctaveNet(seed=0, trunk_dim=32)


def test_guard_material_check_is_the_blunder_metric() -> None:
    """The circularity, stated as a test so nobody re-derives it as a discovery.

    If this ever fails, the blunder rate has become an independent number and the report's
    caveat about it can be dropped.
    """
    corpus = generate_tactical_corpus(4, seed=11)
    assert corpus
    checked = 0
    for item in corpus:
        board = chess.Board(item.fen)
        for mv in list(board.legal_moves)[:6]:
            checked += 1
            assert worst_material_loss(board, mv)[0] == blunder_loss(board, mv)
    assert checked > 0


def test_eval_drop_is_zero_for_the_best_move_and_positive_for_a_worse_one() -> None:
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    best = best_move_value(board)
    # Rxd5 wins a queen: it is the best move, so it gives up nothing.
    assert eval_drop(board, chess.Move.from_uci("d1d5"), best=best) == 0.0
    # A quiet king move gives up the queen capture.
    assert eval_drop(board, chess.Move.from_uci("e1e2"), best=best) > 0


def test_eval_drop_is_nan_for_an_illegal_move() -> None:
    board = chess.Board()
    assert np.isnan(eval_drop(board, chess.Move.from_uci("e2e5")))


def _concept_only_cut_losses(net, guard, corpus) -> list[int]:
    """Material loss of every cut the guard made *without* the material check firing."""
    losses: list[int] = []
    for item in corpus:
        board = chess.Board(item.fen)
        probs = net.policy(encode_batch([board]), PuctSearch._mask(board))[0]
        cands = top_k_legal_moves(board, probs, k=12)
        if not cands:
            continue
        _, cut = guard.filter(board, cands)
        for v in cut:
            if chess.Move.from_uci(v.move_uci) not in board.legal_moves:
                continue
            if v.material_loss < guard.material_threshold:
                losses.append(v.material_loss)
    return losses


def test_uncorroborated_head_cuts_moves_with_no_material_loss() -> None:
    """The defect, stated as the invariant it violated.

    Allowed to cut on its own, the concept head removes moves that lose *nothing* by SEE — the
    head's scale is not calibrated to material, so an outlier axis drop is not evidence of a bad
    move. Corroborated, the guard can never do that: every cut it makes has material behind it.
    """
    net = _net()
    corpus = generate_tactical_corpus(40, seed=7)
    assert corpus

    uncorroborated = _concept_only_cut_losses(net, ConceptGuard(net, concept_corroboration=0), corpus)
    assert uncorroborated, "the uncorroborated head should cut on its own"
    assert min(uncorroborated) == 0, (
        "the uncorroborated head is expected to cut moves with zero material loss"
    )

    corroborated = _concept_only_cut_losses(net, ConceptGuard(net), corpus)
    assert all(loss >= 1 for loss in corroborated), (
        "a corroborated cut must always have material behind it"
    )


def test_corroborated_guard_never_cuts_a_move_the_teacher_prefers() -> None:
    """Every cut the corroborated guard makes must be a move the teacher also dislikes."""
    net = _net()
    corpus = generate_tactical_corpus(10, seed=31337)
    guard = ConceptGuard(net)
    for item in corpus:
        board = chess.Board(item.fen)
        probs = net.policy(encode_batch([board]), PuctSearch._mask(board))[0]
        cands = top_k_legal_moves(board, probs, k=12)
        if not cands:
            continue
        _, cut = guard.filter(board, cands)
        best = best_move_value(board)
        for v in cut:
            mv = chess.Move.from_uci(v.move_uci)
            if mv not in board.legal_moves:
                continue
            # a cut move must not be the teacher's best move
            assert eval_drop(board, mv, best=best) > 0, (
                f"guard cut {v.move_uci} which the teacher rates as best at {item.fen}"
            )


def test_guarded_model_reports_eval_drop() -> None:
    net = _net()
    corpus = generate_tactical_corpus(6, seed=99)
    result = evaluate_guarded_model(net, corpus, ConceptGuard(net), name="guard")
    assert result.legality == 1.0
    assert result.eval_drop == result.eval_drop  # not NaN
    assert result.eval_drop >= 0.0


def test_head_to_head_reports_the_decided_count() -> None:
    """0.5/0.5 from six draws must not read as a measurement."""
    from caissa2.selfplay.player import LatticePlayer

    net = _net()
    a = LatticePlayer(net, simulations=2, use_memory=False, seed=0, name="a")
    b = LatticePlayer(net, simulations=2, use_memory=False, seed=1, name="b")
    h2h = head_to_head(a, b, games=2, max_plies=6, seed=0)
    assert "decided" in h2h
    assert h2h["decided"] == h2h["wins_a"] + h2h["losses_a"]
    assert h2h["wins_a"] + h2h["losses_a"] + h2h["draws"] == h2h["games"]
    assert abs(h2h["score_a"] + h2h["score_b"] - 1.0) < 1e-9
