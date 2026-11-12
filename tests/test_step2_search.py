"""Step 2 — search over the field, and the guard as a cut with a reason in axis terms."""

from __future__ import annotations

import chess

from caissa2.guard.guard import ConceptGuard, static_exchange_eval, worst_material_loss
from caissa2.model.nets import OctaveNet
from caissa2.search.puct import PuctSearch


def _net() -> OctaveNet:
    return OctaveNet(seed=0, trunk_dim=32)


def test_see_finds_a_free_queen() -> None:
    # black queen on d5 is undefended; white rook on d1 can take it
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    mv = chess.Move.from_uci("d1d5")
    assert static_exchange_eval(board, mv) == 9


def test_guard_cuts_a_hanging_rook() -> None:
    # white rook a1, black queen a8, white king h1: Ra1-a2 walks onto the queen's file
    board = chess.Board("q6k/8/8/8/8/8/8/R6K w - - 0 1")
    guard = ConceptGuard(_net())
    bad = chess.Move.from_uci("h1h2")  # legal, but leaves the rook on a1 hanging to Qxa1
    verdict = guard.inspect(board, bad)
    assert verdict.material_loss >= 2
    assert not verdict.passed
    assert "material" in verdict.explain()


def test_guard_keeps_a_safe_move() -> None:
    board = chess.Board()
    guard = ConceptGuard(_net())
    kept, cut = guard.filter(board, list(board.legal_moves))
    assert kept, "the guard must never return an empty candidate list"
    assert all(mv in board.legal_moves for mv in kept)


def test_guard_never_empties_the_candidate_list() -> None:
    """The invariant, checked on real tactical positions where most moves are cut."""
    from caissa2.eval.corpus import generate_tactical_corpus

    guard = ConceptGuard(_net())
    for item in generate_tactical_corpus(6, seed=7):
        board = chess.Board(item.fen)
        moves = list(board.legal_moves)
        kept, cut = guard.filter(board, moves)
        assert len(kept) >= 1, f"guard emptied the list at {item.fen}"
        assert all(mv in board.legal_moves for mv in kept)


def test_search_returns_a_legal_move() -> None:
    board = chess.Board()
    search = PuctSearch(_net(), simulations=8, seed=0)
    mv = search.best_move(board)
    assert mv is not None and mv in board.legal_moves


def test_search_explains_discarded_branches_in_axis_terms() -> None:
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    search = PuctSearch(_net(), simulations=8, seed=0)
    report = search.explain(board)
    assert report["chosen"] in {m.uci() for m in board.legal_moves}
    assert isinstance(report["discarded"], list)
    for line in report["discarded"]:
        assert ":" in line
    for rec in report["discarded_axes"]:
        assert "move" in rec and "axes" in rec


def test_search_visits_are_a_distribution() -> None:
    board = chess.Board()
    search = PuctSearch(_net(), simulations=12, seed=0)
    dist = search.visit_policy(board, temperature=1.0)
    assert dist
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert all(p >= 0 for p in dist.values())


def test_search_does_not_hang_material() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    search = PuctSearch(_net(), simulations=16, seed=0)
    mv = search.best_move(board)
    assert mv is not None
    assert worst_material_loss(board, mv)[0] < 2
