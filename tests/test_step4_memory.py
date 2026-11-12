"""Step 4 — memory as geometry: a 9-float concept index and a retrieval prior."""

from __future__ import annotations

import chess
import numpy as np

from caissa2.memory.index import INDEX_DIM, ConceptMemory, material_balance, position_key


def test_index_is_nine_floats() -> None:
    key = position_key(chess.Board())
    assert key.shape == (INDEX_DIM,) == (9,)
    assert np.all(np.isfinite(key))


def test_material_balance_is_bounded_and_signed() -> None:
    assert abs(material_balance(chess.Board())) < 1e-9
    up = chess.Board("4k3/8/8/8/8/8/8/QQ2K3 w - - 0 1")
    assert material_balance(up) > 0.5


def test_memory_retrieves_the_move_it_was_taught() -> None:
    mem = ConceptMemory(k=4)
    board = chess.Board()
    mem.add(board, chess.Move.from_uci("e2e4"))
    prior = mem.retrieval_prior(board)
    assert prior
    assert max(prior, key=prior.get) == "e2e4"


def test_retrieval_prior_is_a_distribution() -> None:
    mem = ConceptMemory(k=4)
    board = chess.Board()
    mem.add(board, "e2e4")
    mem.add(board, "d2d4")
    prior = mem.retrieval_prior(board)
    assert abs(sum(prior.values()) - 1.0) < 1e-6


def test_memory_generalises_across_positions_that_look_alike() -> None:
    """The point of a concept index: a *different* position can retrieve the same lesson."""
    mem = ConceptMemory(k=4, temperature=0.5)
    a = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    mem.add(a, "f3g5")
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 2")
    prior = mem.retrieval_prior(b)
    assert prior, "a near-identical position must retrieve something"
    assert max(prior, key=prior.get) == "f3g5"


def test_empty_memory_returns_no_prior() -> None:
    assert ConceptMemory().retrieval_prior(chess.Board()) == {}


def test_memory_feeds_the_search_prior() -> None:
    from caissa2.model.nets import OctaveNet
    from caissa2.search.puct import PuctSearch

    mem = ConceptMemory(k=4)
    board = chess.Board()
    mem.add(board, "e2e4")
    net = OctaveNet(seed=0, trunk_dim=32)
    with_mem = PuctSearch(net, simulations=8, memory=mem, seed=0, memory_weight=0.9)
    without = PuctSearch(net, simulations=8, memory=None, seed=0)
    p_with, _ = with_mem._prior(board, 0)
    p_without, _ = without._prior(board, 0)
    assert p_with["e2e4"] > p_without["e2e4"]
