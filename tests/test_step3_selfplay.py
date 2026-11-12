"""Step 3 — self-play distillation: the network imitates the lattice player, not humans."""

from __future__ import annotations

import chess

from caissa2.model.nets import OctaveNet
from caissa2.selfplay.distill import run_loop, samples_from_game, train_on_samples
from caissa2.selfplay.player import LatticePlayer, play_game


def test_player_chooses_legal_moves() -> None:
    player = LatticePlayer(OctaveNet(seed=0, trunk_dim=32), simulations=4, seed=0)
    board = chess.Board()
    for _ in range(6):
        mv = player.choose(board, temperature=0.0)
        assert mv is not None and mv in board.legal_moves
        board.push(mv)


def test_player_distribution_is_a_distribution() -> None:
    player = LatticePlayer(OctaveNet(seed=0, trunk_dim=32), simulations=4, seed=0)
    dist = player.policy_distribution(chess.Board(), temperature=1.0)
    assert dist
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_game_record_has_samples_and_a_result() -> None:
    player = LatticePlayer(OctaveNet(seed=0, trunk_dim=32), simulations=4, seed=0)
    rec = play_game(player, player, max_plies=8, temperature=0.8)
    assert rec.plies > 0
    assert len(rec.samples) == rec.plies
    assert rec.result in (-1.0, 0.0, 1.0)
    for s in rec.samples:
        assert "fen" in s and "dist" in s and "move_uci" in s


def test_distillation_target_is_the_players_own_distribution() -> None:
    """The target is the player's distribution, not a human move — that is the whole point."""
    player = LatticePlayer(OctaveNet(seed=0, trunk_dim=32), simulations=4, seed=0)
    rec = play_game(player, player, max_plies=4, temperature=0.9)
    samples = samples_from_game(rec)
    assert samples
    for s in samples:
        assert s.target.shape == (4096,)
        assert abs(float(s.target.sum()) - 1.0) < 1e-5
        assert int((s.target > 0).sum()) >= 1


def test_training_on_samples_reduces_cross_entropy() -> None:
    player = LatticePlayer(OctaveNet(seed=0, trunk_dim=32), simulations=4, seed=0)
    rec = play_game(player, player, max_plies=10, temperature=0.9)
    samples = samples_from_game(rec)
    net = OctaveNet(seed=1, trunk_dim=32)
    first = train_on_samples(net, samples, epochs=1, seed=0)
    second = train_on_samples(net, samples, epochs=4, seed=0)
    assert second["ce"] <= first["ce"] + 1e-6


def test_loop_runs_and_reports_each_round() -> None:
    net = OctaveNet(seed=0, trunk_dim=32)
    report = run_loop(net, rounds=1, games_per_round=1, simulations=4, max_plies=6, epochs=1)
    assert len(report.rounds) == 1
    r = report.rounds[0]
    assert r["games"] == 1
    assert r["samples"] > 0
    assert "ce" in r and "concept_mse" in r
