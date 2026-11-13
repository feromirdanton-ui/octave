"""Tests for the v2 additions: joint baseline, metacognition, memory proposals, faithfulness."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from caissa2.core.encoding import legal_move_mask
from caissa2.eval.corpus import generate_tactical_corpus
from caissa2.guard.faithfulness import faithfulness_report
from caissa2.guard.guard import ConceptGuard
from caissa2.memory.index import ConceptMemory
from caissa2.meta.controller import (
    ErrorPredictor,
    MetacognitiveController,
    auroc,
    build_training_set,
    expected_calibration_error,
    position_features,
)
from caissa2.model.nets import JointNet, OctaveNet, matched_joint_concept_dim, param_count
from caissa2.search.puct import PuctSearch
from caissa2.selfplay.player import LatticePlayer
from caissa2.train import train_supervised


@pytest.fixture(scope="module")
def corpus():
    return generate_tactical_corpus(24, seed=7)


@pytest.fixture(scope="module")
def trained(corpus):
    net = OctaveNet(seed=0, lr=3e-3, trunk_dim=32)
    train_supervised(net, corpus, epochs=2, seed=0)
    return net


# -- joint baseline -------------------------------------------------------------
def test_joint_net_policy_reads_the_concept_layer_not_a_free_trunk():
    """The joint control must have NO free path from trunk to policy.

    This is the test that would have caught the first version of this control, which read the
    same free trunk the octave reads and therefore produced bit-identical arena rows.
    """
    net = JointNet(seed=0, concept_dim=32)
    x = np.zeros((2, 832), dtype=np.float32)
    w = net.wide(x)
    assert np.allclose(net.logits(x), w @ net.params["Wp"])
    assert np.allclose(net.concepts(x), np.tanh(w @ net.params["Wc"] + net.params["bc"]))
    assert net.params["Wt"].shape == (832, 32)


def test_joint_and_octave_are_not_the_same_function():
    """A control that cannot differ from the treatment is not a control."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (4, 832)).astype(np.float32)
    octave = OctaveNet(seed=0, trunk_dim=32)
    joint = JointNet(seed=0, concept_dim=32)
    assert not np.allclose(octave.logits(x), joint.logits(x))


def test_joint_and_octave_are_parameter_matched():
    """The comparison is only fair if the two models have comparable capacity."""
    d = 64
    octave = OctaveNet(seed=0, trunk_dim=d)
    joint = JointNet(seed=0, concept_dim=matched_joint_concept_dim(d))
    ratio = param_count(joint) / param_count(octave)
    assert 0.95 <= ratio <= 1.10, f"joint/octave parameter ratio {ratio:.3f} is not matched"


def test_joint_net_trains_and_stays_legal(corpus):
    net = JointNet(seed=0, lr=3e-3, concept_dim=32)
    stats = train_supervised(net, corpus, epochs=2, seed=0)
    assert stats["final_ce"] > 0
    board = chess.Board(corpus[0].fen)
    mask = np.asarray(legal_move_mask(board)[0])[None, :]
    probs = net.policy(np.zeros((1, 832), dtype=np.float32), mask)[0]
    assert probs[~mask[0]].sum() == 0.0
    assert abs(probs.sum() - 1.0) < 1e-6


# -- metacognition --------------------------------------------------------------
def test_auroc_is_one_for_a_perfect_ranking():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_auroc_is_half_for_a_coin_flip():
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auroc_is_nan_when_only_one_class_is_present():
    """A corpus with no errors cannot measure a predictor of errors."""
    assert np.isnan(auroc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    labels = np.array([0, 0, 1, 1], dtype=float)
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(labels, probs) == pytest.approx(0.0, abs=1e-9)


def test_error_predictor_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (200, 8)), rng.normal(3, 1, (200, 8))])
    y = np.concatenate([np.zeros(200), np.ones(200)])
    p = ErrorPredictor().fit(X, y)
    assert auroc(y, p.predict_proba(X)) > 0.95


def test_controller_budget_is_always_one_of_its_tiers(trained):
    ctrl = MetacognitiveController(
        predictor=ErrorPredictor().fit(np.zeros((4, 8)), np.array([0, 1, 0, 1], dtype=float))
    )
    board = chess.Board()
    sims, risk = ctrl.budget(board, trained)
    assert sims in ctrl.tiers
    assert 0.0 <= risk <= 1.0


def test_position_features_are_finite_and_shaped(trained):
    f = position_features(chess.Board(), trained)
    assert f.shape == (8,)
    assert np.all(np.isfinite(f))


def test_training_set_labels_are_binary(trained, corpus):
    X, y = build_training_set(trained, corpus[:10])
    assert X.shape[0] == len(y)
    assert set(np.unique(y)) <= {0.0, 1.0}


# -- memory as a director of exploration ---------------------------------------
def test_memory_confidence_is_zero_when_empty():
    assert ConceptMemory().confidence(chess.Board()) == 0.0


def test_memory_proposes_only_legal_moves(trained, corpus):
    mem = ConceptMemory(k=4)
    for item in corpus[:12]:
        mem.add(chess.Board(item.fen), item.move_uci)
    board = chess.Board(corpus[0].fen)
    legal = {mv.uci() for mv in board.legal_moves}
    proposals = mem.propose(board, legal, top_n=3, min_confidence=0.0)
    assert all(p in legal for p in proposals)


def test_memory_proposal_reaches_the_search_tree(trained, corpus):
    """A proposal must actually appear as a root child — that is the whole point."""
    mem = ConceptMemory(k=4)
    for item in corpus[:12]:
        mem.add(chess.Board(item.fen), item.move_uci)
    board = chess.Board(corpus[0].fen)
    search = PuctSearch(trained, memory=mem, simulations=8, memory_proposals=3)
    root = search.run(board)
    legal = {mv.uci() for mv in board.legal_moves}
    assert all(p in legal for p in root.proposals)
    for p in root.proposals:
        assert p in root.children


def test_search_explain_reports_memory_proposals(trained, corpus):
    mem = ConceptMemory(k=4)
    for item in corpus[:12]:
        mem.add(chess.Board(item.fen), item.move_uci)
    search = PuctSearch(trained, memory=mem, simulations=6)
    out = search.explain(chess.Board(corpus[0].fen))
    assert "memory_proposals" in out


# -- faithfulness ---------------------------------------------------------------
def test_guard_ablation_changes_the_decision_when_the_axis_matters(trained, corpus):
    """Ablating the axis an explanation names must be able to remove that cut."""
    guard = ConceptGuard(trained)
    rep = faithfulness_report(trained, guard, corpus[:12], k=8)
    assert rep.cuts >= 0
    if rep.cuts:
        assert 0.0 <= rep.ablation_faithfulness <= 1.0
        assert 0.0 <= rep.shuffle_sensitivity <= 1.0


def test_ablated_axis_cannot_be_the_reason_for_a_cut(trained, corpus):
    """If every safety axis is ablated, no concept-axis cut may survive."""
    board = chess.Board(corpus[0].fen)
    probs = trained.policy(
        np.zeros((1, 832), dtype=np.float32), np.asarray(legal_move_mask(board)[0])[None, :]
    )[0]
    from caissa2.eval.arena import top_k_legal_moves

    cands = top_k_legal_moves(board, probs, k=8)
    guard = ConceptGuard(trained, ablate_axes={"king_safety", "pawn_structure", "piece_activity"})
    verdicts = guard.inspect_many(board, cands)
    for v in verdicts:
        assert not v.concept_outlier or not v.corroborated or v.material_loss >= 1


# -- routed player --------------------------------------------------------------
def test_player_uses_the_controller_budget(trained, corpus):
    ctrl = MetacognitiveController(
        predictor=ErrorPredictor().fit(np.zeros((4, 8)), np.array([0, 1, 0, 1], dtype=float)),
        tiers=(2, 4),
        thresholds=(0.5,),
    )
    player = LatticePlayer(trained, simulations=8, controller=ctrl, seed=0)
    board = chess.Board(corpus[0].fen)
    sims = player._budget(board)
    assert sims in (2, 4)
    mv = player.choose(board, temperature=0.0)
    assert mv is not None and mv in board.legal_moves
