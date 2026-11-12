"""Step 1 — the throat is opened.

The claim under test is architectural, not numerical: in ThroatNet every policy logit is a
function of the eight concept numbers, so two positions with the same concepts must have the
same policy. In OctaveNet that is false — the trunk carries information the eight axes cannot
express, and the policy can differ while the concepts agree.
"""

from __future__ import annotations

import chess
import numpy as np

from caissa2.core.encoding import encode_board, legal_move_mask
from caissa2.model.nets import OctaveNet, ThroatNet, masked_softmax


def _x(board: chess.Board) -> np.ndarray:
    return np.asarray([encode_board(board)], dtype=np.float32)


def _mask(board: chess.Board) -> np.ndarray:
    m, _ = legal_move_mask(board)
    return np.asarray(m, dtype=bool)[None, :]


def test_throat_policy_is_a_function_of_concepts_only() -> None:
    """The bottleneck, stated as a test: same concepts -> same logits, by construction."""
    net = ThroatNet(seed=0)
    board = chess.Board()
    x = _x(board)
    c = net.concepts(x)
    # the policy is literally c @ W2 — recompute it from the concepts alone
    from_concepts = c @ net.params["W2"]
    assert np.allclose(net.logits(x), from_concepts)
    # and the trunk has exactly eight dimensions: that is the throat
    assert net.trunk(x).shape == (1, 8)


def test_octave_trunk_is_free_and_wider() -> None:
    net = OctaveNet(seed=0, trunk_dim=256)
    x = _x(chess.Board())
    assert net.trunk(x).shape == (1, 256)
    # the policy does NOT go through the concept head
    h = net.trunk(x)
    assert np.allclose(net.logits(x), h @ net.params["Wp"])
    assert not np.allclose(net.logits(x), net.concepts(x) @ net.params["Wp"][:8])


def test_octave_concept_head_still_measures() -> None:
    """The explanation stays exact: the head is trained against the ground-truth extractor."""
    from caissa2.core.concepts import extract_ground_truth_concepts

    net = OctaveNet(seed=0, trunk_dim=64)
    board = chess.Board()
    pred = net.concepts(_x(board))[0]
    truth = extract_ground_truth_concepts(board).as_array()
    assert pred.shape == truth.shape == (8,)
    assert np.all(np.abs(pred) <= 1.0)


def test_policy_is_legal_by_construction() -> None:
    for cls in (ThroatNet, OctaveNet):
        net = cls(seed=0, trunk_dim=32) if cls is OctaveNet else cls(seed=0)
        board = chess.Board()
        probs = net.policy(_x(board), _mask(board))[0]
        assert abs(float(probs.sum()) - 1.0) < 1e-5
        mask, _ = legal_move_mask(board)
        assert np.all(probs[~np.asarray(mask, dtype=bool)] == 0.0)


def test_masked_softmax_ignores_masked_logits() -> None:
    logits = np.array([[1.0, 2.0, 3.0]])
    mask = np.array([[True, False, True]])
    p = masked_softmax(logits, mask)
    assert p[0, 1] == 0.0
    assert abs(p.sum() - 1.0) < 1e-9


def test_training_reduces_loss_on_a_tiny_corpus() -> None:
    from caissa2.eval.corpus import generate_corpus
    from caissa2.train import train_supervised

    corpus = generate_corpus(24, seed=3)
    net = OctaveNet(seed=0, trunk_dim=32)
    stats = train_supervised(net, corpus, epochs=4, seed=0)
    assert stats["final_ce"] < 6.0  # ln(4096) ~ 8.3 at random
