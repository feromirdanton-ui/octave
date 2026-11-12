"""Supervised training on the teacher corpus — the same recipe for both architectures.

The point of this module is *fairness*: ThroatNet and OctaveNet see the same positions, the same
targets, the same number of epochs and the same optimiser. Any difference in the arena is a
difference of architecture, not of training budget.
"""

from __future__ import annotations

import numpy as np

from caissa2.core.encoding import legal_move_mask, move_to_index
from caissa2.eval.corpus import CorpusItem
from caissa2.model.features import encode_batch
import chess


def train_supervised(
    model,
    corpus: list[CorpusItem],
    epochs: int = 6,
    batch_size: int = 32,
    seed: int = 0,
    lr: float | None = None,
) -> dict:
    """Cross-entropy on the teacher move + concept aux loss, over the whole corpus."""
    if lr is not None:
        model.opt.lr = lr
    rng = np.random.default_rng(seed)
    boards = [chess.Board(item.fen) for item in corpus]
    x = encode_batch(boards)
    mask = np.zeros((len(boards), 64 * 64), dtype=bool)
    for i, b in enumerate(boards):
        m, _ = legal_move_mask(b)
        mask[i] = np.asarray(m, dtype=bool)
    target_idx = np.asarray(
        [move_to_index(chess.Move.from_uci(item.move_uci)) for item in corpus], dtype=np.int64
    )
    concepts = np.stack([item.concepts for item in corpus])

    hist = {"ce": [], "concept_mse": []}
    n = len(corpus)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            stats = model.train_batch(x[idx], mask[idx], target_idx[idx], concepts[idx])
            hist["ce"].append(stats["ce"])
            hist["concept_mse"].append(stats["concept_mse"])
    return {
        "epochs": epochs,
        "n": n,
        "final_ce": float(np.mean(hist["ce"][-max(1, n // batch_size) :])),
        "final_concept_mse": float(np.mean(hist["concept_mse"][-max(1, n // batch_size) :])),
    }
