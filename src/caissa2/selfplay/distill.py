"""Step 3 — the loop: field -> player -> games -> net -> better field.

Distillation here is *not* imitation of human moves. The target is the lattice player's own
distribution over moves (search visits, guard-filtered, memory-primed). The network is trained
on that distribution with a cross-entropy over the legal-move mask, plus the concept head's
auxiliary loss against the ground-truth extractor.

One round is: play N games with the current player, collect (position, player distribution)
pairs, train the network on them, rebuild the player on the new network. Repeat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from caissa2.core.concepts import extract_ground_truth_concepts
from caissa2.core.encoding import legal_move_mask, move_to_index
from caissa2.model.features import encode_batch
from caissa2.selfplay.player import LatticePlayer, play_game


@dataclass
class DistillSample:
    fen: str
    target: np.ndarray  # 4096-D distribution over moves (0 outside the legal mask)
    concepts: np.ndarray
    value: float = 0.0


def samples_from_game(rec, result: float | None = None) -> list[DistillSample]:
    """Turn a game record into distillation samples with the player's own distribution."""
    out: list[DistillSample] = []
    res = rec.result if result is None else result
    for s in rec.samples:
        board = chess.Board(s["fen"])
        target = np.zeros(64 * 64, dtype=np.float32)
        for uci, p in s["dist"].items():
            mv = chess.Move.from_uci(uci)
            target[move_to_index(mv)] += float(p)
        total = target.sum()
        if total > 0:
            target /= total
        # value from the perspective of the side to move in that position
        sign = 1.0 if s["turn"] == chess.WHITE else -1.0
        out.append(
            DistillSample(
                fen=s["fen"],
                target=target,
                concepts=extract_ground_truth_concepts(board).as_array(),
                value=sign * res,
            )
        )
    return out


def train_on_samples(
    model,
    samples: list[DistillSample],
    epochs: int = 3,
    batch_size: int = 32,
    lr: float | None = None,
    seed: int = 0,
) -> dict:
    """Cross-entropy against the player's distribution + concept aux loss + value loss."""
    if not samples:
        return {"ce": 0.0, "concept_mse": 0.0, "value_mse": 0.0, "n": 0}
    if lr is not None:
        model.opt.lr = lr
    rng = np.random.default_rng(seed)
    boards = [chess.Board(s.fen) for s in samples]
    x = encode_batch(boards)
    mask = np.zeros((len(boards), 64 * 64), dtype=bool)
    for i, b in enumerate(boards):
        m, _ = legal_move_mask(b)
        mask[i] = np.asarray(m, dtype=bool)
    targets = np.stack([s.target for s in samples])
    concepts = np.stack([s.concepts for s in samples])
    values = np.asarray([s.value for s in samples], dtype=np.float32)

    hist = {"ce": [], "concept_mse": [], "value_mse": []}
    n = len(samples)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            # soft-target cross-entropy: the model's train_batch takes a hard index, so we
            # implement the soft version here through the same parameter dict.
            stats = _soft_step(model, x[idx], mask[idx], targets[idx], concepts[idx], values[idx])
            for k in hist:
                hist[k].append(stats[k])
    return {
        "ce": float(np.mean(hist["ce"])) if hist["ce"] else 0.0,
        "concept_mse": float(np.mean(hist["concept_mse"])) if hist["concept_mse"] else 0.0,
        "value_mse": float(np.mean(hist["value_mse"])) if hist["value_mse"] else 0.0,
        "n": n,
    }


def _soft_step(model, x, mask, target, concepts, values) -> dict:
    """One Adam step on soft-target CE + concept aux + value, for either net kind."""
    from caissa2.model.nets import masked_softmax

    b = x.shape[0]
    is_octave = getattr(model, "kind", "") == "octave"
    p = model.params

    if is_octave:
        h = model.trunk(x)
        logits = h @ p["Wp"]
    else:
        h = model.concepts(x)
        logits = h @ p["W2"]

    probs = masked_softmax(logits, mask)
    ce = float(-np.sum(target * np.log(np.maximum(probs, 1e-12))) / b)

    dlogits = (probs - target) / b
    dlogits[~mask] = 0.0

    grads = {}
    if is_octave:
        grads["Wp"] = h.T @ dlogits
        dh = dlogits @ p["Wp"].T
    else:
        grads["W2"] = h.T @ dlogits
        dh = dlogits @ p["W2"].T

    c = model.concepts(x)
    mse = float(np.mean((c - concepts) ** 2))
    dpre = model.concept_weight * 2.0 * (c - concepts) / (b * c.shape[1])
    dpre *= 1.0 - c * c

    if is_octave:
        grads["Wc"] = h.T @ dpre
        grads["bc"] = dpre.sum(axis=0)
        dh = dh + dpre @ p["Wc"].T
    else:
        dh = dh + dpre

    v = np.tanh(h @ p["wv"]).ravel()
    vloss = float(np.mean((v - values) ** 2))
    dv = 2.0 * (v - values)[:, None] / b
    grads["wv"] = h.T @ (dv * (1.0 - v[:, None] ** 2))

    if is_octave:
        dh = dh * (h > 0)
        grads["Wt"] = x.T @ dh
        grads["bt"] = dh.sum(axis=0)
    else:
        grads["W1"] = x.T @ dh
        grads["b1"] = dh.sum(axis=0)

    model.opt.step(p, grads)
    return {"ce": ce, "concept_mse": mse, "value_mse": vloss}


@dataclass
class LoopReport:
    rounds: list[dict] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.rounds.append(kw)


def run_loop(
    model,
    rounds: int = 2,
    games_per_round: int = 4,
    simulations: int = 16,
    max_plies: int = 60,
    epochs: int = 2,
    seed: int = 0,
    memory=None,
) -> LoopReport:
    """The closed loop: field -> player -> games -> net -> better field."""
    report = LoopReport()
    for r in range(rounds):
        player = LatticePlayer(
            model,
            simulations=simulations,
            memory=memory,
            seed=seed + r,
            name=f"lattice-r{r}",
        )
        samples: list[DistillSample] = []
        results = []
        for g in range(games_per_round):
            rec = play_game(player, player, max_plies=max_plies, temperature=0.7)
            results.append(rec.result)
            samples.extend(samples_from_game(rec))
        stats = train_on_samples(model, samples, epochs=epochs, seed=seed + r)
        report.add(
            round=r,
            games=games_per_round,
            samples=len(samples),
            mean_result=float(np.mean(results)) if results else 0.0,
            **stats,
        )
    return report
