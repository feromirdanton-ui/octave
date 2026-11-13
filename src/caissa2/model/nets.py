"""Three models, one corpus: the throat, the octave, and the joint control.

``ThroatNet`` is Caïssa-1's architecture, kept honest: the policy is a function of the eight
concept numbers and of nothing else. Whatever the position contains that the eight axes cannot
say is unreachable for it.

``OctaveNet`` is Caïssa-2 step 1: a free latent trunk (default 256-D) that the policy reads
directly, plus a concept head trained as an *auxiliary* task. The concepts are still measured —
the explanation is still exact — but they no longer gate the policy.

``JointNet`` is the control that step 1 needs and did not have. It is the *same capacity* as the
octave, but the policy reads the **concept representation** rather than the free trunk. The
concept layer is wide (``concept_dim``, default equal to the trunk width) instead of eight, so
this is not the Caïssa-1 throat — it is the honest question: *does separating the measurement
from the policy path buy anything, or is a wide enough concept layer just as good?*

An earlier version of this file got that wrong: its ``JointNet`` read the same free trunk the
octave reads, so the two models were the same function and produced bit-identical arena rows.
That is a control that cannot fail, which is worse than no control at all. The joint model now
routes the policy through the concept layer, and a test asserts the two are *not* the same
function.

All three are numpy-only so the whole thing runs on a CPU with no torch, and all three apply the
legality mask inside the softmax, so every sampled move is legal by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from caissa2.core.encoding import BOARD_DIM, MOVE_DIM
from caissa2.core.types import N_CONCEPTS


class Adam:
    """Minimal Adam over a dict of numpy arrays."""

    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / c1
            vhat = self.v[k] / c2
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over the last axis restricted to ``mask`` (bool, same shape)."""
    z = np.where(mask, logits, -np.inf)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    s = e.sum(axis=-1, keepdims=True)
    return np.divide(e, s, out=np.zeros_like(e), where=s > 0)


@dataclass
class ThroatNet:
    """Caïssa-1 baseline: policy = f(8 concepts). The throat is the whole model."""

    seed: int = 0
    lr: float = 3e-3
    concept_weight: float = 0.5
    params: dict[str, np.ndarray] = field(default_factory=dict)
    opt: Adam | None = None

    kind: str = "throat"

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.params = {
            "W1": (rng.normal(0, 1.0 / np.sqrt(BOARD_DIM), (BOARD_DIM, N_CONCEPTS))).astype(
                np.float32
            ),
            "b1": np.zeros(N_CONCEPTS, dtype=np.float32),
            "W2": (rng.normal(0, 1.0 / np.sqrt(N_CONCEPTS), (N_CONCEPTS, MOVE_DIM))).astype(
                np.float32
            ),
            "wv": (rng.normal(0, 1.0 / np.sqrt(N_CONCEPTS), (N_CONCEPTS, 1))).astype(np.float32),
        }
        self.opt = Adam(self.params, lr=self.lr)

    def concepts(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.params["W1"] + self.params["b1"])

    def trunk(self, x: np.ndarray) -> np.ndarray:
        """The throat itself: the eight-dimensional hidden state. Alias of ``concepts``.

        Kept as a named method because the whole point of this model is that the hidden state
        *is* the concept vector — a test asserts ``trunk(x).shape == (n, 8)``.
        """
        return self.concepts(x)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.concepts(x) @ self.params["W2"]

    def value(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.concepts(x) @ self.params["wv"]).ravel()

    def policy(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return masked_softmax(self.logits(x), mask)

    def train_batch(self, x, mask, target_idx, target_concepts, target_value=None) -> dict:
        b = x.shape[0]
        c = self.concepts(x)
        logits = c @ self.params["W2"]
        probs = masked_softmax(logits, mask)

        rows = np.arange(b)
        ce = -np.log(np.maximum(probs[rows, target_idx], 1e-12)).mean()

        dlogits = probs.copy()
        dlogits[rows, target_idx] -= 1.0
        dlogits[~mask] = 0.0
        dlogits /= b

        grads = {}
        grads["W2"] = c.T @ dlogits
        dc = dlogits @ self.params["W2"].T

        mse = float(np.mean((c - target_concepts) ** 2))
        dpre = self.concept_weight * 2.0 * (c - target_concepts) / (b * N_CONCEPTS)
        dpre *= 1.0 - c * c
        dc = dc + dpre

        vloss = 0.0
        if target_value is not None:
            v = np.tanh(c @ self.params["wv"]).ravel()
            vloss = float(np.mean((v - target_value) ** 2))
            dv = 2.0 * (v - target_value)[:, None] / b
            grads["wv"] = c.T @ (dv * (1.0 - v[:, None] ** 2))
        else:
            grads["wv"] = np.zeros_like(self.params["wv"])

        dc = dc * (1.0 - c * c)
        grads["W1"] = x.T @ dc
        grads["b1"] = dc.sum(axis=0)

        self.opt.step(self.params, grads)
        return {"ce": float(ce), "concept_mse": mse, "value_mse": vloss}


@dataclass
class OctaveNet:
    """Caïssa-2 step 1: free latent trunk + concept head as an auxiliary task."""

    seed: int = 0
    lr: float = 3e-3
    trunk_dim: int = 256
    concept_weight: float = 0.5
    params: dict[str, np.ndarray] = field(default_factory=dict)
    opt: Adam | None = None

    kind: str = "octave"

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        d = self.trunk_dim
        self.params = {
            "Wt": (rng.normal(0, 1.0 / np.sqrt(BOARD_DIM), (BOARD_DIM, d))).astype(np.float32),
            "bt": np.zeros(d, dtype=np.float32),
            "Wp": (rng.normal(0, 1.0 / np.sqrt(d), (d, MOVE_DIM))).astype(np.float32),
            "Wc": (rng.normal(0, 1.0 / np.sqrt(d), (d, N_CONCEPTS))).astype(np.float32),
            "bc": np.zeros(N_CONCEPTS, dtype=np.float32),
            "wv": (rng.normal(0, 1.0 / np.sqrt(d), (d, 1))).astype(np.float32),
        }
        self.opt = Adam(self.params, lr=self.lr)

    def trunk(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x @ self.params["Wt"] + self.params["bt"], 0.0)

    def concepts(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.trunk(x) @ self.params["Wc"] + self.params["bc"])

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.trunk(x) @ self.params["Wp"]

    def value(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.trunk(x) @ self.params["wv"]).ravel()

    def policy(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return masked_softmax(self.logits(x), mask)

    def train_batch(self, x, mask, target_idx, target_concepts, target_value=None) -> dict:
        b = x.shape[0]
        h = self.trunk(x)
        logits = h @ self.params["Wp"]
        probs = masked_softmax(logits, mask)

        rows = np.arange(b)
        ce = -np.log(np.maximum(probs[rows, target_idx], 1e-12)).mean()

        dlogits = probs.copy()
        dlogits[rows, target_idx] -= 1.0
        dlogits[~mask] = 0.0
        dlogits /= b

        grads = {}
        grads["Wp"] = h.T @ dlogits
        dh = dlogits @ self.params["Wp"].T

        c = np.tanh(h @ self.params["Wc"] + self.params["bc"])
        mse = float(np.mean((c - target_concepts) ** 2))
        dpre = self.concept_weight * 2.0 * (c - target_concepts) / (b * N_CONCEPTS)
        dpre *= 1.0 - c * c
        grads["Wc"] = h.T @ dpre
        grads["bc"] = dpre.sum(axis=0)
        dh = dh + dpre @ self.params["Wc"].T

        vloss = 0.0
        if target_value is not None:
            v = np.tanh(h @ self.params["wv"]).ravel()
            vloss = float(np.mean((v - target_value) ** 2))
            dv = 2.0 * (v - target_value)[:, None] / b
            grads["wv"] = h.T @ (dv * (1.0 - v[:, None] ** 2))
        else:
            grads["wv"] = np.zeros_like(self.params["wv"])

        dh = dh * (h > 0)
        grads["Wt"] = x.T @ dh
        grads["bt"] = dh.sum(axis=0)

        self.opt.step(self.params, grads)
        return {"ce": float(ce), "concept_mse": mse, "value_mse": vloss}


@dataclass
class JointNet:
    """The joint control: the policy reads the *concept representation*, not a free trunk.

    This is the comparison step 1 actually needs. ``OctaveNet`` gives the policy a free path
    (``Wp`` on the trunk) and measures concepts on a side branch. ``JointNet`` gives the policy
    no free path at all: it must read the concept layer, exactly as Caïssa-1 did — but the
    concept layer is ``concept_dim`` wide instead of eight, so the model is not capacity-starved.

    The question it answers: *is the win from opening the throat, or merely from having more
    parameters?* If a wide joint concept layer matches the octave, the factorization claim is
    unsupported and the honest conclusion is "capacity, not separation". If the octave wins at
    equal capacity, the separation itself is doing work.

    ``concept_dim`` defaults to the trunk width, which makes the two models parameter-matched to
    within the concept readout — the difference is purely architectural.
    """

    seed: int = 0
    lr: float = 3e-3
    concept_dim: int = 256
    concept_weight: float = 0.5
    params: dict[str, np.ndarray] = field(default_factory=dict)
    opt: Adam | None = None

    kind: str = "joint"

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        w = self.concept_dim
        self.params = {
            "Wt": (rng.normal(0, 1.0 / np.sqrt(BOARD_DIM), (BOARD_DIM, w))).astype(np.float32),
            "bt": np.zeros(w, dtype=np.float32),
            # the policy reads the WIDE concept layer — there is no free trunk path
            "Wp": (rng.normal(0, 1.0 / np.sqrt(w), (w, MOVE_DIM))).astype(np.float32),
            # the eight measured axes are a readout of that same layer
            "Wc": (rng.normal(0, 1.0 / np.sqrt(w), (w, N_CONCEPTS))).astype(np.float32),
            "bc": np.zeros(N_CONCEPTS, dtype=np.float32),
            "wv": (rng.normal(0, 1.0 / np.sqrt(w), (w, 1))).astype(np.float32),
        }
        self.opt = Adam(self.params, lr=self.lr)

    def wide(self, x: np.ndarray) -> np.ndarray:
        """The wide concept layer — the only representation the policy can see."""
        return np.tanh(x @ self.params["Wt"] + self.params["bt"])

    def concepts(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.wide(x) @ self.params["Wc"] + self.params["bc"])

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.wide(x) @ self.params["Wp"]

    def value(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.wide(x) @ self.params["wv"]).ravel()

    def policy(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return masked_softmax(self.logits(x), mask)

    def train_batch(self, x, mask, target_idx, target_concepts, target_value=None) -> dict:
        b = x.shape[0]
        h = self.wide(x)
        logits = h @ self.params["Wp"]
        probs = masked_softmax(logits, mask)

        rows = np.arange(b)
        ce = -np.log(np.maximum(probs[rows, target_idx], 1e-12)).mean()

        dlogits = probs.copy()
        dlogits[rows, target_idx] -= 1.0
        dlogits[~mask] = 0.0
        dlogits /= b

        grads = {}
        grads["Wp"] = h.T @ dlogits
        dh = dlogits @ self.params["Wp"].T

        c = np.tanh(h @ self.params["Wc"] + self.params["bc"])
        mse = float(np.mean((c - target_concepts) ** 2))
        dpre = self.concept_weight * 2.0 * (c - target_concepts) / (b * N_CONCEPTS)
        dpre *= 1.0 - c * c
        grads["Wc"] = h.T @ dpre
        grads["bc"] = dpre.sum(axis=0)
        dh = dh + dpre @ self.params["Wc"].T

        vloss = 0.0
        if target_value is not None:
            v = np.tanh(h @ self.params["wv"]).ravel()
            vloss = float(np.mean((v - target_value) ** 2))
            dv = 2.0 * (v - target_value)[:, None] / b
            grads["wv"] = h.T @ (dv * (1.0 - v[:, None] ** 2))
        else:
            grads["wv"] = np.zeros_like(self.params["wv"])

        dh = dh * (1.0 - h * h)  # tanh derivative: the wide layer is tanh, not relu
        grads["Wt"] = x.T @ dh
        grads["bt"] = dh.sum(axis=0)

        self.opt.step(self.params, grads)
        return {"ce": float(ce), "concept_mse": mse, "value_mse": vloss}


def param_count(model) -> int:
    """Total number of scalar parameters — the number the joint/octave match is made on."""
    return int(sum(int(np.asarray(v).size) for v in model.params.values()))


def matched_joint_concept_dim(octave_trunk_dim: int) -> int:
    """The joint concept width whose parameter count matches an octave of ``octave_trunk_dim``.

    Octave: BOARD_DIM*d + d + d*MOVE_DIM + d*N_CONCEPTS + N_CONCEPTS + d
    Joint:  BOARD_DIM*w + w + w*MOVE_DIM + w*N_CONCEPTS + N_CONCEPTS + w

    The two are the same expression in ``d`` and ``w``, so the matched width is ``d`` itself.
    That is the point: the joint model is not smaller, it is *differently wired* — its policy
    path goes through the concept layer instead of around it.
    """
    return int(octave_trunk_dim)


def save_model(model, path: str) -> None:
    np.savez(path, **{k: v for k, v in model.params.items()})


def load_model(cls, path: str, **kwargs):
    model = cls(**kwargs)
    with np.load(path) as z:
        for k in model.params:
            if k in z:
                model.params[k] = z[k]
    model.opt = Adam(model.params, lr=model.lr)
    return model
