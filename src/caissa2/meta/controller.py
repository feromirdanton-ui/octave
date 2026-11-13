"""Metacognition — knowing when you are about to be wrong, and paying only then.

The arena showed a saturation problem: the full lattice player already scores 0.000 blunder rate
on the tactical corpus, so *more* search cannot show up as a better number. The only remaining
axis is cost. A metacognitive controller answers one question per position — "how likely is my
own fast answer to be a mistake?" — and spends search only where the answer is "likely".

Two pieces:

``ErrorPredictor``
    A logistic model over cheap, pre-search features (policy entropy, top-1 margin, concept
    spread, material, legal-move count, guard cut fraction). It is trained on labels the arena
    can produce for free: whether the bare policy's top-1 move gives up >= 1 pawn of teacher
    evaluation. That label is *not* the guard's material check, so the predictor is not learning
    the guard's own definition.

``MetacognitiveController``
    Maps the predicted risk to a search budget from a fixed tier set. The claim it has to earn is
    the one from the paper: equal or better safety at a fraction of the simulations.

The honest metric is ``auroc``: 0.5 means the predictor is a coin flip and the whole routing idea
is dead for this feature set. We report it rather than assume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from caissa2.core.concepts import extract_ground_truth_concepts
from caissa2.core.encoding import legal_move_mask, move_to_index
from caissa2.model.features import encode_batch


def auroc(labels, scores) -> float:
    """Area under the ROC curve by rank statistics (ties get average rank).

    Implemented directly rather than through sklearn so the number is reproducible with numpy
    alone. Returns ``nan`` when only one class is present — a corpus with no errors cannot
    measure a predictor of errors, and silently returning 0.5 would hide that.
    """
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_calibration_error(labels, probs, bins: int = 10) -> float:
    """Mean |predicted - observed| over equal-width probability bins."""
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi if b < bins - 1 else p <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


FEATURE_NAMES: tuple[str, ...] = (
    "policy_entropy",
    "top1_prob",
    "margin",
    "n_legal",
    "concept_spread",
    "material",
    "abs_value",
    "guard_cut_fraction",
)


def position_features(board: chess.Board, model, guard=None) -> np.ndarray:
    """Cheap pre-search features. Nothing here runs a search or a SEE sweep."""
    x = encode_batch([board])
    m, _ = legal_move_mask(board)
    mask = np.asarray(m, dtype=bool)[None, :]
    probs = model.policy(x, mask)[0]
    legal = np.sort(probs[probs > 0])[::-1]
    p1 = float(legal[0]) if len(legal) else 0.0
    p2 = float(legal[1]) if len(legal) > 1 else 0.0
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(np.maximum(nz, 1e-12))).sum())
    n_legal = int(mask.sum())
    entropy_norm = entropy / max(np.log(max(n_legal, 2)), 1e-6)

    c = extract_ground_truth_concepts(board).as_array()
    spread = float(np.std(c))
    mat = float(np.tanh(_material(board) / 10.0))
    val = float(abs(model.value(x)[0]))

    cut_frac = 0.0
    if guard is not None:
        from caissa2.eval.arena import top_k_legal_moves

        cands = top_k_legal_moves(board, probs, k=12)
        if cands:
            kept, cut = guard.filter(board, cands)
            cut_frac = len(cut) / max(len(cands), 1)

    return np.asarray(
        [entropy_norm, p1, p1 - p2, n_legal / 40.0, spread, mat, val, cut_frac],
        dtype=float,
    )


def _material(board: chess.Board) -> float:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    us, them = board.turn, not board.turn
    mine = sum(len(board.pieces(pt, us)) * v for pt, v in vals.items())
    theirs = sum(len(board.pieces(pt, them)) * v for pt, v in vals.items())
    return float(mine - theirs)


@dataclass
class ErrorPredictor:
    """Logistic regression: P(the fast answer is a mistake) from pre-search features."""

    lr: float = 0.5
    epochs: int = 400
    l2: float = 1e-3
    w: np.ndarray | None = None
    b: float = 0.0
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ErrorPredictor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-9
        Z = (X - self.mean) / self.std
        n, d = Z.shape
        w = np.zeros(d)
        b = float(np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6)))
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
            g = p - y
            w -= self.lr * (Z.T @ g / n + self.l2 * w)
            b -= self.lr * float(g.mean())
        self.w, self.b = w, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("ErrorPredictor.fit() must be called first")
        Z = (np.asarray(X, dtype=float) - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))

    def weights(self) -> dict[str, float]:
        if self.w is None:
            return {}
        return {name: round(float(v), 4) for name, v in zip(FEATURE_NAMES, self.w)}


@dataclass
class MetacognitiveController:
    """Routes a search budget from predicted risk.

    ``tiers`` are the only budgets the controller may spend. A tiered controller is honest about
    what it is: it cannot invent a budget, and the cost of a position is one of three numbers, so
    the FLOP accounting is exact rather than an average over a continuum.
    """

    predictor: ErrorPredictor
    tiers: tuple[int, ...] = (4, 16, 64)
    thresholds: tuple[float, ...] = (0.25, 0.6)
    guard: object | None = None

    def risk(self, board: chess.Board, model) -> float:
        f = position_features(board, model, guard=self.guard)
        return float(self.predictor.predict_proba(f[None, :])[0])

    def budget(self, board: chess.Board, model) -> tuple[int, float]:
        p = self.risk(board, model)
        for t, thr in zip(self.tiers, self.thresholds):
            if p < thr:
                return int(t), p
        return int(self.tiers[-1]), p


def build_training_set(model, corpus, guard=None, error_threshold: float = 1.0) -> tuple:
    """Features + labels: did the bare policy's top-1 give up >= ``error_threshold`` pawns?

    The label comes from the teacher's evaluation, which the guard never reads, so a predictor
    trained on it is measuring something the guard does not already know.
    """
    from caissa2.eval.arena import eval_drop, top_k_legal_moves

    X, y = [], []
    for item in corpus:
        board = chess.Board(item.fen)
        X.append(position_features(board, model, guard=guard))
        probs = model.policy(encode_batch([board]), np.asarray(legal_move_mask(board)[0])[None, :])[0]
        cands = top_k_legal_moves(board, probs, k=1)
        if not cands:
            y.append(0.0)
            continue
        d = eval_drop(board, cands[0])
        y.append(1.0 if (d == d and d >= error_threshold) else 0.0)
    return np.stack(X), np.asarray(y, dtype=float)


def evaluate_routing(controller, model, corpus, player_factory, fixed_budget: int) -> dict:
    """Compare a fixed budget against the routed one on the same positions.

    Reports safety (blunder rate, eval drop) and cost (mean simulations). The claim worth making
    is "same safety, fewer simulations"; if safety drops, the routing is not free and must be
    reported as a trade rather than a win.
    """
    from caissa2.eval.arena import blunder_loss, eval_drop

    routed_blunders = fixed_blunders = 0
    routed_drops, fixed_drops = [], []
    routed_sims = fixed_sims = 0
    n = 0

    for item in corpus:
        board = chess.Board(item.fen)
        n += 1

        sims, _ = controller.budget(board, model)
        routed_sims += sims
        p_r = player_factory(sims)
        mv_r = p_r.choose(board, temperature=0.0)
        if mv_r is not None and mv_r in board.legal_moves:
            if blunder_loss(board, mv_r) >= 2:
                routed_blunders += 1
            routed_drops.append(eval_drop(board, mv_r))

        fixed_sims += fixed_budget
        p_f = player_factory(fixed_budget)
        mv_f = p_f.choose(board, temperature=0.0)
        if mv_f is not None and mv_f in board.legal_moves:
            if blunder_loss(board, mv_f) >= 2:
                fixed_blunders += 1
            fixed_drops.append(eval_drop(board, mv_f))

    return {
        "n": n,
        "routed": {
            "blunder_rate": round(routed_blunders / max(n, 1), 4),
            "eval_drop": round(float(np.mean(routed_drops)), 4) if routed_drops else None,
            "mean_simulations": round(routed_sims / max(n, 1), 2),
        },
        "fixed": {
            "blunder_rate": round(fixed_blunders / max(n, 1), 4),
            "eval_drop": round(float(np.mean(fixed_drops)), 4) if fixed_drops else None,
            "mean_simulations": round(fixed_sims / max(n, 1), 2),
        },
        "simulation_saving": round(1.0 - routed_sims / max(fixed_sims, 1), 4),
    }
