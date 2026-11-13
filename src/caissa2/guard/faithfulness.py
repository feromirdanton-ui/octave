"""Faithfulness of the guard's explanations — the test we never ran.

The guard says "I cut this move because ``king_safety`` dropped by 0.31, an outlier against its
siblings". That sentence is only worth something if it is *causally* true of the decision. Two
standard tests, both cheap, both previously absent:

**Ablation (masking).** Re-run the decision with the named axis ignored. If the cut survives, the
axis was not the reason — the explanation was decoration. Faithfulness is the fraction of cuts
that disappear when their own stated axis is ablated.

**Shuffle.** Replace the concept deltas with a random permutation of themselves and re-run the
cut test. A faithful explanation system should change its decisions when its evidence is
scrambled; if the cut set is identical, the guard is not reading the deltas it reports.

Both are computed against the *same* candidate lists, so the only thing that changes between runs
is the evidence, never the position or the policy.

A third number, ``agreement``, is the one that matters for the product: when the guard cuts, does
an independent reference (the teacher's evaluation) agree the move was worse? That is
``guard_cut_audit``'s job and is reported next to these so a faithful-but-wrong guard cannot hide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from caissa2.core.types import CONCEPT_NAMES, SAFETY_AXES


@dataclass
class FaithfulnessReport:
    cuts: int = 0
    ablated_cuts_removed: int = 0
    shuffled_cuts_changed: int = 0
    per_axis: dict[str, dict] = field(default_factory=dict)

    @property
    def ablation_faithfulness(self) -> float | None:
        return round(self.ablated_cuts_removed / self.cuts, 4) if self.cuts else None

    @property
    def shuffle_sensitivity(self) -> float | None:
        return round(self.shuffled_cuts_changed / self.cuts, 4) if self.cuts else None

    def as_dict(self) -> dict:
        return {
            "cuts": self.cuts,
            "ablation_faithfulness": self.ablation_faithfulness,
            "shuffle_sensitivity": self.shuffle_sensitivity,
            "per_axis": self.per_axis,
        }


def _cut_ucis(guard, board: chess.Board, cands: list[chess.Move]) -> set[str]:
    _, cut = guard.filter(board, cands)
    return {v.move_uci for v in cut}


def _stated_axis(verdict) -> str | None:
    """The safety axis the verdict's reason names, if any."""
    for axis in SAFETY_AXES:
        if axis in " ".join(verdict.reasons):
            return axis
    return None


def faithfulness_report(model, guard, corpus, k: int = 12, seed: int = 0) -> FaithfulnessReport:
    """Ablation + shuffle faithfulness of the guard's axis-level explanations."""
    from caissa2.eval.arena import top_k_legal_moves
    from caissa2.guard.guard import ConceptGuard
    from caissa2.model.features import encode_batch
    from caissa2.core.encoding import legal_move_mask

    rng = np.random.default_rng(seed)
    rep = FaithfulnessReport()
    per_axis: dict[str, dict] = {a: {"cuts": 0, "removed": 0} for a in SAFETY_AXES}
    # One ablated guard per axis, built once: rebuilding it per cut made this O(cuts) model
    # constructions, which dominated the runtime of the whole report.
    ablated_guards = {
        axis: ConceptGuard(
            model,
            z_threshold=guard.z_threshold,
            min_drop=guard.min_drop,
            material_threshold=guard.material_threshold,
            use_head=guard.use_head,
            concept_corroboration=guard.concept_corroboration,
            ablate_axes={axis},
        )
        for axis in SAFETY_AXES
    }

    for item in corpus:
        board = chess.Board(item.fen)
        probs = model.policy(
            encode_batch([board]), np.asarray(legal_move_mask(board)[0])[None, :]
        )[0]
        cands = top_k_legal_moves(board, probs, k=k)
        if len(cands) < 2:
            continue

        verdicts = guard.inspect_many(board, cands)
        cut_verdicts = [v for v in verdicts if not v.passed]
        if not cut_verdicts:
            continue
        base = {v.move_uci for v in cut_verdicts}

        for v in cut_verdicts:
            axis = _stated_axis(v)
            if axis is None:
                continue
            rep.cuts += 1
            per_axis[axis]["cuts"] += 1

            # -- ablation: ignore the axis the explanation names -------------------
            ablated = ablated_guards[axis]
            after = _cut_ucis(ablated, board, cands)
            if v.move_uci not in after:
                rep.ablated_cuts_removed += 1
                per_axis[axis]["removed"] += 1

            # -- shuffle: scramble the evidence, keep the position -----------------
            # A shuffled guard is the same guard with its concept deltas permuted across
            # candidates. If the cut set is unchanged, the deltas are not driving the decision.
            shuffled = _ShuffledGuard(guard, rng)
            after_s = _cut_ucis(shuffled, board, cands)
            if after_s != base:
                rep.shuffled_cuts_changed += 1

    rep.per_axis = {
        a: {
            "cuts": d["cuts"],
            "removed": d["removed"],
            "faithfulness": round(d["removed"] / d["cuts"], 4) if d["cuts"] else None,
        }
        for a, d in per_axis.items()
    }
    return rep


class _ShuffledGuard:
    """A guard whose concept deltas are permuted across candidates before the cut test.

    Implemented by wrapping ``inspect_many``: the underlying guard computes verdicts normally, and
    we then re-decide using a permutation of the *deltas* while keeping the material check intact.
    Material is not shuffled because it is not a concept axis and is not part of the explanation
    being tested.
    """

    def __init__(self, guard, rng: np.random.Generator) -> None:
        self.guard = guard
        self.rng = rng
        self.material_threshold = guard.material_threshold
        self.z_threshold = guard.z_threshold
        self.min_drop = guard.min_drop
        self.use_head = guard.use_head
        self.concept_corroboration = guard.concept_corroboration
        self.ablate_axes = guard.ablate_axes

    def inspect_many(self, board: chess.Board, moves: list[chess.Move]):
        verdicts = self.guard.inspect_many(board, moves)
        if len(verdicts) < 2:
            return verdicts
        perm = self.rng.permutation(len(verdicts))
        out = []
        for i, v in enumerate(verdicts):
            src = verdicts[perm[i]]
            nv = type(v)(
                move_uci=v.move_uci,
                passed=v.passed,
                deltas=dict(src.deltas),
                material_loss=v.material_loss,
                material_move=v.material_move,
                reasons=list(v.reasons),
                z_score=src.z_score,
                concept_outlier=src.concept_outlier,
                corroborated=v.corroborated,
            )
            # re-decide on the shuffled evidence, keeping the material verdict
            if v.material_loss >= self.material_threshold:
                nv.passed = False
            elif src.concept_outlier and src.corroborated:
                nv.passed = False
            else:
                nv.passed = True
            out.append(nv)
        return out

    def filter(self, board: chess.Board, moves: list[chess.Move]):
        verdicts = self.inspect_many(board, moves)
        kept = [mv for mv, v in zip(moves, verdicts) if v.passed]
        cut = [v for v in verdicts if not v.passed]
        if not kept and cut:
            best = min(cut, key=lambda v: (v.material_loss, v.z_score))
            kept = [m for m in moves if m.uci() == best.move_uci]
            cut = [v for v in cut if v.move_uci != best.move_uci]
        return kept, cut
