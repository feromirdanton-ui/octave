"""Shared types for Caïssa-2 (OCTAVE).

The eight concept axes are the *measurement* vocabulary of the system. In Caïssa-1 they were
also the *bottleneck*: the policy was computed from these eight numbers and nothing else, so
the model could never express anything the eight axes could not say. In Caïssa-2 they are a
head — a measurement — while the policy reads a free latent trunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

CONCEPT_NAMES: tuple[str, ...] = (
    "king_safety",
    "center_control",
    "piece_activity",
    "pawn_structure",
    "tactical_tension",
    "passed_pawn_threat",
    "open_file_dominance",
    "endgame_simplification",
)

N_CONCEPTS = len(CONCEPT_NAMES)

#: Axes whose *drop* means "this move walks into something". The guard reads these.
SAFETY_AXES: tuple[str, ...] = ("king_safety", "pawn_structure", "piece_activity")

#: Axes that are *not* monotone-good: a rise in tension is not automatically bad, so the guard
#: treats them as context for the explanation rather than as a cut criterion.
CONTEXT_AXES: tuple[str, ...] = (
    "center_control",
    "tactical_tension",
    "passed_pawn_threat",
    "open_file_dominance",
    "endgame_simplification",
)


@dataclass(frozen=True)
class ConceptVector:
    """An ordered, named concept measurement in [-1, 1]."""

    values: Mapping[str, float]

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(float(self.values[name]) for name in CONCEPT_NAMES)

    def as_array(self):
        import numpy as np

        return np.asarray(self.as_tuple(), dtype=float)

    def get(self, name: str) -> float:
        return float(self.values[name])

    def __len__(self) -> int:
        return len(CONCEPT_NAMES)

    def __iter__(self) -> Iterable[str]:
        return iter(CONCEPT_NAMES)


def concept_array(values: Mapping[str, float]):
    """Build a numpy concept vector from a name->value mapping, in canonical axis order."""
    import numpy as np

    return np.asarray([float(values[name]) for name in CONCEPT_NAMES], dtype=float)


def axis_deltas(before, after) -> dict[str, float]:
    """Per-axis difference ``after - before`` for two concept vectors (arrays or mappings)."""
    import numpy as np

    b = np.asarray(before, dtype=float) if not isinstance(before, Mapping) else concept_array(before)
    a = np.asarray(after, dtype=float) if not isinstance(after, Mapping) else concept_array(after)
    return {name: float(a[i] - b[i]) for i, name in enumerate(CONCEPT_NAMES)}
