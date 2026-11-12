"""Core: encoding, concepts, types."""

from caissa2.core.concepts import concepts_after, extract_ground_truth_concepts
from caissa2.core.encoding import (
    BOARD_DIM,
    MOVE_DIM,
    encode_board,
    index_to_move,
    legal_move_mask,
    legal_moves_by_index,
    move_to_index,
)
from caissa2.core.types import CONCEPT_NAMES, N_CONCEPTS, ConceptVector

__all__ = [
    "BOARD_DIM",
    "MOVE_DIM",
    "CONCEPT_NAMES",
    "N_CONCEPTS",
    "ConceptVector",
    "concepts_after",
    "encode_board",
    "extract_ground_truth_concepts",
    "index_to_move",
    "legal_move_mask",
    "legal_moves_by_index",
    "move_to_index",
]
