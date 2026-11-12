"""The arena: four numbers, measured the same way for every model.

1. **blunder rate** — fraction of chosen moves that lose >= 2 pawns of material by SEE.
2. **top-1 policy** — fraction of positions where the model's argmax equals the teacher's move.
3. **concept MAE** — mean absolute error of the concept head against the ground-truth extractor.
4. **legality** — fraction of chosen moves that are legal (must be 1.0 by construction).

The arena is deliberately engine-free: every number is reproducible from a seed, and the
blunder definition is a static exchange evaluation rather than an opinion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from caissa2.core.encoding import legal_move_mask, move_to_index
from caissa2.core.concepts import extract_ground_truth_concepts
from caissa2.eval.corpus import CorpusItem, blunder_loss
from caissa2.model.features import encode_batch


@dataclass
class ArenaResult:
    name: str
    blunder_rate: float
    top1: float
    concept_mae: float
    legality: float
    n: int
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "blunder_rate": round(self.blunder_rate, 4),
            "top1": round(self.top1, 4),
            "concept_mae": round(self.concept_mae, 4),
            "legality": round(self.legality, 4),
            "n": self.n,
            **self.extra,
        }


def _mask(board: chess.Board) -> np.ndarray:
    m, _ = legal_move_mask(board)
    return np.asarray(m, dtype=bool)[None, :]


def evaluate_model(model, corpus: list[CorpusItem], name: str | None = None) -> ArenaResult:
    """Measure a bare network (no search, no guard) on the corpus."""
    boards = [chess.Board(item.fen) for item in corpus]
    x = encode_batch(boards)

    blunders = 0
    top1 = 0
    legal = 0
    mae_terms = []

    for i, (board, item) in enumerate(zip(boards, corpus)):
        mask = _mask(board)
        probs = model.policy(x[i : i + 1], mask)[0]
        idx = int(np.argmax(probs))
        frm, to = divmod(idx, 64)
        move = chess.Move(frm, to)
        if move not in board.legal_moves:
            move = chess.Move(frm, to, promotion=chess.QUEEN)
        if move in board.legal_moves:
            legal += 1
            if blunder_loss(board, move) >= 2:
                blunders += 1
        if move.uci() == item.move_uci:
            top1 += 1
        pred = model.concepts(x[i : i + 1])[0]
        truth = extract_ground_truth_concepts(board).as_array()
        mae_terms.append(float(np.mean(np.abs(pred - truth))))

    n = len(corpus)
    return ArenaResult(
        name=name or getattr(model, "kind", "model"),
        blunder_rate=blunders / max(n, 1),
        top1=top1 / max(n, 1),
        concept_mae=float(np.mean(mae_terms)) if mae_terms else 0.0,
        legality=legal / max(n, 1),
        n=n,
    )


def evaluate_player(player, corpus: list[CorpusItem], name: str | None = None) -> ArenaResult:
    """Measure a full lattice player (search + guard + memory) on the corpus."""
    blunders = 0
    top1 = 0
    legal = 0
    mae_terms = []

    for item in corpus:
        board = chess.Board(item.fen)
        move = player.choose(board, temperature=0.0)
        if move is not None and move in board.legal_moves:
            legal += 1
            if blunder_loss(board, move) >= 2:
                blunders += 1
            if move.uci() == item.move_uci:
                top1 += 1
        x = encode_batch([board])
        pred = player.model.concepts(x)[0]
        truth = extract_ground_truth_concepts(board).as_array()
        mae_terms.append(float(np.mean(np.abs(pred - truth))))

    n = len(corpus)
    return ArenaResult(
        name=name or getattr(player, "name", "player"),
        blunder_rate=blunders / max(n, 1),
        top1=top1 / max(n, 1),
        concept_mae=float(np.mean(mae_terms)) if mae_terms else 0.0,
        legality=legal / max(n, 1),
        n=n,
    )


def head_to_head(a, b, games: int = 20, max_plies: int = 80, seed: int = 0) -> dict:
    """Play two players against each other, alternating colours."""
    from caissa2.selfplay.player import play_game

    score_a = 0.0
    results = []
    for g in range(games):
        if g % 2 == 0:
            rec = play_game(a, b, max_plies=max_plies, temperature=0.0, remember=False)
            score_a += (rec.result + 1) / 2  # white is a
        else:
            rec = play_game(b, a, max_plies=max_plies, temperature=0.0, remember=False)
            score_a += (1 - rec.result) / 2  # black is a
        results.append(rec.result)
    return {
        "games": games,
        "score_a": score_a / max(games, 1),
        "score_b": 1.0 - score_a / max(games, 1),
        "results": results,
    }
