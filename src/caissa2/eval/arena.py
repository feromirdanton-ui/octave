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


def top_k_legal_moves(board: chess.Board, probs: np.ndarray, k: int = 12) -> list[chess.Move]:
    """The ``k`` most likely **legal** moves, best first.

    This is the only correct way to turn a 4096-way policy into a candidate list. Taking the
    top-k *indices* and then discarding the illegal ones is a bug: the mask zeroes every
    illegal index, so on a weakly trained net the top of the ranking is full of masked entries
    and the candidate list can come back empty — which silently turns "the guard chose a bad
    move" into "the guard chose no move at all" and corrupts legality and blunder rate.
    """
    legal = list(board.legal_moves)
    if not legal:
        return []
    scored = sorted(legal, key=lambda mv: -float(probs[move_to_index(mv)]))
    return scored[:k]


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


def evaluate_guarded_model(
    model, corpus: list[CorpusItem], guard, name: str | None = None, k: int = 12
) -> ArenaResult:
    """Measure a bare network with the concept guard as a *filter* (no search).

    Candidates come from :func:`top_k_legal_moves`, so the candidate list is never empty and
    legality is 1.0 by construction. The guard may reorder or cut, but it can never make the
    player pass.
    """
    blunders = 0
    top1 = 0
    legal = 0
    mae_terms = []

    for item in corpus:
        board = chess.Board(item.fen)
        probs = model.policy(encode_batch([board]), _mask(board))[0]
        candidates = top_k_legal_moves(board, probs, k=k)
        kept, _ = guard.filter(board, candidates) if candidates else ([], [])
        chosen = kept[0] if kept else (candidates[0] if candidates else None)
        if chosen is not None and chosen in board.legal_moves:
            legal += 1
            if blunder_loss(board, chosen) >= 2:
                blunders += 1
            if chosen.uci() == item.move_uci:
                top1 += 1
        pred = model.concepts(encode_batch([board]))[0]
        truth = extract_ground_truth_concepts(board).as_array()
        mae_terms.append(float(np.mean(np.abs(pred - truth))))

    n = len(corpus)
    return ArenaResult(
        name=name or f"{getattr(model, 'kind', 'model')}+guard",
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
    """Play two players against each other, alternating colours.

    ``score_a`` counts a draw as half a point, which is the usual convention — but a match of
    six drawn games and a match of three wins each both read 0.5, so the win/loss/draw split is
    reported alongside it. Without that split a 0.5/0.5 result looks like a measurement when it
    is really "nothing was decided".
    """
    from caissa2.selfplay.player import play_game

    score_a = 0.0
    results = []
    wins = losses = draws = 0
    for g in range(games):
        if g % 2 == 0:
            rec = play_game(a, b, max_plies=max_plies, temperature=0.0, remember=False)
            score_a += (rec.result + 1) / 2  # white is a
            outcome = rec.result
        else:
            rec = play_game(b, a, max_plies=max_plies, temperature=0.0, remember=False)
            score_a += (1 - rec.result) / 2  # black is a
            outcome = -rec.result
        if outcome > 0:
            wins += 1
        elif outcome < 0:
            losses += 1
        else:
            draws += 1
        results.append(rec.result)
    return {
        "games": games,
        "score_a": score_a / max(games, 1),
        "score_b": 1.0 - score_a / max(games, 1),
        "wins_a": wins,
        "losses_a": losses,
        "draws": draws,
        "decided": wins + losses,
        "results": results,
    }
