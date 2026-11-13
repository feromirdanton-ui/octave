"""The arena: five numbers, measured the same way for every model.

1. **blunder rate** — fraction of chosen moves that lose >= 2 pawns of material by SEE.
2. **eval drop** — mean teacher-evaluation loss of the chosen move against the best legal move,
   in pawns. This is the *independent* safety number, and it exists because of a circularity:
   the guard's material check and the blunder metric are the same function
   (``worst_material_loss == blunder_loss`` on every pair we tested), so a guarded player's
   blunder rate is partly a statement about its own definition. ``eval_drop`` is computed from
   the teacher's 1-ply evaluation, which the guard never consults, so it can disagree.
3. **top-1 policy** — fraction of positions where the model's argmax equals the teacher's move.
4. **concept MAE** — mean absolute error of the concept head against the ground-truth extractor.
5. **legality** — fraction of chosen moves that are legal (must be 1.0 by construction).

The arena is deliberately engine-free: every number is reproducible from a seed, and the
blunder definition is a static exchange evaluation rather than an opinion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from caissa2.core.encoding import legal_move_mask, move_to_index
from caissa2.core.concepts import extract_ground_truth_concepts
from caissa2.eval.corpus import CorpusItem, blunder_loss, evaluate
from caissa2.model.features import encode_batch


def move_value(board: chess.Board, move: chess.Move) -> float:
    """The teacher's 1-ply value of ``move`` for the mover, in pawns."""
    after = board.copy()
    after.push(move)
    return -evaluate(after)


def best_move_value(board: chess.Board) -> float:
    """The teacher's value of the best legal move, in pawns."""
    return max(move_value(board, mv) for mv in board.legal_moves)


def eval_drop(board: chess.Board, move: chess.Move | None, best: float | None = None) -> float:
    """How much teacher-evaluation the chosen move gives up against the best legal move.

    Independent of the guard: the guard reads the concept head and SEE, this reads the
    teacher's material+concept evaluation. A move the guard keeps can still be a bad move.
    """
    if move is None or move not in board.legal_moves:
        return float("nan")
    if best is None:
        best = best_move_value(board)
    return float(best - move_value(board, move))


@dataclass
class ArenaResult:
    name: str
    blunder_rate: float
    top1: float
    concept_mae: float
    legality: float
    n: int
    eval_drop: float = float("nan")
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "blunder_rate": round(self.blunder_rate, 4),
            "eval_drop": round(self.eval_drop, 4) if self.eval_drop == self.eval_drop else None,
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
    drops = []

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
            drops.append(eval_drop(board, move))
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
        eval_drop=float(np.mean(drops)) if drops else float("nan"),
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
    drops = []

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
            drops.append(eval_drop(board, chosen))
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
        eval_drop=float(np.mean(drops)) if drops else float("nan"),
    )


def evaluate_player(player, corpus: list[CorpusItem], name: str | None = None) -> ArenaResult:
    """Measure a full lattice player (search + guard + memory) on the corpus."""
    blunders = 0
    top1 = 0
    legal = 0
    mae_terms = []
    drops = []

    for item in corpus:
        board = chess.Board(item.fen)
        move = player.choose(board, temperature=0.0)
        if move is not None and move in board.legal_moves:
            legal += 1
            if blunder_loss(board, move) >= 2:
                blunders += 1
            drops.append(eval_drop(board, move))
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
        eval_drop=float(np.mean(drops)) if drops else float("nan"),
    )


def guard_cut_audit(model, guard, corpus: list[CorpusItem], k: int = 12) -> dict:
    """Audit the guard's cuts against an independent reference.

    Splits the cuts into the ones the *material* check made and the ones the *concept head* made
    on its own, and asks of the latter: does the teacher's evaluation agree that the cut move was
    worse than the best legal move? This is the check that caught the guard cutting good moves.
    """
    material_cuts = 0
    concept_cuts = 0
    concept_faithful = 0
    concept_deltas: list[float] = []

    for item in corpus:
        board = chess.Board(item.fen)
        probs = model.policy(encode_batch([board]), _mask(board))[0]
        cands = top_k_legal_moves(board, probs, k=k)
        if not cands:
            continue
        _, cut = guard.filter(board, cands)
        best = best_move_value(board)
        for v in cut:
            mv = chess.Move.from_uci(v.move_uci)
            if mv not in board.legal_moves:
                continue
            if v.material_loss >= guard.material_threshold:
                material_cuts += 1
            else:
                concept_cuts += 1
                d = eval_drop(board, mv, best=best)
                concept_deltas.append(d)
                if d > 0:
                    concept_faithful += 1

    return {
        "material_cuts": material_cuts,
        "concept_cuts": concept_cuts,
        "concept_cuts_faithful": concept_faithful,
        "concept_cut_faithfulness": (
            round(concept_faithful / concept_cuts, 4) if concept_cuts else None
        ),
        "concept_cut_mean_eval_drop": (
            round(float(np.mean(concept_deltas)), 4) if concept_deltas else None
        ),
    }


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
