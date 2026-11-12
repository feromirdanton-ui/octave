"""The experiment: measure Caïssa-1's throat against Caïssa-2's octave, step by step.

Run:  python -m caissa2.eval.cli --out reports/octave.json

Stages, each measured on the same held-out corpus:

  A. throat        — ThroatNet (policy = f(8 concepts)), no search, no guard
  B. octave        — OctaveNet (free 256-D trunk + concept head), no search, no guard
  C. octave+guard  — the same net with the concept-derivative guard as a filter
  D. lattice       — octave + PUCT search + guard + memory (the full player)
  E. lattice+loop  — the same, after self-play distillation rounds

Every stage reports blunder rate, top-1, concept MAE and legality.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import chess
import numpy as np

from caissa2.eval.arena import (
    evaluate_guarded_model,
    evaluate_model,
    evaluate_player,
    head_to_head,
)
from caissa2.eval.corpus import generate_corpus, generate_tactical_corpus
from caissa2.guard.guard import ConceptGuard
from caissa2.memory.index import ConceptMemory
from caissa2.model.nets import OctaveNet, ThroatNet, save_model
from caissa2.selfplay.distill import run_loop
from caissa2.selfplay.player import LatticePlayer
from caissa2.train import train_supervised


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Caïssa-2 (OCTAVE) experiment")
    ap.add_argument("--train-positions", type=int, default=400)
    ap.add_argument("--test-positions", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--trunk", type=int, default=256)
    ap.add_argument("--simulations", type=int, default=24)
    ap.add_argument("--loop-rounds", type=int, default=2)
    ap.add_argument("--loop-games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/octave.json")
    ap.add_argument("--models-dir", type=str, default="models")
    args = ap.parse_args(argv)

    t0 = time.time()
    print(f"[1/6] corpus: {args.train_positions} train / {args.test_positions} test ...", flush=True)
    # Half the corpus is teacher games (broad coverage), half is tactical positions where a
    # blunder is actually possible — otherwise the blunder rate measures nothing.
    half = max(1, args.train_positions // 2)
    train_corpus = generate_corpus(half, seed=args.seed) + generate_tactical_corpus(
        args.train_positions - half, seed=args.seed + 1
    )
    test_corpus = generate_tactical_corpus(args.test_positions, seed=args.seed + 10_000)
    print(f"      train={len(train_corpus)} test={len(test_corpus)}", flush=True)

    print("[2/6] training ThroatNet (Caïssa-1 baseline) ...", flush=True)
    throat = ThroatNet(seed=args.seed, lr=3e-3)
    throat_stats = train_supervised(throat, train_corpus, epochs=args.epochs, seed=args.seed)

    print(f"[3/6] training OctaveNet (trunk {args.trunk}-D) ...", flush=True)
    octave = OctaveNet(seed=args.seed, lr=3e-3, trunk_dim=args.trunk)
    octave_stats = train_supervised(octave, train_corpus, epochs=args.epochs, seed=args.seed)

    print("[4/6] arena: throat vs octave ...", flush=True)
    r_throat = evaluate_model(throat, test_corpus, name="A_throat")
    r_octave = evaluate_model(octave, test_corpus, name="B_octave")

    # C: octave + guard as a filter (no search). Candidates are the top-k *legal* moves, so an
    # empty candidate list (and the legality collapse it used to cause) is impossible.
    guard = ConceptGuard(octave)
    r_guard = evaluate_guarded_model(octave, test_corpus, guard, name="C_octave_guard").as_dict()

    print("[5/6] lattice player: search + guard + memory ...", flush=True)
    memory = ConceptMemory(k=8)
    for item in train_corpus:
        memory.add(chess.Board(item.fen), item.move_uci)
    lattice = LatticePlayer(
        octave, simulations=args.simulations, memory=memory, seed=args.seed, name="D_lattice"
    )
    r_lattice = evaluate_player(lattice, test_corpus, name="D_lattice")

    print("[6/6] self-play distillation loop ...", flush=True)
    loop = run_loop(
        octave,
        rounds=args.loop_rounds,
        games_per_round=args.loop_games,
        simulations=max(8, args.simulations // 2),
        max_plies=40,
        epochs=2,
        seed=args.seed,
        memory=memory,
    )
    lattice2 = LatticePlayer(
        octave, simulations=args.simulations, memory=memory, seed=args.seed, name="E_lattice_loop"
    )
    r_loop = evaluate_player(lattice2, test_corpus, name="E_lattice_loop")
    r_octave_after = evaluate_model(octave, test_corpus, name="E_octave_after_loop")

    h2h = head_to_head(lattice, lattice2, games=6, max_plies=60, seed=args.seed)

    os.makedirs(args.models_dir, exist_ok=True)
    save_model(throat, os.path.join(args.models_dir, "throat.npz"))
    save_model(octave, os.path.join(args.models_dir, "octave.npz"))

    report = {
        "config": vars(args),
        "train": {"throat": throat_stats, "octave": octave_stats},
        "arena": [
            r_throat.as_dict(),
            r_octave.as_dict(),
            r_guard,
            r_lattice.as_dict(),
            r_loop.as_dict(),
            r_octave_after.as_dict(),
        ],
        "selfplay_loop": loop.rounds,
        "head_to_head": h2h,
        "memory_entries": len(memory),
        "elapsed_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\n=== ARENA ===")
    for row in report["arena"]:
        print(
            f"{row['name']:<20} blunder={row['blunder_rate']:.3f}  top1={row['top1']:.3f}  "
            f"concept_mae={row['concept_mae']:.4f}  legality={row['legality']:.3f}"
        )
    print(f"\nwrote {args.out} in {report['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
