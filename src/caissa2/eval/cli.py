"""The experiment: measure Caïssa-1's throat against Caïssa-2's octave, step by step.

Run:  python -m caissa2.eval.cli --out reports/octave.json

Stages, each measured on the same held-out corpus:

  A. throat        — ThroatNet (policy = f(8 concepts)), no search, no guard
  B. octave        — OctaveNet (free 256-D trunk + concept head), no search, no guard
  C0. octave+guard(uncorroborated) — the concept head allowed to cut on its own (the defect)
  C. octave+guard  — the same net with the concept-derivative guard, corroborated by material
  D. lattice       — octave + PUCT search + guard + memory (the full player)
  E. lattice+loop  — the same, after self-play distillation rounds

Every stage reports blunder rate, eval drop, top-1, concept MAE and legality. ``eval_drop`` is
the independent safety number: the guard's material check and the blunder metric are the same
function, so the blunder rate alone cannot tell a real guard from a tautology.
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
    guard_cut_audit,
    head_to_head,
)
from caissa2.eval.corpus import generate_corpus, generate_tactical_corpus
from caissa2.guard.faithfulness import faithfulness_report
from caissa2.guard.guard import ConceptGuard
from caissa2.memory.index import ConceptMemory
from caissa2.meta.controller import (
    ErrorPredictor,
    MetacognitiveController,
    auroc,
    build_training_set,
    evaluate_routing,
    expected_calibration_error,
)
from caissa2.model.nets import (
    JointNet,
    OctaveNet,
    ThroatNet,
    matched_joint_concept_dim,
    param_count,
    save_model,
)
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

    # The joint control: same capacity, no separate concept head. This is the baseline the
    # factorization claim needs — without it, "octave beats throat" only says "a bigger model
    # beats a smaller one", which nobody doubted.
    joint_dim = matched_joint_concept_dim(args.trunk)
    print(f"[3b/6] training JointNet (concept layer {joint_dim}-D, parameter-matched) ...", flush=True)
    joint = JointNet(seed=args.seed, lr=3e-3, concept_dim=joint_dim)
    joint_stats = train_supervised(joint, train_corpus, epochs=args.epochs, seed=args.seed)

    print("[4/6] arena: throat vs octave ...", flush=True)
    r_throat = evaluate_model(throat, test_corpus, name="A_throat")
    r_octave = evaluate_model(octave, test_corpus, name="B_octave")
    r_joint = evaluate_model(joint, test_corpus, name="B2_joint_matched")

    # C: octave + guard as a filter (no search). Candidates are the top-k *legal* moves, so an
    # empty candidate list (and the legality collapse it used to cause) is impossible.
    #
    # C0 is the guard with the concept head allowed to cut on its own. It is kept in the report
    # because it is the measurement that found the defect: an uncorroborated concept outlier cut
    # moves that were *better* than the alternatives, and the blunder rate could not see it.
    guard_uncorroborated = ConceptGuard(octave, concept_corroboration=0)
    r_guard_unc = evaluate_guarded_model(
        octave, test_corpus, guard_uncorroborated, name="C0_guard_uncorroborated"
    ).as_dict()
    guard = ConceptGuard(octave)
    r_guard = evaluate_guarded_model(octave, test_corpus, guard, name="C_octave_guard").as_dict()
    cut_audit = guard_cut_audit(octave, guard, test_corpus)
    cut_audit_unc = guard_cut_audit(octave, guard_uncorroborated, test_corpus)

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

    # -- metacognition: predict your own error, then pay for search only where it is needed --
    print("[7/8] metacognitive controller ...", flush=True)
    X_train, y_train = build_training_set(octave, train_corpus, guard=guard)
    X_test, y_test = build_training_set(octave, test_corpus, guard=guard)
    predictor = ErrorPredictor().fit(X_train, y_train)
    risk_test = predictor.predict_proba(X_test)
    meta = {
        "auroc": round(auroc(y_test, risk_test), 4) if not np.isnan(auroc(y_test, risk_test)) else None,
        "ece": round(expected_calibration_error(y_test, risk_test), 4),
        "base_error_rate_train": round(float(y_train.mean()), 4),
        "base_error_rate_test": round(float(y_test.mean()), 4),
        "weights": predictor.weights(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    controller = MetacognitiveController(
        predictor=predictor, tiers=(4, 12, 24), thresholds=(0.25, 0.6), guard=guard
    )
    routing = evaluate_routing(
        controller,
        octave,
        test_corpus[:40],
        player_factory=lambda sims: LatticePlayer(
            octave, simulations=sims, memory=memory, seed=args.seed, name=f"routed-{sims}"
        ),
        fixed_budget=args.simulations,
    )

    # -- faithfulness of the guard's axis-level explanations -------------------------------
    print("[8/8] explanation faithfulness (ablation + shuffle) ...", flush=True)
    faith = faithfulness_report(octave, guard, test_corpus[:30], k=12, seed=args.seed).as_dict()

    h2h = head_to_head(lattice, lattice2, games=6, max_plies=60, seed=args.seed)

    os.makedirs(args.models_dir, exist_ok=True)
    save_model(throat, os.path.join(args.models_dir, "throat.npz"))
    save_model(octave, os.path.join(args.models_dir, "octave.npz"))
    save_model(joint, os.path.join(args.models_dir, "joint.npz"))

    report = {
        "config": vars(args),
        "params": {
            "throat": param_count(throat),
            "octave": param_count(octave),
            "joint": param_count(joint),
            "joint_concept_dim": joint_dim,
        },
        "train": {"throat": throat_stats, "octave": octave_stats, "joint": joint_stats},
        "arena": [
            r_throat.as_dict(),
            r_octave.as_dict(),
            r_joint.as_dict(),
            r_guard_unc,
            r_guard,
            r_lattice.as_dict(),
            r_loop.as_dict(),
            r_octave_after.as_dict(),
        ],
        "guard_cut_audit": {
            "corroborated": cut_audit,
            "uncorroborated": cut_audit_unc,
        },
        "metacognition": meta,
        "routing": routing,
        "faithfulness": faith,
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
        drop = row.get("eval_drop")
        drop_s = f"{drop:.3f}" if drop is not None else "  n/a"
        print(
            f"{row['name']:<24} blunder={row['blunder_rate']:.3f}  eval_drop={drop_s}  "
            f"top1={row['top1']:.3f}  concept_mae={row['concept_mae']:.4f}  "
            f"legality={row['legality']:.3f}"
        )
    print("\n=== GUARD CUT AUDIT ===")
    for label, audit in report["guard_cut_audit"].items():
        print(f"{label:<16} {audit}")
    print("\n=== METACOGNITION ===")
    print(f"auroc={meta['auroc']}  ece={meta['ece']}  base_error={meta['base_error_rate_test']}")
    print(f"routing: {routing}")
    print("\n=== FAITHFULNESS ===")
    print(faith)
    print(f"\nwrote {args.out} in {report['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
