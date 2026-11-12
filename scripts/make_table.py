"""Render the arena table from reports/octave.json as markdown.

Run:  python scripts/make_table.py reports/octave.json > reports/ARENA.md
"""

from __future__ import annotations

import json
import sys

LABELS = {
    "A_throat": "A. throat (Caïssa-1: policy = f(8 concepts))",
    "B_octave": "B. octave (free trunk + concept head, no search)",
    "C_octave_guard": "C. octave + guard (concept-derivative filter)",
    "D_lattice": "D. lattice (PUCT + guard + memory)",
    "E_lattice_loop": "E. lattice after self-play distillation",
    "E_octave_after_loop": "E'. bare octave after the same distillation",
}


def main(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)

    cfg = report["config"]
    print("### Arena (same held-out tactical corpus for every row)\n")
    print(
        f"train={cfg['train_positions']} positions, test={cfg['test_positions']}, "
        f"epochs={cfg['epochs']}, trunk={cfg['trunk']}-D, "
        f"simulations={cfg['simulations']}, "
        f"loop={cfg['loop_rounds']}x{cfg['loop_games']} games, seed={cfg['seed']}\n"
    )
    print("| player | blunder rate | top-1 policy | concept MAE | legality | n |")
    print("|---|---|---|---|---|---|")
    for row in report["arena"]:
        print(
            f"| {LABELS.get(row['name'], row['name'])} "
            f"| {row['blunder_rate']:.3f} "
            f"| {row['top1']:.3f} "
            f"| {row['concept_mae']:.4f} "
            f"| {row['legality']:.3f} "
            f"| {row['n']} |"
        )

    h2h = report.get("head_to_head")
    if h2h:
        print(
            f"\nHead-to-head D vs E: {h2h['games']} games, "
            f"score {h2h['score_a']:.2f}/{h2h['score_b']:.2f}, "
            f"W/L/D {h2h['wins_a']}/{h2h['losses_a']}/{h2h['draws']} "
            f"({h2h['decided']} decided)."
        )

    loop = report.get("selfplay_loop") or []
    if loop:
        print("\n### Self-play distillation rounds\n")
        print("| round | games | samples | mean result | CE | concept MSE | value MSE |")
        print("|---|---|---|---|---|---|---|")
        for r in loop:
            print(
                f"| {r['round']} | {r['games']} | {r['samples']} | {r['mean_result']:+.2f} "
                f"| {r['ce']:.3f} | {r['concept_mse']:.4f} | {r['value_mse']:.4f} |"
            )

    print(f"\nMemory entries: {report.get('memory_entries')}. "
          f"Elapsed: {report.get('elapsed_s')}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "reports/octave.json"))
