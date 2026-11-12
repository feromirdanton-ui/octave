"""Render reports/octave.json into a readable Markdown table (docs/RESULTS.md)."""

from __future__ import annotations

import json
import sys


def main(path: str = "reports/octave.json", out: str = "docs/RESULTS.md") -> int:
    with open(path, encoding="utf-8") as fh:
        r = json.load(fh)

    lines = ["# Caïssa-2 (OCTAVE) — measured results", ""]
    cfg = r["config"]
    lines += [
        f"Corpus: {cfg['train_positions']} train / {cfg['test_positions']} test positions "
        f"(test set is tactical: at least one legal move loses >= 2 pawns by SEE).",
        f"Trunk: {cfg['trunk']}-D · epochs: {cfg['epochs']} · search: {cfg['simulations']} sims · "
        f"self-play: {cfg['loop_rounds']} rounds x {cfg['loop_games']} games · seed {cfg['seed']}.",
        "",
        "| stage | blunder rate | top-1 | concept MAE | legality |",
        "|---|---|---|---|---|",
    ]
    for row in r["arena"]:
        lines.append(
            f"| `{row['name']}` | {row['blunder_rate']:.3f} | {row['top1']:.3f} | "
            f"{row['concept_mae']:.4f} | {row['legality']:.3f} |"
        )

    lines += [
        "",
        "## Self-play loop",
        "",
        "| round | games | samples | mean result | CE | concept MSE |",
        "|---|---|---|---|---|---|",
    ]
    for rd in r["selfplay_loop"]:
        lines.append(
            f"| {rd['round']} | {rd['games']} | {rd['samples']} | {rd['mean_result']:+.2f} | "
            f"{rd['ce']:.3f} | {rd['concept_mse']:.4f} |"
        )

    h = r["head_to_head"]
    lines += [
        "",
        "## Head to head (lattice before vs after the loop)",
        "",
        f"{h['games']} games, alternating colours: before {h['score_a']:.2f} — "
        f"after {h['score_b']:.2f}.",
        "",
        f"Memory entries: {r['memory_entries']} · wall clock: {r['elapsed_s']} s.",
        "",
    ]

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
