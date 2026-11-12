# Caïssa-2 — OCTAVE

A chess being whose policy reads a **free latent trunk**, whose concepts are a **measurement**
rather than a bottleneck, whose search **explains what it discarded**, whose training signal is
**its own play**, and whose memory is **geometry**.

This is a separate project from [Caïssa-1](https://github.com/feromirdanton-ui/caissa). It
starts from Caïssa-1's contracts — the 832-D board encoding, the 4096-D legality mask, the eight
concept axes — and changes the architecture around them.

Read [`docs/PARADIGM.md`](docs/PARADIGM.md) for the reasoning. This file is the map.

---

## The four steps

| # | Step | Where |
|---|---|---|
| 1 | **Open the throat** — free 256–512-D trunk, concept head as an auxiliary task | `src/caissa2/model/nets.py` |
| 2 | **Search over the field** — PUCT/MCTS with network prior+value, guard as a cut, reasons in axis terms | `src/caissa2/search/puct.py` |
| 3 | **Self-play distillation** — the network imitates the lattice player, not humans | `src/caissa2/selfplay/` |
| 4 | **Memory as geometry** — 9-float concept index, retrieval prior in search | `src/caissa2/memory/index.py` |

---

## Install and run

```bash
pip install -e ".[dev]"

# the whole experiment: train both architectures, run the arena, close the loop
python -m caissa2.eval.cli --out reports/octave.json

# tests
pytest -q
```

No GPU, no engine, no network. Everything is numpy + python-chess, and every number is
reproducible from a seed.

---

## The two architectures, side by side

```python
# Caïssa-1 — the throat: the concept vector IS the hidden state
h = tanh(W1 @ board)          # h ∈ R^8
policy = softmax(W2 @ h)      # capacity capped at eight numbers

# Caïssa-2 — the octave: the trunk is free, the concepts are a sibling head
h = relu(Wt @ board)          # h ∈ R^256
policy = softmax(Wp @ h)      # the policy reads the trunk directly
concepts = tanh(Wc @ h + bc)  # the head still measures — it just no longer gates
```

`tests/test_step1_throat.py` asserts this difference as a test rather than a comment.

---

## The lattice player

```
field (prior + value)  ->  PUCT search  ->  guard (concept derivative)  ->  memory (retrieval)
```

It is the agent whose choices the network is distilled toward. `LatticePlayer.explain(board)`
returns the chosen move, the visit counts, and for every discarded branch a reason in axis terms:

```python
{
  "chosen": "f7f5",
  "visits": {"f7f5": 9, "g8f6": 6, ...},
  "discarded": [
    "g8e7: material -3 to c8g4 (SEE)",
    "g7f6: king_safety -0.41 is an outlier drop (z=2.6 vs siblings, median +0.02)"
  ],
  "discarded_axes": [{"move": "g7f6", "axes": {"king_safety": -0.41, ...}}]
}
```

---

## The four numbers

| metric | definition |
|---|---|
| blunder rate | fraction of chosen moves losing ≥ 2 pawns by static exchange evaluation |
| top-1 | fraction of positions where the argmax equals the teacher's move |
| concept MAE | mean absolute error of the concept head vs the ground-truth extractor |
| legality | fraction of chosen moves that are legal (1.0 by construction) |

Measured on a **tactical** held-out corpus — positions where at least one legal move loses
material. On quiet positions every model scores 0% and the metric measures nothing.

---

## Layout

```
src/caissa2/
  core/       encoding (832-D), concepts (8 axes), types
  model/      ThroatNet (Caïssa-1 baseline), OctaveNet (free trunk + concept head)
  guard/      ConceptGuard — the derivative of the head, plus SEE
  search/     PuctSearch — prior+value, guard cuts, axis-level explanations
  memory/     ConceptMemory — 9-float index, k-NN retrieval prior
  selfplay/   LatticePlayer, play_game, distillation loop
  eval/       corpus, arena, CLI
tests/        one file per step, plus the core contract
docs/         PARADIGM.md
```

---

## Licence

Apache-2.0.
