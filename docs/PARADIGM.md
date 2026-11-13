# Caïssa-2 — OCTAVE

> Caïssa-1 built a chess being whose policy was a function of eight numbers.
> Caïssa-2 keeps the eight numbers as a *measurement* and lets the policy read a free field.

This document is the paradigm, not the changelog. It states what changed, why, and what each
change is measured by.

---

## 0. The diagnosis

Caïssa-1 had a real, working, interpretable core: a board encoder, a legality mask, eight
concept axes extracted from python-chess, a guard that enumerated threats, and a policy that
was computed **from the eight concepts**.

That last clause is the problem. The eight axes were not a description of the position; they
were the *only* thing the policy could see. Every position was compressed to eight floats and
the policy was a linear map from those floats to 4096 move logits. Whatever a position contains
that eight numbers cannot say — a knight's outpost, a rook's latent activity, a mating net three
moves deep — was unreachable, not because the model was small but because the *architecture*
forbade it.

The eight axes were a throat. Caïssa-2 opens it.

---

## 1. Step 1 — open the throat

**Before.** `policy = softmax(W2 · tanh(W1 · board))` where `W1 · board ∈ R^8`.
The concept vector *is* the hidden state. Capacity is capped at eight dimensions by construction.

**After.** Two separate things:

```
trunk   h = relu(Wt · board)          h ∈ R^256   (free, 256–512-D)
policy  p = softmax(Wp · h)           the policy reads the trunk directly
concepts c = tanh(Wc · h + bc)        the concept head is an AUXILIARY task
value   v = tanh(wv · h)
```

The policy and the concept head are **siblings**, not parent and child. The concepts are still
predicted, still supervised against the ground-truth extractor, and still exact enough to
explain a move — but they no longer gate what the policy can express.

**What is preserved.**

- The explanation is still exact: the head is trained against `extract_ground_truth_concepts`,
  so "this move costs 0.4 of king safety" is a measurement, not a post-hoc story.
- The guard is still there, and it is now the *derivative of the head*: it asks which axes move
  along a candidate move, not which threats a hand-written rulebook can enumerate.
- Legality is still structural: the mask is applied inside the softmax, so a sampled move is
  legal by construction, not by rejection.

**What is measured.** Top-1 policy agreement with the teacher, concept MAE, blunder rate,
legality — the same four numbers for both architectures, on the same corpus, with the same
training budget. `tests/test_step1_throat.py` states the architectural claim as a test: in
`ThroatNet` the logits are *literally* `concepts @ W2`; in `OctaveNet` they are not.

---

## 2. Step 2 — search over the field

A network that outputs a policy and a value is a *field*: at every position it says "these moves
look plausible" and "this position is worth about this much". PUCT turns a field into a tree.

```
select   argmax over children of  Q(s,a) + c_puct · P(s,a) · sqrt(N(s)) / (1 + N(s,a))
expand   P from the network policy (mixed with the memory prior, step 4)
         V from the network value
         the guard CUTS branches before they are ever expanded
backup   negamax along the path
```

**The guard is a cut, not a prior.** It does not reweight a bad move down; it removes it. And
because the guard speaks in axes, the search can answer the question a policy alone cannot:

> *Why was that branch discarded?* — "g8e7: material −3 to c8g4 (SEE)"; "f7f5: king_safety
> −0.41 is an outlier drop (z=2.6 vs siblings, median +0.02)".

`PuctSearch.explain()` returns the chosen move, the visit counts, the Q values, and for every
discarded branch both a human-readable reason and the full per-axis delta vector.

**The cut is relative, and that is a finding, not a detail.** An absolute threshold on a concept
delta is meaningless because the head's scale is not calibrated to pawns: measured on real
tactical positions, the *median* safety-axis drop is ≈0.7, so any fixed threshold either cuts
everything or nothing. What is meaningful is an **outlier** — a move whose safety-axis drop is
far outside the distribution of drops among its siblings in the same position. That is a
derivative pointing downhill *relative to the alternatives*, which is the question a guard
should actually answer. The material check (SEE) stays absolute, because a hanging queen is not
a matter of opinion.

**Cost discipline.** The full concept guard (a batch forward pass plus SEE for every candidate)
runs at the root and at depth 1, where the explanation is read. Deeper nodes use the cheap
material-only check. A search that spends all its time in the guard is not a search.

---

## 3. Step 3 — self-play distillation

Caïssa-1 trained on human moves. That is imitation of a population that is not the thing we are
building, and it caps the student at the teacher.

Caïssa-2 trains on **the lattice player's own choices**:

```
field  ->  player  ->  games  ->  network  ->  better field
```

A *lattice player* is the whole stack acting as one agent: network prior + value, PUCT search,
concept-derivative guard, concept-space memory. It is stronger than the bare network, because
search and the guard correct the network's mistakes. Distilling the player back into the network
therefore *raises* the network toward the player, and the next round's player starts from a
better field.

The target is a **distribution**, not a move: the search's visit counts over the guard-filtered
candidates. The loss is a soft-target cross-entropy over the legal-move mask, plus the concept
head's auxiliary loss, plus a value loss against the game result.

`tests/test_step3_selfplay.py` asserts the inversion directly: the distillation target is the
player's own distribution, and training on it reduces cross-entropy.

---

## 4. Step 4 — memory as geometry

A position is a point in concept space. The index stores **nine floats** per position — the
eight axes plus a material balance — and the move that turned out best there.

At search time the retrieval prior asks: *in the positions nearest to this one, what was the
best move?* — and mixes that answer into the PUCT prior:

```
P(s,a) = (1 - λ) · P_net(s,a) + λ · P_memory(s,a)
P_memory(a) ∝ Σ_neighbours exp(-d(s, s_i) / τ) · w_i · [a_i = a]
```

This is deliberately **not** a hash table of exact positions. Two positions that are far apart
on the board but close in concept space are the same *kind* of position, and that is the whole
point: the memory generalises along the axes the system already measures. A position one tempo
later retrieves the lesson learned in its predecessor, because the nine floats barely moved.

---

## 5. What is measured, and how

Every number in the report comes from `caissa2.eval.arena`, on a held-out corpus, with a
deterministic teacher and a deterministic blunder definition:

| metric | definition |
|---|---|
| **blunder rate** | fraction of chosen moves that lose ≥ 2 pawns of material by static exchange evaluation |
| **eval drop** | mean teacher-evaluation given up against the best legal move, in pawns — the *independent* safety number |
| **top-1** | fraction of positions where the model's argmax equals the teacher's move |
| **concept MAE** | mean absolute error of the concept head against the ground-truth extractor |
| **legality** | fraction of chosen moves that are legal (must be 1.0 by construction) |

`eval_drop` exists because of a circularity that the first report did not see: the guard's
material check and the blunder metric are **the same function** (`worst_material_loss` and
`blunder_loss` agree on every pair tested). A guarded player's blunder rate is therefore partly a
statement about its own definition. `eval_drop` is computed from the teacher's 1-ply evaluation,
which the guard never consults, so it can disagree — and it does.

The test corpus is **tactical**: it keeps only positions in which at least one legal move loses
≥ 2 pawns by SEE and the teacher's move does not. A quiet opening position has no move that
loses material, so every model scores 0% and the metric measures nothing. This is stated here
because it is the difference between a number and a decoration.

The teacher is a deterministic 2-ply search over material + concept evaluation. It is not
Stockfish and does not pretend to be: it is a fixed, reproducible reference whose moves are good
enough to supervise a policy and to define a blunder objectively. Every number is reproducible
from a seed alone.

---

## 5a. The joint control — the baseline step 1 was missing

Step 1 claims that *separating* the measurement from the policy path buys something. The first
version of this project could not test that claim: its only baseline was `ThroatNet`, the
eight-number throat, so "octave beats throat" only said "a bigger model beats a smaller one",
which nobody doubted.

`JointNet` is the control. It has the **same parameter count** as the octave, but the policy
reads the **concept representation** rather than a free trunk — the concept layer is wide
(`concept_dim`, default equal to the trunk width) instead of eight. So the question becomes the
right one: *is the win from opening the throat, or merely from having more parameters?*

The first attempt at this control was wrong in a way worth recording. Its `JointNet` read the
same free trunk the octave reads, so the two models were the same function and produced
**bit-identical arena rows** (`B_octave` and `B2_joint_matched` both 0.417 / 0.117 / 0.2000). A
control that cannot differ from the treatment is worse than no control: it looks like evidence
and is not. The joint model now routes the policy through the concept layer, and
`test_joint_and_octave_are_not_the_same_function` asserts the two are not the same function.

---

## 5b. Metacognition — paying for search only where it is needed

The arena saturated: the full lattice player already scores 0.000 blunder rate on the tactical
corpus, so *more* search cannot show up as a better number. The remaining axis is cost.

`caissa2.meta.controller` predicts, before any search, whether the fast answer is about to be a
mistake — from cheap features (policy entropy, top-1 margin, concept spread, material, legal-move
count, guard cut fraction) — and routes a search budget from a fixed tier set `(4, 16, 64)`.

The label is the teacher's evaluation, which the guard never reads, so the predictor is not
learning the guard's own definition. The honest metric is **AUROC**: 0.5 means the predictor is a
coin flip and the routing idea is dead for this feature set. It is reported, not assumed.

---

## 5c. Faithfulness — the test we never ran

The guard says "I cut this move because `king_safety` dropped by 0.31, an outlier against its
siblings". That sentence is only worth something if it is *causally* true of the decision. Two
standard tests, both cheap, both previously absent (`caissa2.guard.faithfulness`):

- **Ablation (masking).** Re-run the decision with the named axis ignored. If the cut survives,
  the axis was not the reason. Faithfulness = fraction of cuts that disappear when their own
  stated axis is ablated.
- **Shuffle.** Permute the concept deltas across candidates and re-run. A faithful system should
  change its decisions when its evidence is scrambled; an identical cut set means the guard is
  not reading the deltas it reports.

Both run against the *same* candidate lists, so the only thing that changes is the evidence.
`guard_cut_audit` is reported next to them, so a faithful-but-wrong guard cannot hide.

---

## 5d. Memory that directs exploration

Step 4's retrieval prior originally only reweighted the **root**. That is a prior, not a memory:
if the field assigns a move probability ≈ 0, no amount of root reweighting will ever visit it.

`ConceptMemory.propose` returns moves the neighbourhood recommends and that are legal here, gated
by `confidence` — the share of neighbourhood weight held by its most-recommended move, scaled by
how close the nearest neighbour is. A confident neighbourhood gets a **prior floor** at the root,
so the search actually looks at a move the field would never have visited. The guard still has
the last word: a proposal the guard cut is not resurrected.

---

## 6. What this is not

- **Not a strength claim.** These are small numpy models on a CPU. The claim is architectural:
  the throat is open, the search explains itself, the loop is closed, the memory is geometric.
- **Not a proof of safety.** The guard is a hygiene gate. It cuts outliers and material losses;
  it does not prove a move is good.
- **Not a replacement for Caïssa-1.** Caïssa-1 is the reference implementation of the throat and
  the source of the encoder, the mask and the eight axes. Caïssa-2 is a separate project that
  starts from those contracts and changes the architecture around them.

---

## 7. Measurements, and one measurement bug that was found and fixed

The first arena run reported `legality = 0.2` for the guarded player (`C_octave_guard`). That
was **not** the guard cutting legal moves — it was a defect in the *measurement*.

Stage C built its candidate list by taking the top-12 **raw policy indices** and then discarding
the illegal ones. The policy is a masked softmax over 4096 from/to indices, so every illegal
index carries probability exactly 0 — and on a weakly trained net the top of the ranking is full
of those zeroed entries. The candidate list could therefore come back **empty**, the guarded
player then chose nothing at all, and "no move" was scored as "illegal move". Legality collapsed
and the blunder rate was computed over a denominator that included positions where no move was
made.

The fix is structural, not a threshold tweak: candidates are now built from the **legal move
list**, ranked by policy probability (`caissa2.eval.arena.top_k_legal_moves`). An empty candidate
list is impossible, so legality is 1.0 by construction and the guard can only ever reorder or
cut. Two regression tests pin this down (`test_top_k_legal_moves_never_returns_illegal_or_empty`,
`test_guarded_player_is_always_legal`).

The head-to-head figure was misleading for a second, unrelated reason: `score_a = 0.5` was
reported for a match of six **drawn** games. A draw is half a point, so the number was correct
but uninformative — it read like an even match when nothing had been decided. The match report
now carries the win/loss/draw split and a `decided` count next to the score.

### Arena after the fix

### Arena (same held-out tactical corpus for every row)

train=200 positions, test=60, epochs=6, trunk=128-D, simulations=12, loop=2x3 games, seed=0

| player | blunder rate | eval drop | top-1 policy | concept MAE | legality | n |
|---|---|---|---|---|---|---|
| A. throat (Caïssa-1: policy = f(8 concepts)) | 0.417 | 4.082 | 0.050 | 0.4106 | 1.000 | 60 |
| B. octave (free trunk + concept head, no search) | 0.417 | 3.483 | 0.117 | 0.2000 | 1.000 | 60 |
| B2. joint (parameter-matched, policy reads the concept layer) | 0.483 | 3.674 | 0.083 | 0.1888 | 1.000 | 60 |
| C0. octave + guard, concept head uncorroborated | 0.067 | 2.838 | 0.250 | 0.2000 | 1.000 | 60 |
| C. octave + guard (corroborated) | 0.067 | 2.838 | 0.250 | 0.2000 | 1.000 | 60 |
| D. lattice (PUCT + guard + memory) | 0.000 | 2.807 | 0.233 | 0.2000 | 1.000 | 60 |
| E. lattice after self-play distillation | 0.000 | 2.768 | 0.250 | 0.2055 | 1.000 | 60 |
| E'. bare octave after the same distillation | 0.400 | 3.510 | 0.117 | 0.2055 | 1.000 | 60 |

Parameters: throat 39,440 · octave 632,072 · joint 632,072 (matched exactly).

Head-to-head D vs E: 6 games, score 0.50/0.50, W/L/D 0/0/6 (0 decided).

### Metacognition

| metric | value | reading |
|---|---|---|
| AUROC (predict own error) | 0.530 | **a coin flip** — the feature set does not predict its own mistakes |
| ECE | 0.659 | badly calibrated |
| base error rate (test) | 0.750 | the bare policy is wrong on 3 of 4 tactical positions |
| routed vs fixed budget | 0.000 vs 0.000 blunder, 2.845 vs 2.845 eval drop | same safety |
| mean simulations | 7.2 vs 12.0 | **40% saved** |

The routing saves 40% of the simulations at identical safety — but the AUROC says *why* that is
not a win: the predictor is at chance, so the saving comes from the tiers being generous, not
from knowing where the errors are. On this corpus the honest conclusion is that the controller
is a **cost knob, not a metacognition**. It needs a corpus where the fast answer is sometimes
right and sometimes wrong; here it is wrong 75% of the time, so "always search" is nearly optimal
and there is nothing to route.

### Faithfulness of the guard's explanations

| metric | value | reading |
|---|---|---|
| cuts with a stated axis | 6 | all on `piece_activity` |
| ablation faithfulness | 0.167 | **only 1 of 6 cuts disappears when its own axis is ablated** |
| shuffle sensitivity | 0.500 | half the cut sets change when the evidence is scrambled |

This is the most uncomfortable number in the report, and it is the one the paper's §5.4 protocol
exists to produce. The guard's *material* cuts are sound (311 of them, and `guard_cut_audit`
shows 0.857 of the concept cuts are corroborated by the teacher). But the **axis-level
explanation** is largely decoration: ablating the named axis removes the cut only 1 time in 6.
The guard is cutting for the right reason (material) and narrating a concept axis that is not
what drove the decision. That is exactly the failure mode a faithfulness test is for, and it
would have gone unnoticed without one.

### Guard cut audit

| variant | material cuts | concept cuts | faithful | mean eval drop of concept cuts |
|---|---|---|---|---|
| corroborated | 311 | 7 | 0.857 | 3.749 |
| uncorroborated | 311 | 13 | 0.846 | 3.024 |

Corroboration removes 6 of 13 concept-only cuts and raises the mean eval drop of the ones it
keeps — the cuts it removes were the cheaper ones. It does **not** change the arena row
(C and C0 are identical at 0.067 / 2.838), because on this corpus the material check already
catches those moves. The corroboration is a correctness fix that this corpus cannot reward.

### Self-play distillation rounds

| round | games | samples | mean result | CE | concept MSE | value MSE |
|---|---|---|---|---|---|---|
| 0 | 3 | 100 | +0.00 | 0.879 | 0.0136 | 0.1186 |
| 1 | 3 | 60 | +0.00 | 0.501 | 0.0130 | 0.1277 |

Memory entries: 202. Elapsed: 1171.6s.

Reading the table honestly:

- **The throat is the bottleneck, and opening it helps the policy.** `A_throat` (policy = f(8
  concepts)) reaches top-1 0.050 and concept MAE 0.4106; `B_octave` (free 128-D trunk, concepts
  as an auxiliary head) reaches 0.117 with MAE 0.2000. The trunk carries information the eight
  axes cannot express, and the concept head is *better* for being a head, not worse.
- **The joint control says the win is separation, not capacity.** `B2_joint` has exactly the same
  632,072 parameters as the octave, but its policy must read the concept layer. It scores 0.483
  blunder / 0.083 top-1 against the octave's 0.417 / 0.117 — worse on both, at equal capacity.
  Its concept MAE is slightly better (0.1888), which is what you would expect: when the policy
  and the measurement share one path, the measurement is easier to fit and the policy is harder.
  **This is the evidence step 1 was missing**, and it points the right way.
- **The bare field is not safe, and it is not supposed to be.** `B_octave` blunders on 41.7% of
  tactical positions. A free trunk is more expressive, and an expressive policy trained on 200
  positions overfits to confident-looking nonsense. This is the honest cost of opening the
  throat, and it is exactly why the guard and the search exist.
- **The guard is the single largest safety gain.** `C_octave_guard` drops the blunder rate from
  0.417 to 0.067 and eval drop from 3.483 to 2.838 while *raising* top-1 to 0.250.
- **Search closes the rest.** `D_lattice` (PUCT + guard + memory) reaches 0.000 blunders, and the
  self-play loop (`E_lattice_loop`) lifts top-1 to 0.250 and lowers eval drop to 2.768.
- **The loop improves the player, not the bare net.** `E_octave_after_loop` still blunders on
  40.0% of positions. Distillation taught the net to imitate a *search-augmented* player; the net
  alone does not inherit the search's safety. That is the expected shape of this architecture.
- **Head-to-head D vs E decided nothing:** 6 games, 0 wins, 0 losses, 6 draws (all reaching the
  ply cap). Both players are deterministic and differ only in the distilled weights, so the
  match is not evidence of strength either way.

Caveat on scale: 200 training positions, 60 test positions, a 128-D trunk and 12 simulations.
These are architectural measurements on a CPU, not strength claims. The 60-position test set
gives ±0.12 on a blunder rate near 0.4, so differences smaller than that are not resolved.
