# GlassBox — finding the exact step where Gemma 4 starts guessing

**Track:** AI Shield
**Team:** *(your team name)*

---

## Problem

Gemma 4 reasons before it answers — thousands of tokens of visible
chain-of-thought. But **nobody reads 4,000 tokens of reasoning.** The trace is
right there and still a black box. Everyone will *use* thinking mode; we decided
to **instrument** it.

We do not answer "is this true?" — that needs a source. We answer **"is the model
making this up right now, and exactly where?"** That is answerable from the model
alone, and it is what a guardrail needs.

---

## Solution

GlassBox makes Gemma 4 reason **seven times**, aligns the traces step by step, and
measures where they stop agreeing. The output is not *"this answer is
unreliable"* — it is **"step 2 is where it started guessing, and here is what it
guessed instead."**

> **What is the atomic radius of ununoctium in picometres?**
> Gemma 4 answers confidently: *"ununoctium, **element 111**…"* — it is 118.

```
1. [procedure]   Identify the core request…
2. h=0.59  FORK  Identify the element: Ununoctium (Uun) is element 111
       also read as: Ununoctium (Uun) is element 112
3-6. [procedure]
→  ANSWER  h=0.39, 2 readings across 6 traces
       another trace answered: …Ununoctium (Uuo), element 118…
```

Six steps, one real assertion, and it is the one carrying the error — flagged
with no reference to the true answer, purely because the model could not tell the
same story twice.

---

## How Gemma 4 is used

- **Model variant:** Gemma 4 E2B instruction-tuned, Q4_K_M GGUF (2.9 GB).
- **Role:** one model is **both the reasoner and its own judge.** No second
  model, no API, no network.
- **As reasoner:** Gemma 4 emits `<|channel>thought` … `<channel|>` and plans in
  numbered steps, so traces **parse** instead of needing a heuristic.
- **As judge:** the judge never speaks. One forward pass comparing the raw logits
  of the `YES` / `NO` tokens: `P(same) = softmax(logit_YES, logit_NO)`.
- **As repairer:** generation resumes from a hand-edited reasoning prefix — we
  rewrite the model's own thought and continue it.
- **Why E2B:** fits a CPU laptop with room for a 7-trace working set, and at
  2.9 GB runs fully offline.
- **Customisation:** no fine-tuning. Prompt design, logit reading, and a linear
  probe on hidden states.

---

## Architecture

```
question
  ├─► 7 traces (1 greedy + 6 @ temp 0.9, distinct seeds)
  │     └─ parse <|channel>thought into numbered steps
  ├─► classify step: procedure | assertion   ← only assertions can be wrong
  ├─► align assertions by ROLE across traces (not by position)
  ├─► cluster + normalised entropy → fork = most divergent step
  │     └─ every clustering call via the logit judge (0 tokens generated)
  ├─► repair: adjudicate its own competing readings, "none of these" allowed
  └─► linear probe on hidden state → single-pass screening (cascade)
```

**Tech stack:** Python (stdlib `http.server`), one dependency —
`llama-cpp-python` (prebuilt CPU wheel) — one GGUF, vanilla HTML/CSS/JS UI.
Target: an ordinary laptop CPU, offline.

---

## Three engineering decisions that made it work

### 1. Read the logits, not the words

The single biggest correctness fix. Asked out loud, a small Gemma has such a heavy
`YES` prior that it called *"Leonardo painted it"* and *"Michelangelo painted it"*
the same claim — every question collapsed to one cluster and every entropy read
zero. Six prompt strategies against 12 labelled pairs scored **5/12 to 10/12**;
asking "do these *conflict*?" also returned YES to everything. The model was not
comparing, it was agreeing.

Reading the logits gave a graded, calibratable probability: **12/12** — and costs
*less*, one prefill and zero tokens generated.

### 2. Score assertions, not procedure

Our first version pointed at the wrong step. Procedure steps get reworded freely
between samples and scored entropy **1.0**, while the step carrying the real error
scored **0.59** — ranking on raw entropy was actively misleading. Filtering to
assertions took the ununoctium trace from six noisy steps to exactly one, the
right one.

### 3. Alignment and agreement are different questions

Aligning by position made *"Who painted the Mona Lisa?"* a false positive: trace
A's step 2 was "Identify the Subject" where trace B's was "Recall knowledge…", so
index matching pits unlike steps together and invents disagreement. Fixing it
exposed a subtler bug — we had reused the *agreement* prompt for the *alignment*
search, but "is element 111" vs "is element 118" correctly scores NO as a
claim-match, so the **right** counterpart looked no better than an unrelated one.
Alignment needs its own prompt, asking about *role* and ignoring agreement.

---

## Results

| | |
|---|---|
| Judge accuracy | **12/12** labelled equivalence pairs |
| Answer-level detection | **92%** recall at **100%** precision (n=24) |
| Step-level localization | **86%** accuracy, 83% precision, 83% recall (n=14) |
| Single-pass probe | **AUROC 0.885** (n=93) |
| Mean entropy, answerable / obscure | 0.026 / 0.674 |
| Self-repair | 0 corrections, **0 false repairs** |
| Thinking-mode ablation | null result |

Every threshold comes from a sweep over labelled data and is **centred in its
winning range**, never parked on the edge. The fork threshold is 0.46, the centre
of the optimal [0.36, 0.57] range. Our first tuner picked a cut-off 0.009 above
the highest grounded score; entropy from a handful of samples moves between runs,
and that would have flipped on noise. We caught it because the tuner is code, not
a judgement call.

**Cost, measured on one question:** one answer with thinking generates ~640
tokens; GlassBox's 7 traces generate **4,487** — ~7×. That is the honest price,
since disagreement between samples *is* the measurement. Two things offset it:
136 judge calls generate **zero** tokens, and KV-prefix reuse cut them 55s → 18.5s.

---

## Three honest findings

**Thinking mode showed no effect.** We assumed it drove Gemma 4's refusals and ran
the ablation to quantify it. Thinking off: declines all 12. Thinking on: declines
all 12. **0% confabulation in both conditions** — a floor effect, not an effect.
The tempting comparison (Gemma 3-without-thinking vs Gemma 4-with-thinking) moves
two variables and credits the feature for the model's improvement. This reshaped
the project: the easy bait is gone, and what survives is subtler — **partial
knowledge about real entities**, where the model half-remembers and fills the gap.
Exactly what a refusal-based guardrail misses.

**Self-repair recognises but does not correct.** Across 8 questions only 2
produced a divergent step whose readings were factual claims, and one was a clean
test: offered "element 111" vs "element 112", it declined both — correctly, since
ununoctium is 118 and 118 was never in its own candidate set. **n=1 is not a
result.** The flaw is structural: adjudication can only choose among readings the
sampling happened to produce. The safety property did hold — an earlier version
*forced* a choice, picked 112, and confidently rebuilt the answer around it, one
wrong claim swapped for another while *looking* corrected. Adding "none of these"
made it fail closed: **zero false repairs.**

**One forward pass can replace seven traces.** We implemented Semantic Entropy
Probes (Kossen et al., arXiv:2406.15927) — a linear probe on the hidden state, no
sampling. On 93 labelled examples, leave-one-out cross-validated: **AUROC 0.885**,
83% accuracy against a 76% majority baseline. Quote the AUROC — the set is 76%
negative, so accuracy flatters the baseline. That makes GlassBox a cascade: screen
every request in one pass at ~200× less cost, spend the seven traces only on what
the probe flags. We read the final-layer embedding rather than the intermediate
layers the paper probes, and n=93 is small, so treat it as feasibility.

---

## What we do not claim

**Consistent false belief is out of scope.** If all seven traces make the *same*
wrong move, entropy is zero and we call it stable. GlassBox measures whether a
model is **inventing on the spot**, not whether it is **right**.

In the same spirit: our eval flagged *"boiling point of water in Celsius"* at
0.311 where we had labelled it answerable. Gemma had replied *"212 degrees
Celsius"* — the Fahrenheit figure. The detector was right and our label was wrong.
We left the label alone; relabelling after seeing the score manufactures a number.

---

## Why this needs open weights

Three hard requirements, none available through an API:

1. **Logit access.** Without it the project does not function.
2. **Per-call seeds**, so seven traces are reproducible and cacheable.
3. **Trace-prefix continuation.** No chat endpoint lets you rewrite the model's
   own thought and resume it.

It runs offline on a CPU laptop: no server, no API key, no network once the GGUF
is on disk.

---

## Links

- **GitHub repo:** https://github.com/AviDhandhania/glassbox
- **Live demo:** `demo.ipynb` — replays the full pipeline from the committed
  cache in seconds, no model download required. Also `python glassbox.py serve`
  for the local web UI.
- **License:** Apache 2.0

## Acknowledgments

Google DeepMind for the Gemma 4 open weights; `llama-cpp-python` for CPU
inference; Kossen et al. (arXiv:2406.15927) for Semantic Entropy Probes; Huang et
al. (ICLR 2024, arXiv:2310.01798) on the limits of self-correction.
