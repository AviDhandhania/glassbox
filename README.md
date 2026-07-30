# GlassBox — find the step where Gemma 4 starts guessing

**Track: AI Shield** · [Setup](SETUP.md) · [Kaggle writeup](WRITEUP.md) · [Pitch](PITCH.md)

Gemma 4 reasons out loud before it answers. Nobody reads those thousands of
tokens, so the reasoning is a black box wearing a glass coat.

GlassBox makes it reason **seven times**, aligns the traces step by step, and
measures where they stop agreeing. The output is not *"this answer is
unreliable"* — it is **"step 2 is where it started guessing, and here is what it
guessed instead."**

Everything runs on **one 2.9 GB Gemma 4 E2B on a laptop CPU**. Same model does the
reasoning and the judging. No API, no second model, no network.

---

## The example

> **What is the atomic radius of ununoctium in picometres?**
> Gemma answers confidently: *"ununoctium, **element 111**…"* — it is 118.

```
1. [procedure]   Identify the core request…
2. h=0.59  FORK  Identify the element: Ununoctium (Uun) is element 111
       also read as: Ununoctium (Uun) is element 112
3-6. [procedure]
→  ANSWER  h=0.39, 2 readings across 6 traces
       another trace answered: …Ununoctium (Uuo), element 118…
```

Six steps, one real assertion, and it is the one carrying the error — flagged
without any reference to the true answer, purely because the model could not tell
the same story twice.

## Results

| | |
|---|---|
| Judge accuracy | **12/12** labelled equivalence pairs |
| Answer-level detection | **92%** recall, **100%** precision (n=24) |
| Step-level localization | **86%** accuracy, 83% precision, 83% recall (n=14) |
| Single-pass probe | **AUROC 0.885** (n=93) |
| Self-repair | 0 corrections, **0 false repairs** — see below |
| Thinking-mode ablation | null result — see below |

Every threshold comes from a sweep over labelled data, and each sits inside its
winning range rather than on the edge. `traces.py --report`, `eval.py --report`
and `judgecheck` all re-derive their thresholds from saved results without
touching the model.

## How it works

1. **Seven reasoning traces** — one greedy, six at temp 0.9 with distinct seeds.
   Gemma 4 emits `<|channel>thought` … `<channel|>` and plans in numbered steps,
   so traces parse rather than needing a heuristic.
2. **Classify each step** as procedure ("I'll check my knowledge base") or
   assertion ("Ununoctium is element 111"). **Only assertions can be wrong.**
3. **Align by role** across traces — not by position.
4. **Cluster and score** with normalised Shannon entropy. 0 = every trace made the
   same move, 1 = they all differ.
5. **Score the final answer too**, as a trailing assertion. Some traces are pure
   procedure and only commit in the answer; without this they score nothing and
   the error walks through.
6. **Repair** — show the model its own competing readings and let it adjudicate,
   with "none of these" allowed.

### Why we read logits instead of text

The judge never speaks. One forward pass, then compare the raw logits of the
`YES` and `NO` tokens: `P(same) = softmax(logit_YES, logit_NO)`.

This was the single biggest correctness fix. Asked out loud, a 2B Gemma has such a
heavy `YES` prior that it called *"Leonardo painted it"* and *"Michelangelo painted
it"* the same claim — every question collapsed to one cluster and every entropy
read zero. Six prompt strategies scored between 5/12 and 10/12; asking "do these
*conflict*?" also returned YES to everything. Reading the logits gave **12/12**,
costs *less* (one prefill, zero tokens generated), and is impossible through a
hosted API.

### Why the claim filter matters

Without it the tool points at the wrong step. Procedure steps get reworded freely
between samples and scored entropy **1.0**, while the step carrying the real error
scored **0.59**. Filtering to assertions took the ununoctium trace from six noisy
steps to exactly one — the right one.

## What it costs

Measured on one question, not estimated:

| | Generated | Prefill |
|---|---|---|
| One answer with thinking | ~640 | — |
| GlassBox, 7 traces | **4,487** | — |
| 136 judge calls | **0** | 25,160 |
| Probe screening | **0** | **21** |

**Detection costs ~7× the generated tokens.** That is the honest price —
disagreement between samples *is* the measurement, so it cannot come from one
sample. Two things offset it: the judging generates **zero** tokens (logits, not
sampling), and KV-prefix reuse across judge calls cut that 55s → 18.5s.

The real answer is `probe.py`: a linear probe on the hidden state predicts
confabulation from **one forward pass** at AUROC 0.885, roughly 200× cheaper. Screen
everything, spend the seven traces only on what it flags. At a 20% flag rate that
averages ~2.4×, and locally there is no per-token bill anyway.

## Two honest nulls

**Thinking mode showed no effect.** We assumed it was responsible for Gemma 4's
refusals and ran the ablation to quantify it. Thinking off: declines all 12.
Thinking on: declines all 12. **A floor effect, not an effect** — the model already
declines these unaided. The tempting comparison (Gemma 3-without-thinking vs
Gemma 4-with-thinking) moves two variables and credits the feature for the model's
improvement.

**Self-repair does not correct.** Across 8 questions only 2 produced a divergent
step whose readings were factual claims, and one was a clean test: offered
"element 111" vs "element 112", it declined both — correctly, since ununoctium is
118 and 118 was never in its own candidate set. **n=1 is not a result.** The flaw
is structural: adjudication can only pick from what sampling produced.

What did hold is the safety property. An earlier version *forced* a choice, picked
112, and confidently rebuilt the answer around it — one wrong claim swapped for
another, output now *looking* corrected. Adding "none of these" made it fail
closed. **Zero false repairs.**

## What it does not catch

**Consistent false belief.** If all seven traces make the *same* wrong move,
entropy is zero and we call it stable. GlassBox measures whether a model is
**inventing on the spot**, not whether it is **right**. Catching a memorised
falsehood needs retrieval against a source — a different tool.

## Layout

| File | |
|---|---|
| `glassbox.py` | model layer, logit judge, entropy, trace parsing, repair, web server |
| `show.py` | terminal view of one inspection (`--repair` to adjudicate too) |
| `index.html` | the UI |
| `demo.ipynb` | clonable notebook — replays from cache, no model download |
| `traces.py` | step-level test set + fork threshold sweep |
| `eval.py` | answer-level test set |
| `repair_check.py` | self-repair measurement |
| `ablate.py` | thinking on/off study |
| `probe.py` / `bulk_label.py` | single-pass probe and its training data |
| `cachecheck.py` | guards that the demo still replays without weights |

## Running it

See [SETUP.md](SETUP.md). Short version:

```bash
python glassbox.py test         # offline self-check, no model needed
python glassbox.py judgecheck   # measure the judge, expect 12/12
python show.py "What is the atomic radius of ununoctium in picometres?"
python glassbox.py serve        # http://127.0.0.1:8000
```

One dependency (`llama-cpp-python`, prebuilt CPU wheel) and one GGUF that is both
the reasoner and the judge.

**Caches are machine-specific.** llama.cpp generation is not bit-identical across
CPUs — merging two machines' caches surfaced 5 of 1630 entries disagreeing. Results
are reproducible on one machine, not across two, so re-run `traces.py --report`
wherever you intend to demo.
