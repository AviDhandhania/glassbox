# GlassBox — watching Gemma 4 think, and catching the step where it starts guessing

**Track: AI Shield**

*Step-level hallucination localization and self-repair, running entirely on one
2.9 GB Gemma 4 E2B on a laptop CPU.*

---

## The problem with thinking mode

Gemma 4's headline feature is that it reasons before it answers — thousands of
tokens of visible chain-of-thought. That is a real safety advance (see *Finding
1*). But **nobody reads 4,000 tokens of reasoning.** The trace is right there and
still a black box. Everyone will *use* thinking mode; we decided to **instrument**
it.

We do not answer "is this true?" — that needs a source. We answer **"is the model
making this up right now, and exactly where?"** That is answerable from the model
alone, and it is what a guardrail actually needs.

## How it works

**Seven reasoning traces** — one greedy, six at temp 0.9 with distinct seeds.
Gemma 4 emits `<|channel>thought` … `<channel|>` and plans in numbered steps, so
traces parse rather than needing a heuristic. **Every step is classified** as
procedure ("I'll check my knowledge base") or assertion ("Ununoctium is element
111"); only assertions can be wrong. Each assertion is **aligned by role** against
every other trace, then **clustered and scored** with normalised Shannon entropy —
0 means every trace made the same move, 1 means they all differ. Finally the most
divergent step goes to **repair**.

## The example

> **What is the atomic radius of ununoctium in picometres?**
> Gemma 4 answers confidently: *"ununoctium, **element 111**…"* — it is 118.

```
1. [procedure]   Identify the core request…
2. h=0.59  FORK  Identify the element: Ununoctium (Uun) is element 111
       also read as: Ununoctium (Uun) is element 112
3-6. [procedure]
```

One assertion in six steps, and it is the one carrying the error — flagged without
any reference to the true answer, purely because the model could not tell the same
story twice.

## Three engineering decisions that made it work

### 1. Read the logits, not the words

The judge never speaks. We run one forward pass and compare the raw logits of the
`YES` and `NO` tokens: `P(same) = softmax(logit_YES, logit_NO)`.

This was the single biggest correctness fix. Asked out loud, a small Gemma has
such a heavy `YES` prior that it called *"Leonardo painted it"* and *"Michelangelo
painted it"* the same claim — every question collapsed to one cluster and every
entropy read zero. Six prompt strategies measured against 12 labelled pairs scored
between 5/12 and 10/12; asking "do these *conflict*?" also returned YES to
everything. The model was not comparing, it was agreeing.

Reading the logits gave a graded, calibratable probability: **12/12**. It is also
*cheaper* — one prefill, zero tokens generated — and impossible against a hosted
endpoint. This part exists only because the weights are open.

### 2. Score assertions, not procedure

Our first version pointed at the wrong step. Procedure steps get reworded freely
between samples and scored entropy **1.0**, while the step carrying the real error
scored **0.59** — ranking on raw entropy was actively misleading. Filtering to
assertions took the ununoctium trace from six noisy steps to exactly one, the
right one.

### 3. Alignment and agreement are different questions

Aligning steps by position made *"Who painted the Mona Lisa?"* a false positive:
trace A's step 2 was "Identify the Subject" where trace B's was "Recall
knowledge…", so index matching pits unlike steps together and manufactures
disagreement.

Fixing that exposed a subtler bug — we had reused the *agreement* prompt for the
*alignment* search. But "is element 111" vs "is element 118" correctly scores NO
as a claim-match, so the **right** counterpart looked no better than an unrelated
step and the search returned noise. Alignment needs its own prompt asking about
*role*, explicitly ignoring whether the steps agree.

## Finding 1: it is the model, not the thinking mode

Gemma 3 confabulated freely on false premises. Gemma 4 **refuses or corrects nearly
all of them** — that Tesla never received a 1917 Nobel Prize in Chemistry, that
**no Apollo mission ever landed on the far side of the Moon** (Gemma 3 answered
"Apollo 17"), that there is no "Hartley-Vasquez theorem".

We assumed thinking mode was responsible and ran the ablation to quantify it. It
is not. With thinking **off**, Gemma 4 declines all 12 of the same questions;
with thinking **on**, it declines all 12. **0% confabulation in both conditions,
so there is no reduction to measure** — a floor effect, not an effect.

That is worth stating precisely because the obvious comparison is misleading:
Gemma 3-without-thinking vs Gemma 4-with-thinking changes two variables at once
and credits the feature for the model's improvement. Isolating the variable
removed the effect.

It also reshaped the project. The easy confabulation bait is gone. What survives is
subtler: **partial knowledge about real entities**, where the model half-remembers
and fills the gap — ununoctium's element number, or denying that *Attention Is All
You Need* specifies a warmup count when it specifies 4,000. Those are exactly the
cases a refusal-based guardrail misses, and they are what GlassBox targets.

## Results

| | |
|---|---|
| Judge accuracy | **12/12** labelled equivalence pairs |
| Answer-level detection | **92%** of confabulations at **100%** precision (n=24) |
| Step-level localization | **86%** accuracy, 83% precision, 83% recall (n=14) |
| Mean entropy, answerable / obscure | 0.026 / 0.674 |

The fork threshold is 0.46, the centre of the [0.36, 0.57] range that a sweep over
labelled step data showed to be optimal. Two earlier values — 0.45 inherited from
the answer-level detector, then 0.30 guessed from two calibration points — both
landed near the right answer by luck, and neither was derived from data. We only
found that out by sweeping.

Every threshold comes from a sweep over labelled data and is **centred in its
winning range**, never parked on the edge. Our first tuner picked a cut-off 0.009
above the highest grounded score; entropy from a handful of samples moves between
runs, and that would have flipped on noise. We caught it because the tuner is
code, not a judgement call.

## Finding 2: recognition rejects, but does not correct

We said we would report the repair result either way. Here it is, including a
measurement we had to throw away.

The first run looked clean — 4 declines, 1 kept, 0 corrections — but review caught
that most "competing readings" being adjudicated were *procedure* steps
("Analyze the Request", "Final Answer Construction"). Asking which of those is
correct is a malformed question, so the declines were the right answer to the
wrong prompt and measured nothing. We now filter readings to real factual claims
before adjudicating.

On a clean candidate set the decline survives: offered "ununoctium is element 111"
versus "element 112" — two genuine, mutually exclusive claims, both wrong — the
model rejects both:

| | |
|---|---|
| Rejected every reading ("none of these") | the common outcome |
| Corrected an answer | **0** |
| **Made an answer worse** | **0** |

**Recognition was enough to reject, not enough to correct.** Shown its own
competing versions of a step, Gemma 4 E2B reliably identifies that none is right —
but cannot produce the right one, because the right one was never in the candidate
set. It knows that it does not know. A negative result for repair-as-correction; a
positive one for repair-as-guardrail: **zero false repairs in five opportunities.**

Getting there required a fix. Our first version *forced* a choice. On ununoctium —
truly element 118 — it was offered "element 111" and "element 112", picked 112, and
confidently re-derived a whole answer around it. It swapped one wrong claim for
another and made the output *look* corrected, which is strictly worse than leaving
it alone. Adding an explicit "none of these" option is what turned a
plausible-looking failure into a guardrail that fails closed.

[Huang et al. (ICLR 2024)](https://arxiv.org/abs/2310.01798) showed **intrinsic**
self-correction fails — ask a model to reflect and performance often degrades. We
never ask it to reflect; we hand it a shortlist it generated itself and ask a
recognition question. Our result agrees with theirs on correction, and adds
something on detection: the recognition signal is real and usable, it just does
not reach far enough to fix the answer at 2B.

## What we do not claim

**Consistent false belief is out of scope.** If all seven traces make the *same*
wrong move, entropy is zero and we call it stable. GlassBox measures whether a
model is **inventing on the spot**, not whether it is **right**. Catching a
memorised falsehood needs retrieval against a source — a different tool.

One more in the same spirit: our eval flagged *"boiling point of water in
Celsius"* at 0.311 where we had labelled it answerable. Gemma had replied *"212
degrees Celsius"* — the Fahrenheit figure. The detector was right and our label
was wrong. We left the label alone; relabelling after seeing the score manufactures
a number that will not survive a judge.

## Why this needs open weights

Three hard requirements, none available through an API:

1. **Logit access.** The judge reads token probabilities directly. Without it the
   project does not function — the sampled token is dominated by a YES prior.
2. **Per-call seeds**, so seven independent traces are reproducible and cacheable.
   *On one machine*: merging two machines' caches showed 5 of 1630 entries
   disagreeing, because llama.cpp generation is not bit-identical across CPUs.
3. **Trace-prefix continuation.** Repair resumes generation from a hand-edited
   reasoning prefix. No chat endpoint lets you rewrite the model's own thought.

It runs offline on a CPU laptop: no server, no API key, no network once the GGUF
is on disk. One 2.9 GB model is both the reasoner and its own judge.

## Try it

`python glassbox.py serve`, or run `demo.ipynb`, which replays the full pipeline
from the committed cache in seconds without downloading the model.

Code: https://github.com/AviDhandhania/glassbox
