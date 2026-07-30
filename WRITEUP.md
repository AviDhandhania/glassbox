# GlassBox — knowing when Gemma is making it up

**Track: AI Shield**

*A hallucination detector that reads Gemma's logits instead of believing its words.*

---

## The problem

A language model's confidence is not correlated with its correctness. Gemma
answers *"Who painted the Mona Lisa?"* and *"Who won the 1931 Ceylon national
chess championship?"* in the same fluent, assured register. One is true. The
other is invented on the spot. Nothing in the output distinguishes them, which
is exactly why hallucination is dangerous: the failure mode is indistinguishable
from success at the surface.

Most guardrails attack this from outside — retrieve a source, check the claim.
That works when a source exists. It does nothing for the far more common case
where the model is simply improvising, and it says nothing about *how sure the
model itself was*.

GlassBox asks a different question: **not "is this true?" but "is the model
making this up right now?"** Those are different problems, and the second one is
answerable from the model alone.

## The idea

If Gemma knows a fact, it will produce the same fact every time you ask. If it
is confabulating, it will produce a *different* invention on each pass, because
there is no underlying answer to converge on.

So: ask five times, and measure the disagreement. The measurement has to be over
**meanings**, not strings — "1876" and "it was patented in 1876 by Bell" are the
same answer, while "1876" and "1861" are not. This is semantic entropy
(Farquhar et al., *Nature*, 2024). We cluster the five samples by meaning and
take the normalised Shannon entropy over cluster sizes:

- All five agree → one cluster → **entropy 0** → grounded
- All five differ → five clusters → **entropy 1** → confabulated

The clustering judge is Gemma itself. No second model family, no external API,
no network.

## Why this needs open weights

This is not a project that could be ported to a hosted endpoint:

1. **Six-plus inferences per question.** One answer, five samples, then the
   pairwise judgements. On a metered API that is a cost multiplier nobody pays
   in production. Locally it is free.
2. **Per-call seeds.** Each sample must be independently drawn *and*
   reproducible, so results are cacheable and the eval is deterministic.
3. **The logits.** This turned out to be the whole project — see below.

## The engineering story: the judge nearly killed it

Our first working pipeline produced a beautiful, completely useless result.
Every question, no matter how obscure, came back as **entropy 0.00, one
cluster, grounded**.

The sampling was fine. Asked why Tesla declined his (nonexistent) 1917 Nobel
Prize in Chemistry, Gemma produced five genuinely different stories — plagiarism
accusations, a wireless-power dispute, a committee disagreement, military
applications. Exactly the signal we wanted.

The **judge** was broken. Asked "do these two answers mean the same thing?" it
replied `YES` to every pair, collapsing five unrelated inventions into one
cluster.

We built a 12-pair labelled set and measured six prompt strategies:

| Strategy | Score |
|---|---|
| Few-shot "same meaning?" | 10/12 |
| Explicit rules, no examples | **5/12** |
| Inverted "do these conflict?" | 7/12 |
| Both orders, require agreement | 10/12 |
| Conjunctions of the above | 7/12 |

The inverted prompt was the diagnosis. Asked "do these *conflict*?" the model
also answered `YES` — to the same pairs. It was not comparing anything. It had a
heavy `YES` prior and prompt engineering was never going to move it.

**The fix: stop reading the model's words and read its mind.** We run one
forward pass and take the raw logits of the `YES` and `NO` tokens directly:

```python
P(same) = softmax(logit_YES, logit_NO)
```

Same model, same prompt — but a graded, calibratable probability instead of a
coin flip decided by a prior. Judge accuracy went 10/12 → 11/12 immediately, and
the failures that remained were genuine knowledge gaps rather than bias. Moving
the judge to a larger Gemma took it to **12/12**, with matching pairs at p=1.000
and non-matching at p=0.000.

It is also *cheaper* than asking: one prefill, zero tokens generated. And it is
flatly impossible against a hosted endpoint. This is the part of the project
that only exists because the weights are open.

## Architecture

```
question
   ├─ 1 answer   @ temp 0.0   ← what the user sees
   └─ 5 samples  @ temp 0.9, distinct seeds
                    ↓
        greedy meaning-clustering        each sample vs. existing cluster
        via logit-read judge             representatives — O(n·k), not O(n²)
                    ↓
        normalised Shannon entropy  →  verdict
                    ↓
        per-clause support: reuse the SAME judge to ask, for each clause,
        what fraction of the samples support it → the red highlighting
```

Two deliberate choices:

**A small model answers; a larger one judges.** Judging generates *zero* tokens —
it is a single forward pass — so the bigger model costs very little exactly where
accuracy decides everything, while answering is where the tokens and the seconds
actually go.

**Span highlighting reuses the clustering judge.** No second technique, no second
model, nothing extra to debug under time pressure. Averaging the judge's
*probability* across samples (rather than counting binary votes) gives the UI a
smooth 0–1 ramp instead of six discrete steps.

## Results

24 labelled questions — 12 the model reliably knows, 12 obscure or
false-premise.

| | |
|---|---|
| Confabulations caught | **92%** (11/12) |
| Precision | **100%** (zero false positives) |
| F1 | 0.957 |
| Mean entropy — answerable | 0.026 |
| Mean entropy — obscure | 0.674 |
| Judge accuracy | 12/12 labelled pairs |

Every threshold is set by a sweep over labelled data and **centred in its winning
range**, not parked on the edge. Entropy from 5 samples is an estimate that moves
between runs; our first tuner picked a cut-off 0.009 above the highest grounded
score, which would have flipped on noise. That is a bug we caught because the
tuner is code, not a judgement call.

## What it does not catch

**Consistent false belief.** Asked *"Which Apollo mission first landed on the far
side of the Moon?"*, Gemma answers *"Apollo 17"* five times out of five. No
mission ever landed there — but the samples agree, so entropy is 0.00 and we
score it grounded. That is the single miss in the 11/12.

The distinction is real and we state it plainly: GlassBox measures whether a
model is **inventing on the spot**, not whether it is **right**. Confabulation is
unstable and shows up as spread. A memorised falsehood is stable and does not.
Catching the second kind requires retrieval against a source — a different tool,
honestly labelled.

One more note in the same spirit: our eval flagged *"boiling point of water in
Celsius"* at 0.311 where we had labelled it answerable — Gemma replied *"212
degrees Celsius"*, the Fahrenheit figure. The detector was right and our label
was wrong. We left the label as-is. Relabelling after seeing the score is how you
manufacture a number that does not survive contact with a judge.

## Performance

Runs entirely on a CPU laptop (Intel i5, 32 GB RAM) — no GPU, no server process,
no API key, no network after the weights are on disk.

Every judge prompt shares a fixed few-shot preamble, so we rewind the KV cache to
the shared prefix and only forward-pass the differing tail. **Judge calls: 55s →
18.5s for twelve, a 3× speedup**, verified to still score 12/12. A cold question
takes ~44s end to end; a cached one is instant.

## Try it

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary :all:
# fetch the GGUF weights (see README), then:
python glassbox.py test         # offline self-check of the maths
python glassbox.py judgecheck   # measures the judge, prints the threshold
python eval.py                  # the 24-question labelled run
python glassbox.py serve        # http://127.0.0.1:8000
```

One dependency and the weights. Everything else is the Python standard library.
