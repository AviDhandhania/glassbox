# Pitch script — GlassBox

**Track: AI Shield** · 3 minutes + Q&A

Before you start: browser at `http://127.0.0.1:8000`, four preset chips visible,
`python glassbox.py serve` already running. **Turn the wifi off.** Do it visibly —
it costs three seconds and it proves the whole edge story without a sentence of
explanation.

---

## 0:00 — The hook (20s)

> "When a model doesn't know something, it doesn't tell you. It makes something up
> in exactly the same confident voice it uses for things it knows.
>
> A 2026 audit of deployed medical GPTs found 25 to 30 percent had low factual
> accuracy. Those systems sound identical whether they're right or inventing.
>
> We don't try to tell you whether Gemma is *right*. We tell you whether it's
> **making it up right now** — and exactly which step it started on."

## 0:20 — Demo part 1: it knows this (20s)

Click **"Who painted the Mona Lisa?"** — instant, it's cached.

> "Gemma 4 reasons out loud before answering. We made it reason seven separate
> times. Six of these steps are just procedure — 'I'll check my knowledge base' —
> and we don't score those, because only a claim can be wrong.
>
> One real assertion: 'the artist is Leonardo da Vinci.' Entropy zero. All seven
> runs agree. It knows this cold."

## 0:40 — Demo part 2: the catch (40s)

Click **"What is the atomic radius of ununoctium?"**

> "Same confident voice. And it's wrong — it says element 111. Ununoctium is 118.
>
> But look where it lights up. **Step 2.** Not 'the answer is bad' — *this step,
> right here, is where it started guessing.* Across seven traces it said 111, 112,
> and 118.
>
> And look at that." — point at 118 — "**The correct answer was already in there.**
> It was a minority reading among its own attempts. The model had it and picked
> wrong."

Pause on that. It's the moment the room gets it.

## 1:20 — Demo part 3: it fails closed (25s)

**Measured position — say exactly this, no more:**

> "So we hand it back its own two readings — 111 and 112 — and ask which is right,
> with 'none of these' allowed.
>
> **It rejects both.** Correctly: the answer is 118, and 118 was never in its own
> candidate set. So we flag it and stop.
>
> An earlier version *forced* a choice. It picked 112 and confidently rebuilt the
> whole answer around it — one wrong claim swapped for another, and now it *looks*
> corrected. That's worse than doing nothing. A guardrail has to fail closed."

**Do not claim repair fixes answers.** It never did, in 8 questions. Only one
produced a clean factual adjudication, so there is no measurement — and the
writeup says so. If asked "does it repair?":

> "No. We built it, measured it, and it doesn't — the true answer usually isn't in
> the model's own candidate set, so there's nothing to pick. What it does reliably
> is refuse to invent a fix. Zero false repairs."

## 1:45 — The cost answer (20s)

> "Seven traces sounds expensive. So we implemented Semantic Entropy Probes —
> Kossen et al. — and trained a linear probe on the hidden state to predict
> confabulation from **one forward pass**, no sampling.
>
> **AUROC 0.885 across 93 examples.** It ranks a confabulating answer above a
> grounded one nearly nine times in ten. So you screen everything in one pass and
> spend the seven traces only on what gets flagged."

Quote **AUROC, not accuracy** — the set is 76% negative, so accuracy (83%) only
beats the majority baseline by 7 points and is easy to wave away. AUROC is
threshold-independent.

## 1:50 — The numbers (30s)

Scroll to the eval panel.

> "Everything is measured, not vibes.
>
> The judge scores 12 out of 12 on labelled pairs. Answer-level detection catches
> 92 percent of confabulations at 100 percent precision across 24 questions.
>
> Every threshold comes from a sweep over labelled data and sits in the *middle*
> of its winning range — our first tuner picked a cut-off nine thousandths above
> real data and would have flipped on noise. We caught that because the tuner is
> code, not a judgement call."

## 2:20 — Why this needs Gemma (25s)

> "Three things here are impossible against an API.
>
> One — we never read the judge's words. A 2B model says YES to almost anything;
> it told us 'Leonardo painted it' and 'Michelangelo painted it' were the same
> claim. So we skip the text and read the raw YES/NO **logits** underneath. That
> took us from 10 out of 12 to 12 out of 12, and it's *cheaper* — one forward
> pass, zero tokens generated.
>
> Two — seeded sampling, so seven independent traces are reproducible.
>
> Three — repair rewrites the model's own thought and resumes generation from it.
> No chat endpoint lets you do that.
>
> One 2.9 gigabyte model. It's the reasoner and its own judge. Laptop CPU, wifi
> off, nothing leaves the machine — which is the only way this is deployable in a
> clinic or a law office."

## 2:45 — Close (15s)

> "Thinking mode already killed the easy hallucinations — Gemma 4 correctly refuses
> false premises that Gemma 3 confabulated on. What survives is subtler: partial
> knowledge on real entities, where it half-remembers and fills the gap.
>
> That's the dangerous kind, and that's what GlassBox catches."

---

# Q&A — the five you will actually get

### "Isn't this just self-consistency?"

Self-consistency samples N answers and takes the majority — it needs answers you
can compare, and it gives you a vote, not a diagnosis. We do three things it
doesn't: we compare by **meaning** using a logit-read judge so free-form text
works; we operate on **reasoning steps**, so we localise *where* it broke rather
than just flagging the output; and we keep the minority readings, which is what
makes repair possible. Majority voting on ununoctium would have returned 111 —
the wrong answer. Ours surfaced 118.

### "Self-correction has been shown not to work."

Correct, and that's the right paper to cite —
[Huang et al., ICLR 2024](https://arxiv.org/abs/2310.01798) showed **intrinsic**
self-correction fails: ask a model to reflect on its own answer and performance
often *degrades*.

We never ask it to reflect. We don't say "are you sure?" We hand it a shortlist of
readings **it generated itself** at temperature 0.9 and ask a recognition question.
That's adjudication over self-generated evidence, not introspection. Whether it
works at 2B is an empirical question and we measured it rather than assuming.

### "What if the model is confidently wrong every single time?"

Then we miss it, and we say so up front. All seven traces agree, entropy is zero,
we call it stable. GlassBox detects **invention**, not **error**. A memorised
falsehood is stable; a confabulation is not. Catching the first kind needs
retrieval against a real source — a different tool. We're complementary to RAG,
not a replacement, and RAG needs a source to exist in the first place.

### "Why not just use a bigger model, or RAG?"

Both need something we may not have. A bigger model still hallucinates, just more
fluently. RAG needs a trustworthy corpus and a network round-trip. GlassBox needs
neither — it works offline on questions with no document to retrieve, which is
exactly the on-device case. And it's *free* at the margin: local inference has no
per-token cost, so seven traces costs time, not money.

### "Seven traces is 7× the compute. Is that practical?"

For the reasoning, yes — it's the honest cost, and locally that's wall-clock, not
dollars. But the judging, which is the bulk of the calls, is **prefill-only**: we
generate zero tokens and read logits directly. We also reuse the KV cache across
judge calls, since every judge prompt shares a fixed preamble — that alone took
judging from 55s to 18.5s, a 3× speedup, verified to still score 12/12.

---

## Where this actually gets used

Name one concrete deployment, not a category:

> "A rural clinic running an offline medical assistant on a laptop. No internet,
> so no RAG and no API. Patient data legally can't leave the building. Today that
> assistant has no way to signal when it's guessing about a drug interaction.
> GlassBox is a drop-in gate: same model, no network, and it either answers,
> warns, or refuses."

That's the pitch for **why on-device hallucination detection is a distinct
problem** — every cloud mitigation assumes a network and a corpus, and this
setting has neither.

## Numbers to have on the tip of your tongue

| | |
|---|---|
| Judge accuracy | 12/12 labelled pairs |
| Answer-level detection | 92% recall, 100% precision, n=24 |
| **Step-level localization** | **86% accuracy**, 83% precision/recall, n=14 |
| **Single-pass probe** | **AUROC 0.885**, n=93 |
| Self-repair | 0 corrections, **0 false repairs** — not measurable, see writeup |
| Thinking mode ablation | 0% confabulation on *and* off — floor effect, no result |
| Mean entropy, known vs obscure | 0.026 vs 0.674 |
| Judge speedup from KV reuse | 55s → 18.5s (3×) |
| Model footprint | one 2.9 GB GGUF, reasoner + judge |
| Network traffic during demo | zero |

## Do not say

- "It detects hallucinations" — say *confabulation*, and name the blind spot.
- Any repair claim Arc hasn't confirmed.
- "It's 100% accurate." Precision is 100% on our 24-question set. Say the set size.
