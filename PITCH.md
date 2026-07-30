# Pitch script — GlassBox

**Track: AI Shield** · 3 minutes + Q&A

Before you start: browser open at `http://127.0.0.1:8000`, the four preset
buttons visible, server already running. **Turn the wifi off** — do it where
people can see. It takes three seconds and proves the whole story without
explaining anything.

---

## 0:00 — The hook (20s)

> "When a model doesn't know something, it doesn't say so. It makes something up,
> in exactly the same confident voice it uses when it's right.
>
> We don't try to tell you whether the answer is right. We tell you whether the
> model is **guessing** — and which step it started guessing on."

## 0:20 — Demo 1: it knows this (20s)

Click **"Who painted the Mona Lisa?"** — instant, it's already cached.

> "Gemma thinks out loud before it answers. We made it think seven separate
> times, then lined the seven up side by side.
>
> Most of these steps are just planning — 'I'll check what I know.' We ignore
> those. Only a real claim can be wrong.
>
> Here's the one real claim: 'the artist is Leonardo da Vinci.' All seven runs
> said the same thing. It knows this cold."

## 0:40 — Demo 2: the catch (40s)

Click **"What is the atomic radius of ununoctium?"**

> "Same confident voice. But this time it's wrong — it says element 111.
> Ununoctium is 118.
>
> Now look where the screen lights up. **Step 2.** Not 'the answer looks bad' —
> *this exact step is where it started guessing.* Across the seven runs it said
> 111 here, and 112 there, and 118 somewhere else.
>
> It couldn't keep its own story straight, and we can point at the line where it
> came apart."

Pause here. This is the moment the room gets it.

## 1:20 — Demo 3: it knows when to stop (25s)

> "So we show it the versions it came up with itself and ask which one is right —
> and we let it say 'none of these.'
>
> **It says none of these.** And it's right to: the true answer, 118, wasn't in
> the shortlist. So we flag the question and stop there.
>
> That matters. A safety tool that invents a confident correction is worse than
> one that says nothing. This one refuses to guess twice."

## 1:45 — The cost question (25s)

**Bring this up yourself, before anyone asks.**

> "Seven runs is seven times the work. That's real, and we measured it.
>
> So we also built a shortcut. There's a small model on top that reads Gemma's
> internal state and predicts 'this one's a guess' from **a single pass** — no
> repeat runs at all.
>
> **It gets that right nearly nine times in ten**, across 93 questions, for about
> a two-hundredth of the cost.
>
> So the real product is two-stage: the cheap check looks at everything, and the
> seven runs only happen on what it flags. In practice that's about 2.4× the
> work, not 7×. And it's running on your own laptop, so it costs time, not money."

## 2:10 — The numbers (25s)

Scroll to the results panel.

> "All of this is measured, not claimed.
>
> The judge scores 12 out of 12 on labelled examples. On whole answers we catch
> 92 percent of made-up ones, with no false alarms, across 24 questions. On
> individual reasoning steps we find the right one 86 percent of the time.
>
> Every cutoff in the system was chosen by sweeping labelled data, not by
> picking a number that felt right."

## 2:35 — Why this needs an open model (15s)

> "Three things here are impossible through an API.
>
> We read the judge's raw yes/no signal instead of its words — that took us from
> 10 out of 12 to 12 out of 12, and it's cheaper, because nothing gets written.
> We fix the random seed, so the seven runs are reproducible. And we can pick up
> the model's own train of thought mid-sentence and continue it.
>
> One 2.9 gigabyte model does all of it — it's the thinker and its own judge.
> Laptop, wifi off, nothing leaves the room."

## 2:50 — Close (10s)

> "The obvious hallucinations are mostly gone. What's left is the subtle kind —
> where the model half-remembers something real and fills in the rest.
>
> That's the dangerous kind. That's what GlassBox catches."

---

# Q&A — the four you'll actually get

### "Isn't this just self-consistency?"

Self-consistency samples a few answers and takes the majority vote. It needs
answers you can line up and compare, and it gives you a vote, not a diagnosis.
We compare by **meaning**, so free-form text works. We work on **reasoning
steps**, so we can say where it broke rather than just flagging the output. And
we keep the minority readings instead of throwing them away. A majority vote on
ununoctium would have returned 111 — the wrong answer.

### "Hasn't self-correction been shown not to work?"

Yes — [Huang et al., ICLR 2024](https://arxiv.org/abs/2310.01798) showed that
asking a model to reflect on its own answer often makes things worse.

We never ask it to reflect. We don't say "are you sure?" We hand it a shortlist
it wrote itself and ask it to recognise the right one. That's a different
question, and we measured it rather than assuming.

### "Why not just use a bigger model, or look things up?"

Both need something you might not have. A bigger model still makes things up,
just more smoothly. Looking things up needs a trusted source and a network
connection. GlassBox needs neither — it works offline, on questions where
there's no document to find. That's exactly the on-device case.

### "Seven runs is 7× the compute. Is that practical?"

Concede it straight away, then give the numbers.

> "Seven times, and we measured it rather than hand-waving.
>
> Two things pull it back. The judging step generates **no text at all** — we
> read the model's internal signal directly, so the most accurate part is also
> the cheapest. And reusing work between judge calls took one question from 55
> seconds to 18.5.
>
> But the real answer is the shortcut model: one pass, right nearly nine times
> in ten, about 200× cheaper. Screen everything with that, and only spend the
> seven runs on what it flags."

---

## Where this gets used

Name one real place, not a category:

> "A rural clinic running an offline medical assistant on a laptop. No internet,
> so nothing can be looked up. Patient records legally can't leave the building.
> Today that assistant has no way to signal when it's guessing about a drug
> interaction. GlassBox drops in as a gate: same model, no network — it either
> answers, warns, or declines."

## Numbers to have ready

| | |
|---|---|
| Judge accuracy | 12 out of 12 |
| Whole-answer detection | 92% caught, no false alarms, n=24 |
| **Step-level localisation** | **86% accuracy**, n=14 |
| **Single-pass shortcut** | **right ~9 times in 10**, n=93 |
| Known vs obscure questions | 0.026 vs 0.674 average spread |
| Speedup from reusing judge work | 55s → 18.5s |
| Model footprint | one 2.9 GB file — thinker and judge |
| Network traffic during demo | zero |

## Wording

- Say **"making it up"** or **"guessing"** rather than "hallucinating."
- Say the set size out loud whenever you quote a percentage.
