# Pitch script — GlassBox

**Track: AI Shield** · 3 minutes + Q&A

---

## Before the clock starts

While you're plugging in and being introduced:

1. Server running, browser at `http://127.0.0.1:8000`, four preset buttons visible.
2. **Turn the wifi off** where people can see it. Three seconds, and it proves the
   whole story without a word.
3. **Open a second tab. Ask the room for a question** — "give me something factual
   and obscure" — type it in, hit Analyse, and leave that tab running.

That third step matters. Their question is really being answered, live, on this
laptop, and it'll be ready by the time you come back to it at 2:30. Everything
you show before then is cached and instant, so the two never collide.

---

## 0:00 — The hook (20s)

> "When a model doesn't know something, it doesn't tell you. It makes something
> up, in exactly the same confident voice it uses when it's right.
>
> We don't claim to tell you whether an answer is right. We tell you whether the
> model is **guessing** — and which step it started guessing on."

## 0:20 — It knows this one (20s)

Click **"Who painted the Mona Lisa?"** — instant.

> "Gemma thinks out loud before answering. We made it think seven separate times,
> then lined the seven up side by side.
>
> Most steps are just planning — 'I'll check what I know.' We skip those. Only a
> real claim can be wrong. Here's the real claim: the artist is Leonardo da
> Vinci. All seven runs agree. It knows this cold."

## 0:40 — It's guessing, and here's where (45s)

Click **"What is the atomic radius of ununoctium?"**

> "Same confident voice. But it's wrong — it names the wrong element number.
>
> Now watch where the screen lights up. **Step 2.** Not 'this answer looks
> shaky' — *this is the step where it started guessing.* Across the seven runs it
> said one number here, another there, a third somewhere else.
>
> It couldn't keep its own story straight, and we can point at the line where it
> came apart. Not a score on the answer — a finger on the sentence."

Pause. This is the moment the room gets it.

## 1:25 — It won't guess twice (20s)

> "Then we show it the versions it came up with itself and ask which one is
> right — and we let it say 'none of these.'
>
> **It says none of these**, and flags the question instead of inventing a fix.
> A safety tool that confidently corrects you wrongly is worse than one that
> stays quiet. This one knows when to stop."

## 1:45 — The numbers (25s)

Scroll to the results panel.

> "All measured, none of it claimed.
>
> Its built-in fact-checker scores a perfect 12 out of 12. On whole answers we
> catch 92 percent of the made-up ones **with zero false alarms**. And we point
> at the exact bad step 86 percent of the time.
>
> Every cutoff was chosen by sweeping labelled data, not by picking a number that
> felt about right."

## 2:10 — Why it takes an open model (20s)

> "Three things here are impossible through an API.
>
> We read the fact-checker's raw yes-or-no signal instead of its words — that took
> us from 10 out of 12 to a perfect score, *and* made it cheaper, because it
> writes nothing. We lock the randomness, so the seven runs are reproducible. And
> we can pick up the model's own train of thought mid-sentence and carry it on.
>
> One 2.9-gigabyte file does all of it. Laptop, wifi off, nothing leaves this
> room."

## 2:30 — Your question, live (20s)

Switch to the second tab.

> "Everything so far was prepared. This wasn't — it's the question you gave me
> four minutes ago, running on this laptop, wifi off, the whole time we've been
> talking."

Read the verdict out loud. One sentence, no more:

> "Seven runs, and it [agreed with itself / came apart at step *n*] — a question
> we've never seen, and it told you which part to trust."

**If it's still working**, point at the running clock and keep going:

> "Still thinking — seven full runs on a laptop CPU takes a few minutes, and it's
> counting them off as it goes. Which is the honest trade: a couple of minutes of
> your own hardware, and nothing about that question ever left the building."

Come back to it in Q&A. It'll finish mid-answer, and reading it out then lands
even better than on schedule.

## 2:50 — Close (10s)

> "The obvious made-up answers are mostly gone. What's left is the subtle kind,
> where the model half-remembers something real and fills in the rest.
>
> That's the dangerous kind. That's the kind GlassBox catches."

---

# Q&A — the three you'll actually get

### "Isn't this just asking it a few times and taking a vote?"

A vote needs answers you can line up and compare, and it gives you a winner, not
a diagnosis. We compare by **meaning**, so free-form text works. We work on the
**reasoning steps**, so we say *where* it broke. And we keep the minority
readings instead of discarding them — a straight vote on the ununoctium question
would have confidently returned the wrong answer.

### "Doesn't asking a model to check itself make things worse?"

It does, when you ask it to reflect — [Huang et al., ICLR 2024](https://arxiv.org/abs/2310.01798)
showed that. So we never ask "are you sure?" We hand it a shortlist it wrote
itself and ask it to *recognise* the right one. Different question, and we
measured the answer rather than assuming it.

### "Why not a bigger model, or just look it up?"

Both need something you may not have. A bigger model still makes things up, just
more smoothly. Looking things up needs a trusted source and a connection.
GlassBox needs neither — it works offline, on questions where there's no document
to find. That's exactly the on-device case.

---

## Where this lands

Name one real place, not a category:

> "A rural clinic running an offline medical assistant on a laptop. No internet,
> so nothing can be looked up. Patient records legally can't leave the building.
> Today that assistant has no way to signal when it's guessing about a drug
> interaction. GlassBox drops in as a gate: same model, no network — it either
> answers, warns, or declines."

## Numbers to have ready

| | |
|---|---|
| Internal fact-checker | **12 out of 12** |
| Whole-answer detection | **92% caught, zero false alarms**, n=24 |
| **Finding the exact bad step** | **86% accuracy**, n=14 |
| Single-pass shortcut | right ~83% of the time, n=93, ~200× cheaper |
| Known vs obscure questions | 0.026 vs 0.674 average spread |
| Model footprint | one 2.9 GB file — thinker and fact-checker |
| Network traffic during demo | zero |

## Wording

- Say **"making it up"** or **"guessing"**, never "hallucinating."
- Say the set size out loud whenever you quote a percentage.
- Say **"fact-checker"**, not "judge" — people hear "judge" as a second model.
