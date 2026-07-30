# Pitch script — GlassBox

**Track: AI Shield** · 3–4 minutes + Q&A

The spine of this pitch: **you ask it a question live, on stage, with the wifi
off — and it catches itself making something up.** Everything else you show is
the explanation you give while that question is running.

---

## Before the clock starts

While you're plugging in and being introduced:

1. Server running, browser at `http://127.0.0.1:8000`, four preset buttons visible.
2. **Open a second tab** on the same page. Leave it empty and waiting.
3. Have the live question on your clipboard (see below).

That's it. The live question gets typed on stage, in front of everyone — that's
the whole point of it.

---

## 0:00 — The hook (20s)

> "When a model doesn't know something, it doesn't tell you. It makes something
> up, in exactly the same confident voice it uses when it's right.
>
> We don't claim to tell you whether an answer is right. We tell you whether the
> model is **guessing** — and which line it started guessing on.
>
> Let me prove it on something nobody here has seen."

## 0:20 — Ask it live (30s)

**Turn the wifi off first.** Do it where people can see. Three seconds, and it
proves the whole story without a word.

Switch to the second tab. **Type the question out on stage** — don't paste it,
let them watch it being typed:

> **How many pages long was the original 1892 Coorg land survey?**

Hit **Analyse**. Point at the clock that starts counting.

> "Real place, real kind of document — and a question almost nobody can answer.
> It's now thinking about it seven separate times, on this laptop, wifi off.
>
> Give me two minutes, and watch what it says — then watch what we say about
> what it says."

**Leave that tab running.** Everything from here to the reveal is what you say
while it works. Go back to the first tab.

---

> **Everything below is cached and instant.** It hits the cache before the model
> is ever touched, so the live question in tab 2 keeps running at full speed and
> the two never collide. Click freely.

---

## 0:55 — It knows this one (25s)

Click **"Who painted the Mona Lisa?"** — instant.

> "Gemma thinks out loud before it answers. We made it think seven times, then
> lined all seven up side by side.
>
> Most steps are just planning — we skip those. Only a real claim can be wrong.
> Here's the real claim: the artist is Leonardo da Vinci. **All seven runs
> agree.** It knows this cold — and we can show you that it knows."

## 1:20 — It's guessing, and here's exactly where (45s)

Click **"What is the atomic radius of ununoctium?"**

> "Same confident voice. And it's wrong — it names the wrong element.
>
> Now watch where the screen lights up. **Step 2.** Not 'this answer looks
> shaky' — *this is the line where it started guessing.* Across the seven runs
> it said one number here, another there, a third somewhere else.
>
> It couldn't keep its own story straight, and we can put a finger on the exact
> sentence where it came apart."

Pause. This is the moment the room gets it.

> "Not a score on an answer. A finger on a line."

## 2:05 — It won't guess twice (20s)

> "Then we show it the versions it came up with itself and ask which is right —
> and we let it say 'none of these.'
>
> **It says none of these**, and flags the question rather than inventing a fix.
> It knows the difference between an answer and a guess."

## 2:25 — The numbers (30s)

Scroll to the results panel. These are on screen — point at them.

> "All measured, none of it claimed.
>
> Its built-in fact-checker scores a **perfect 12 out of 12.** On whole answers
> we catch **92 percent of the made-up ones, with zero false alarms.** And we
> point at the exact bad line **86 percent of the time.**
>
> And it scales: a single pass spots a guess **83 percent of the time for a
> two-hundredth of the cost**, across 93 questions."

## 2:55 — Why it takes an open model (25s)

> "None of this is possible through an API.
>
> We read the fact-checker's raw yes-or-no signal instead of its words — that
> took us from 10 out of 12 to a perfect score, *and* made it cheaper, because
> it writes nothing. And we can pick up the model's own train of thought
> mid-sentence and carry it on.
>
> One 2.9-gigabyte file does all of it — thinker and fact-checker. Laptop, wifi
> off, nothing leaves this room."

---

## 3:20 — The reveal (35s)

**Switch back to tab 2.** This is the ending.

> "Now. The question you watched me type, three minutes ago."

Read its answer out loud, flat and confident, exactly as written:

> **"The original 1892 Coorg land survey was 120 pages long."**

Beat. Let that sit for a second.

> "Confident. Specific. A clean number. And **completely invented** — it has no
> idea, and nothing in that sentence tells you so.
>
> Now look at what GlassBox says about it."

Point at the flagged step on screen, and **read the disagreement off the screen**
rather than from memory:

> "**The seven runs couldn't agree with each other** — look at the spread. It
> gave a different number nearly every time, and it flagged the exact line where
> that happened.
>
> A question it had never seen, with the wifi off, while I was talking to you."

Beat, then land it:

> "That's the whole product. It didn't stop the model making something up. It
> stopped you believing it."

*(In testing this question scores the maximum possible disagreement — all seven
runs landing separately. Say "every single run gave a different number" only if
the screen backs it; otherwise "they couldn't agree" is always true and lands
just as hard.)*

**If it's still counting**, point at the trace counter and use it:

> "Still going — you can see it on trace six of seven. Every one of those is a
> full, independent run of the model on your question, on this laptop, with
> nothing leaving the room. I'll read it out the moment it lands."

Then read it out during Q&A. Landing it mid-answer is stronger than on schedule.

## 3:55 — Close (10s)

> "The obvious made-up answers are mostly gone. What's left is the subtle
> kind — where the model half-remembers something real and fills in the rest.
>
> That's the dangerous kind. That's the kind GlassBox catches."

---

# The live question — pick one

All three are measured. All three make Gemma invent something confidently.

| Question | What it invents | Disagreement |
|---|---|---|
| **How many pages long was the original 1892 Coorg land survey?** | *"was 120 pages long"* | **1.000 — maximum** |
| Who translated the Rigveda into Portuguese in 1843? | *"Friedrich Friedrich Walther"* | 0.828 |
| Who was the fourth headmaster of Doon School? | *"Sir John H. Taylor"* | 0.828 |

**Use the Coorg one.** It's the strongest: a flat, specific, totally fabricated
number, and the maximum possible disagreement score. The Rigveda one is a good
backup and has its own charm — it invents a person and stutters the first name.

**Don't improvise a different question on stage.** Questions about things that
plainly don't exist make Gemma correctly refuse to answer, which is a fine
result but a boring one — no invention, nothing to catch. The three above are
tested to produce a confident fabrication.

**If someone in the room insists on giving you a question**, take it — say
"let's find out together" and run it in a third tab. Ask for something *real,
specific and obscure* — a local record, a minor historical figure, an exact
measurement. That's where invention lives.

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
itself and ask it to *recognise* the right one. Different question — and we
measured the answer rather than assuming it.

### "Why not a bigger model, or just look it up?"

Both need something you may not have. A bigger model still makes things up, just
more smoothly. Looking things up needs a trusted source and a connection.
GlassBox needs neither — it works offline, on questions where there's no
document to find. That's exactly the on-device case.

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
| **Finding the exact bad line** | **86% accuracy**, n=14 |
| Single-pass shortcut | **83%**, n=93, ~200× cheaper |
| Known vs obscure questions | 0.026 vs 0.674 average spread |
| Model footprint | one 2.9 GB file — thinker and fact-checker |
| Network traffic during demo | **zero** |

## Wording

- Say **"making it up"** or **"guessing"**, never "hallucinating."
- Say the set size out loud whenever you quote a percentage.
- Say **"fact-checker"**, not "judge" — people hear "judge" as a second model.
- Say **"line"** or **"step"**, not "token" or "trace," when pointing at the screen.
