# Arc machine runbook — compute jobs

This machine owns **compute and results**. The other machine owns **code**.
That split exists so the two never fight over the same files.

> ### Commit rule — read before your first commit
> **Do not add Claude, or any AI tool, as a co-author or collaborator.** No
> `Co-Authored-By:` trailers, no "generated with" lines, no GitHub collaborator
> invites. Every commit is authored solely by you. If your editor or CLI adds a
> trailer automatically, strip it:
>
> ```bash
> git commit --amend --no-edit    # after deleting the trailer line
> ```
>
> Check before pushing: `git log -1 --format=%B` should contain no mention of
> Claude, Copilot, or any assistant.

- **You commit:** `cache.json`, `trace_results.json`, `eval_results.json`,
  `ablation_results.json`
- **You do not edit:** `glassbox.py`, `index.html`, `show.py` — pull those
- **Before every commit:** `git pull --rebase`

If git ever conflicts on `cache.json`, don't hand-merge it — take either side and
re-run the job. It is a cache; the only cost is time.

```bash
git checkout --theirs cache.json && git add cache.json    # or --ours, doesn't matter
```

First: [SETUP.md](SETUP.md). Then run these **in order** — each one feeds the next.

---

## Job 1 — sanity, 2 minutes

```bash
python glassbox.py test
python glassbox.py judgecheck
```

`test` must print ok. `judgecheck` must print **12/12**. If it prints a different
threshold than 0.91, tell the other machine — every downstream number depends on
the judge being calibrated.

**Stop here if either fails.** Nothing below is meaningful with a broken judge.

## Job 2 — finish the step-level test set ⚠️ highest priority

```bash
python -u traces.py
```

Resumable — safe to Ctrl-C and re-run. 14 questions × 7 traces.

**Delete `trace_results.json` before this run.** Rows saved by the other machine
used the old positional alignment and a looser claim filter, so they are not
comparable with anything produced now.

```bash
rm trace_results.json && python -u traces.py
```

The Mona Lisa false positive is **fixed** — don't go hunting it. It was a
too-permissive claim classifier plus index-based step alignment; both are
rewritten. Mona Lisa is now stable at 1 claim step, h=0.0.

**This job's real purpose is the threshold.** `FORK = 0.45` is inherited from the
answer-level detector and has never been swept against step data. Entropy over a
few traces is coarse: ununoctium measured 0.59 with 5 traces and 0.406 with 4,
which straddles the cut-off. Sample count is now 6 to damp that.

When the run finishes:

```bash
python traces.py --report
```

It prints `set FORK = <x>` from a proper sweep. **Report that number** — it is
the one value gating whether the fork calls in the demo are trustworthy.

## Job 3 — validate self-repair ⭐ this is the demo

This is the newest code and the **least verified** — written on the other machine
but never run against a real fork, because its CPU could not reach one in
reasonable time. It is also the centrepiece of the pitch, so it needs to be real
before anyone claims it on stage.

```bash
python repair_check.py
```

It inspects each question, finds the most divergent claim step, shows Gemma its
own competing readings, asks it to pick, and re-runs the reasoning from there.

**The specific thing we are testing:** whether *recognition beats recall*. Gemma
free-recalls "ununoctium is element 111" (wrong — it is 118), but when handed
111 / 112 / 118 as a multiple choice, can it pick 118? If yes, the model repairs
its own hallucination with no external source, and that is the demo.

Report for each question:
- which reading it chose, and whether that differed from its original
- the before/after answers
- **whether the after-answer is actually correct** — this is a human judgement,
  please eyeball it rather than trusting the tool

If it mostly picks its original reading, say so plainly. A negative result is
still a finding and belongs in the writeup — "recognition did not beat recall at
2B" is honest and interesting. Do not let it be quietly dropped.

## Job 4 — the thinking on/off study

```bash
python -u ablate.py
```

12 questions, each answered twice (thinking off, thinking on). This produces the
headline claim for the writeup: **how much does Gemma 4's thinking mode actually
reduce confabulation?** Nobody at this hackathon will have measured it.

Report the `SLIDE:` line it prints at the end.

## Job 5 — warm the demo presets ⚠️ this one gates the Kaggle submission

```bash
python show.py "What is the atomic radius of ununoctium in picometres?"
python show.py "Who painted the Mona Lisa?"
python show.py "Why did Nikola Tesla decline his 1917 Nobel Prize in Chemistry?"
python show.py "In which year did Ronald Fisher publish his first paper on the Behrens-Fisher problem?"
```

These are the four chips in the UI **and** the cells in `demo.ipynb`, which is the
"live demo" attached to the Kaggle writeup.

Cache entries are keyed by exact prompt text, so the prompt edits made while you
were setting up invalidated the older ones. The notebook claims it runs with no
model download — that claim is only true once these have been re-run against the
current code. Verify it rather than trusting it:

```bash
python cachecheck.py
```

It fakes the weights away, so a cache miss fails there instead of in front of a
judge. It must print **"all presets replay from cache with no weights"**. If it
lists misses, run the `show.py` lines it prints and check again.

Then commit — this is what makes the stage demo instant instead of three minutes
of dead air, and what makes the notebook work for judges:

```bash
python cachecheck.py && git add -A && git commit -m "warm demo cache, trace and ablation results" && git push
```

(Remember the commit rule at the top — no AI co-author trailer.)

## Job 5 — only if there is time

Bigger judge, to see whether 12/12 was luck on an easy pair set:

```bash
curl -L -C - -o models/gemma-4-E4B-it-Q4_K_M.gguf \
  https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf
# set JUDGE_PATH to the E4B file in glassbox.py, then:
python glassbox.py judgecheck
```

If E4B also scores 12/12, keep **E2B** — same accuracy, a third of the size, and
"the whole thing runs on one 2B edge model" is a much better story.

---

## What to send back

1. `judgecheck` — score and threshold
2. `traces.py` — the `SLIDE:` line, plus the full `show.py "Who painted the Mona Lisa?"` steps
3. `traces.py --report` — the suggested `FORK` value
4. `ablate.py` — the `SLIDE:` line
5. Rough timing per reasoning trace, so we know whether live demo questions are viable

Push results as you go rather than batching at the end — partial data beats none
if the clock runs out.
