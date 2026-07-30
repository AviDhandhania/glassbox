# Arc machine runbook — compute jobs

This machine owns **compute and results**. The other machine owns **code**.
That split exists so the two never fight over the same files.

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

Resumable — safe to Ctrl-C and re-run. 14 questions × 5 traces.

**There is a known bug waiting in here.** The first result on the other machine
was a false positive: *"Who painted the Mona Lisa?"* reported `FORK@2` when it
should be stable, with 4 of 6 steps classified as claims. When the run finishes:

```bash
python show.py "Who painted the Mona Lisa?"
```

Paste that output back. The question is whether the claim classifier is calling
procedure steps "claims", or whether the step judge is splitting on pure
rephrasing. That diagnosis decides the fix, so don't guess at it — send the steps.

Then re-tune the threshold from the saved data, no regeneration needed:

```bash
python traces.py --report
```

It prints `set FORK = <x>`. Report the number.

## Job 3 — the thinking on/off study

```bash
python -u ablate.py
```

12 questions, each answered twice (thinking off, thinking on). This produces the
headline claim for the writeup: **how much does Gemma 4's thinking mode actually
reduce confabulation?** Nobody at this hackathon will have measured it.

Report the `SLIDE:` line it prints at the end.

## Job 4 — warm the demo presets

```bash
python show.py "What is the atomic radius of ununoctium in picometres?"
python show.py "Who painted the Mona Lisa?"
python show.py "Why did Nikola Tesla decline his 1917 Nobel Prize in Chemistry?"
python show.py "In which year did Ronald Fisher publish his first paper on the Behrens-Fisher problem?"
```

These are the four chips in the UI. Running them writes their results into
`cache.json`, so **commit the cache afterwards** — that is what makes the live
demo instant instead of 3 minutes of dead air on stage.

```bash
git add -A && git commit -m "arc: warm demo cache + trace results" && git push
```

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
