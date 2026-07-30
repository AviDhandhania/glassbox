# GlassBox — hallucination detection for Gemma

Track: **🛡️ AI Shield**

Gemma answers your question once. Then it answers it five more times, and judges
its own answers against each other. If it knows the fact, all five samples mean
the same thing — one meaning cluster, entropy 0. If it's confabulating, they
scatter into five different stories — entropy 1. The invented clauses in the
displayed answer light up red.

**This is only possible with open weights.** Semantic entropy needs N independent
samples of the same prompt plus N more self-judgements — six-plus inferences per
question. On a metered API that's a cost multiplier nobody pays. On a local Gemma
it's free, offline, and private. It also needs per-call seeds, so each sample is
independent *and* reproducible — closed endpoints don't reliably give you that.

## Run

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary :all:

mkdir models                 # then fetch the weights (~769MB, one time)
curl -L -C - -o models/gemma-3-1b-it-Q4_K_M.gguf \
  https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf

python glassbox.py test        # self-check of the maths, needs no model
python glassbox.py judgecheck  # measures the judge, prints JUDGE_THRESHOLD
python eval.py                 # labelled run -> eval_results.json (resumable)
python glassbox.py serve       # http://127.0.0.1:8000
```

Two models by default: **1b answers, 4b judges.** That split is deliberate —
judging is a single forward pass with nothing generated, so the bigger model
costs little there, while answering is where the tokens (and the time) go. Set
`JUDGE_PATH = MODEL_PATH` in `glassbox.py` to run everything on the 1b.

```bash
curl -L -C - -o models/gemma-3-4b-it-Q4_K_M.gguf \
  https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf
```

One dependency (`llama-cpp-python`, a 6.6MB prebuilt CPU wheel — no compiler)
plus the weights. Everything else is stdlib, and it runs fully offline: no
server process, no API key, no network once the GGUF is on disk.

`-C -` on that curl means an interrupted download resumes instead of restarting.

Runs CPU-only on an i5. Generation is serialised behind a lock because a
llama.cpp context is not thread-safe — no loss on CPU, where a single
generation already uses every core.

## How the score is built

1. **Primary answer** at temperature 0 — this is what the user sees.
2. **Five samples** at temperature 0.9, each with its own seed.
3. **Meaning clusters** — Gemma as an equivalence judge, but we never read its
   *text*. We run one forward pass and compare the raw logits of the `YES` and
   `NO` tokens: `P(same) = softmax(logit_YES, logit_NO)`.

   This was the single biggest correctness fix in the project. Asked out loud,
   a 1b Gemma has such a heavy YES prior that it called *"Leonardo painted it"*
   and *"Michelangelo painted it"* the same answer — every question collapsed to
   one cluster and every entropy read zero. The distribution underneath knew
   better than the sampled token. Reading the logits took judge accuracy from
   10/12 to 11/12 and turned a coin flip into a calibratable 0–1 score.

   It is also *cheaper* than asking: one prefill, zero tokens generated. And it
   is flatly impossible against a hosted endpoint — you need the weights.

   Each sample is compared only against existing cluster representatives, so
   this costs O(n·k) judgements, not O(n²).
4. **Semantic entropy** over cluster sizes, normalised to 0–1. Entropy over
   *meanings*, not tokens: "1876" and "in the year 1876" are the same answer.
5. **Span highlighting** reuses the same judge — for each clause, what fraction
   of the samples support it? Unsupported clauses get the red wash. No second
   technique, no extra model.

## Numbers

24 labelled questions — 12 the model reliably knows, 12 obscure or
false-premise. `python eval.py` sweeps the threshold and picks the middle of the
winning range.

| | |
|---|---|
| Confabulations caught | **92%** (11/12) |
| Precision | **100%** (0 false positives) |
| F1 | 0.957 |
| Mean entropy, answerable | 0.026 |
| Mean entropy, obscure | 0.674 |
| Judge accuracy | 12/12 labelled pairs |

Every threshold in `glassbox.py` comes from those sweeps, not from taste, and
each is centred in its winning range rather than parked on the edge — entropy is
a 5-sample estimate and a cut-off sitting a hundredth away from real data flips
on noise.

## What it does not catch

**Consistent false belief.** Asked *"Which Apollo mission first landed on the far
side of the Moon?"*, Gemma answers *"Apollo 17"* five times out of five. No
mission ever landed there, but the samples agree, so entropy is 0.00 and we call
it grounded. That is the one miss in the 11/12.

The distinction is real and worth stating plainly: this measures whether a model
is **inventing on the spot**, not whether it is **right**. Confabulation is
unstable and shows up as spread; a memorised falsehood is stable and does not.
Catching the second kind needs retrieval against a source, not sampling.

A related honest note: the eval flagged *"boiling point of water in Celsius"* at
0.311 where we had labelled it answerable — Gemma replied *"212 degrees
Celsius"*, which is the Fahrenheit figure. The detector was right and the label
was wrong. We left the label alone; relabelling after seeing the score is how you
get a number that does not survive contact with a judge.

## Deliberately not built

Single-user, no database (a JSON cache file), no auth, no streaming, no model
switcher. One day, four hours. The cache is keyed on `(model, seed, temperature,
prompt)` so the demo questions are warm and the eval is resumable after Ctrl-C.
