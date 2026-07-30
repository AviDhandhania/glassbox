# Setup on a new machine

Everything below is copy-paste. Total download is ~3 GB of weights; the code has
one dependency.

## 1. Clone and install

```bash
git clone https://github.com/AviDhandhania/glassbox
cd glassbox

pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary :all:
```

`--only-binary :all:` matters — without it pip tries to compile llama.cpp from
source, which needs a full C++ toolchain and takes 20+ minutes. The prebuilt
wheel is 6.6 MB and installs in seconds.

## 2. Get the weights

```bash
mkdir models
curl -L -C - -o models/gemma-4-E2B-it-Q4_K_M.gguf \
  https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf
```

That single 2.9 GB file runs **both** roles — answering and judging. `-C -` means
a dropped download resumes on re-run instead of restarting.

Optional, only if the judge turns out to be the weak link (it currently scores
12/12, so probably don't bother):

```bash
curl -L -C - -o models/gemma-4-E4B-it-Q4_K_M.gguf \
  https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf
# then set JUDGE_PATH to it in glassbox.py
```

## 3. Verify, in this order

```bash
python glassbox.py test         # no model needed - pure maths, instant
python glassbox.py judgecheck   # 12 labelled pairs, expect 12/12
python show.py "What is the atomic radius of ununoctium in picometres?"
python glassbox.py serve        # http://127.0.0.1:8000
```

The committed `cache.json` means step 3 and the four UI presets return
**instantly** on a fresh clone — every generation and judgement is already in
there. Delete `cache.json` to force genuine cold runs.

## Intel Arc acceleration (optional)

The CPU wheel above works everywhere and is what every measurement in this repo
was taken with. Try Arc **only if CPU speed becomes the bottleneck**, and keep
the CPU wheel working first — a broken GPU build with hours on the clock is a bad
trade.

A reasoning trace costs ~40s on a mid-range CPU. Most of the demo is served from
cache, so the CPU path is genuinely viable.

If you do want Arc, Vulkan is the least painful route (SYCL needs the full oneAPI
toolkit):

```bash
# needs the Vulkan SDK and a C++ toolchain installed first
pip install llama-cpp-python --force-reinstall --no-binary llama-cpp-python \
  -C cmake.args="-DGGML_VULKAN=ON"
```

Then pass `n_gpu_layers=-1` in `_llm()` in `glassbox.py` to offload everything.
Confirm it worked by checking that load logs mention a Vulkan device — if they
don't, you are silently still on CPU and just spent the time for nothing.

**Rollback if it breaks:**

```bash
pip install --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary :all:
```

## Where things live

| File | What |
|---|---|
| `glassbox.py` | everything: model loading, logit judge, entropy, trace parsing, web server |
| `show.py` | terminal view of one reasoning inspection |
| `traces.py` | step-level test set → `trace_results.json` |
| `eval.py` | answer-level test set → `eval_results.json` |
| `index.html` | the UI |
| `WRITEUP.md` | Kaggle submission draft |

All tuning constants sit at the top of `glassbox.py`. Every one of them was set
by a sweep (`judgecheck`, `eval.py`, `traces.py`), not by hand — if you change a
model, re-run those before trusting any number.
