"""GlassBox - finding the reasoning step where Gemma 4 starts guessing.

Ask the same question seven times. If Gemma knows the answer, every trace means
the same thing (1 cluster -> entropy 0). If it is confabulating, they scatter
(N clusters -> entropy 1). Gemma itself is the judge.

Two layers:
  * `inspect()` compares the seven *reasoning traces* step by step and locates
    the diverging step. This is the main event.
  * `score()` is the older answer-level detector, kept because eval.py measures
    it (92% recall / 100% precision, n=24) and because `inspect` folds the final
    answer in as a trailing assertion.

Open weights are load-bearing three times over: the judge reads YES/NO logits
rather than sampled text, sampling is seeded so traces are reproducible, and
repair resumes generation from a hand-edited reasoning prefix. None of the three
is reachable through a hosted API. Measured cost is ~7x the generated tokens of
a single answer - see probe.py for the single-forward-pass alternative.

    python glassbox.py test          # self-check, needs no model at all
    python glassbox.py judgecheck    # measure the judge on labelled pairs
    python glassbox.py inspect "..." # step-level reasoning analysis, as JSON
    python glassbox.py ask "..."     # answer-level detection, as JSON
    python glassbox.py serve         # http://127.0.0.1:8000
"""

import json
import math
import pathlib
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).parent
MODEL_PATH = HERE / "models" / "gemma-4-E2B-it-Q4_K_M.gguf"
# The judge only ever runs a single forward pass - no tokens are generated - so a
# bigger model costs far less here than it would for answering. Point this at a
# larger GGUF to trade latency for judge accuracy.
JUDGE_PATH = HERE / "models" / "gemma-4-E2B-it-Q4_K_M.gguf"

N_SAMPLES = 5
MAX_TOKENS = 64  # short answers judge better AND highlight better
THINK_TOKENS = 700  # a Gemma 4 reasoning trace needs room; 700 covers our questions
TEMP = 0.9
N_CTX = 4096  # thinking traces are long - 2048 truncates them
WORKERS = 4  # callers stay threaded, but the model lock serialises - see _llm()

# P(YES) above this counts as "same meaning". judgecheck tunes it - do not guess.
JUDGE_THRESHOLD = 0.91
# Verdict cut-offs from eval.py's sweep over 24 labelled questions - do not guess.
# CONFABULATED sits mid-way through the winning range [0.32, 0.59]; grounded
# questions topped out at 0.311 and caught confabulations started at 0.59.
SHAKY, CONFABULATED = 0.15, 0.45

THINK = "<|think|>"  # Gemma 4 special token: emit a reasoning trace before answering
_formatters = {}


def _prompt_for(llm, messages):
    """Render messages with the model's OWN chat template.

    Worth the extra machinery: Gemma 3's `<start_of_turn>` markers are not
    special tokens in Gemma 4 (they tokenise as seven ordinary tokens), so a
    hand-rolled wrapper silently mis-frames every prompt and quietly
    de-calibrates the judge. Asking the GGUF for its template cannot drift.
    """
    if id(llm) not in _formatters:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        _formatters[id(llm)] = Jinja2ChatFormatter(
            template=llm.metadata["tokenizer.chat_template"],
            bos_token=llm.detokenize([llm.token_bos()]).decode(),
            eos_token=llm.detokenize([llm.token_eos()]).decode(),
            add_generation_prompt=True,
        )
    return _formatters[id(llm)](messages=messages).prompt

CACHE_PATH = HERE / "cache.json"
_cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
_cache_lock = threading.Lock()
_model_lock = threading.Lock()
_models = {}


def _llm(path=None):
    """Load and memoise a model by path. Imported lazily so `glassbox.py test`
    runs with no model file present at all.

    A llama.cpp context is not thread-safe, so every call holds _model_lock for
    its duration. That costs nothing on CPU: one forward pass already spreads
    across every core, so serialising beats several calls fighting for them.
    """
    path = path or MODEL_PATH
    if path not in _models:
        if not path.exists():
            raise RuntimeError(f"model not found at {path} - see README for the download line")
        from llama_cpp import Llama

        _models[path] = Llama(model_path=str(path), n_ctx=N_CTX, verbose=False, logits_all=False)
    return _models[path]


def _save():
    """Persist the cache, merging with what is on disk first.

    _cache is a snapshot taken at import, so a plain overwrite silently drops
    every entry that appeared since - another process's, or a git pull's. A
    long-running `serve` re-dropped 274 pulled entries on each write until this
    existed. Re-read, union, write. Caller must already hold _cache_lock.

    ponytail: re-reads the whole file per write (~1.5MB, tens of ms). Fine while
    writes follow a multi-second generation; switch to append-only JSONL if the
    cache ever gets written in a tight loop.
    """
    if CACHE_PATH.exists():
        disk = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        disk.update(_cache)  # ours wins on a tie
        _cache.update(disk)  # and this process sees the union from here on
    CACHE_PATH.write_text(json.dumps(_cache), encoding="utf-8")


def _gen(prompt, seed, temperature, max_tokens=MAX_TOKENS):
    """One completion. Cached on disk by (seed, temp, max_tokens, prompt).

    The seed is part of the key on purpose: without it the N samples of one
    question would all collapse onto a single cache entry and entropy would
    read 0 forever.
    """
    key = f"{MODEL_PATH.name}|{seed}|{temperature}|{max_tokens}|{prompt}"
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    with _model_lock:
        llm = _llm()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        _last_toks.pop(id(llm), None)  # generation rewrote the KV cache
    text = out["choices"][0]["message"]["content"].strip()

    with _cache_lock:
        _cache[key] = text
        _save()
    return text


ANSWER_TMPL = """Answer the question in one or two short sentences. Be direct and specific. Do not hedge, do not add caveats.

Question: {q}
Answer:"""

JUDGE_TMPL = """You judge whether two answers to the same question mean the same thing. Reply with one word: YES or NO.

Question: What is the capital of Australia?
Answer A: Canberra.
Answer B: The capital is Canberra, in the ACT.
Same meaning: YES

Question: When was the telephone patented?
Answer A: 1876.
Answer B: It was patented in 1861.
Same meaning: NO

Question: {q}
Answer A: {a}
Answer B: {b}
Same meaning:"""

SUPPORT_TMPL = """Does the reference text support the claim? Reply with one word: YES or NO.

Reference: Marie Curie won Nobel Prizes in Physics and Chemistry.
Claim: Curie won two Nobel Prizes.
Supported: YES

Reference: Marie Curie won Nobel Prizes in Physics and Chemistry.
Claim: She was born in France.
Supported: NO

Reference: {ref}
Claim: {claim}
Supported:"""


_yes_no_ids = {}
_last_toks = {}  # per-model: what is currently in the KV cache, for prefix reuse


def p_yes(prompt):
    """P(YES) vs P(NO) for the next token, read straight off the logits.

    Sampling the judge's *text* was the single biggest source of error here: a
    1b model has a heavy YES prior, so the argmax token said "YES" to plainly
    contradictory pairs while the underlying distribution knew better. Reading
    the two logits recovers a graded, calibratable signal. It also costs less
    than generating - one forward pass, zero tokens sampled.

    Only possible because the weights are ours. No hosted endpoint exposes this.
    """
    import llama_cpp
    import numpy as np

    key = f"p_yes|{JUDGE_PATH.name}|{prompt}"
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    llm = _llm(JUDGE_PATH)
    if JUDGE_PATH not in _yes_no_ids:
        _yes_no_ids[JUDGE_PATH] = (
            llm.tokenize(b"YES", add_bos=False, special=False)[0],
            llm.tokenize(b"NO", add_bos=False, special=False)[0],
        )
    yes_id, no_id = _yes_no_ids[JUDGE_PATH]

    with _model_lock:
        rendered = _prompt_for(llm, [{"role": "user", "content": prompt}])
        toks = llm.tokenize(rendered.encode(), add_bos=True, special=True)
        # Every judge prompt opens with the same few-shot preamble, so only the
        # differing tail needs a forward pass. Rewinding n_tokens to the shared
        # prefix keeps that much of the KV cache instead of recomputing it.
        prev = _last_toks.get(id(llm), [])
        keep = 0
        while keep < len(prev) and keep < len(toks) - 1 and prev[keep] == toks[keep]:
            keep += 1
        # Clamp to what the context actually still holds: when JUDGE_PATH and
        # MODEL_PATH are the same file, a generation in between will have moved
        # n_tokens out from under us, and trusting a stale prefix reads the wrong
        # KV entries and silently returns a wrong probability.
        keep = min(keep, llm.n_tokens)
        llm.n_tokens = keep
        llm.eval(toks[keep:])
        _last_toks[id(llm)] = toks
        # llm._scores is a zero buffer unless logits_all=True; the live logits for
        # the last position come from the C API.
        logits = np.ctypeslib.as_array(
            llama_cpp.llama_get_logits(llm._ctx.ctx), shape=(llm._model.n_vocab(),)
        )
        y, n = float(logits[yes_id]), float(logits[no_id])

    top = max(y, n)  # softmax over just the two, shifted for numerical safety
    p = math.exp(y - top) / (math.exp(y - top) + math.exp(n - top))

    with _cache_lock:
        _cache[key] = p
        _save()
    return p


def answer(question, seed, temperature):
    return _gen(ANSWER_TMPL.format(q=question), seed, temperature)


def equivalent(question, a, b):
    return p_yes(JUDGE_TMPL.format(q=question, a=a, b=b)) >= JUDGE_THRESHOLD


def cluster(question, answers):
    """Greedy meaning-clusters. Each answer is compared only against existing
    cluster representatives, so this is O(n*k) judge calls, not O(n^2).
    Returns a list of index-lists.
    """
    clusters = []
    for i, a in enumerate(answers):
        for c in clusters:
            if equivalent(question, answers[c[0]], a):
                c.append(i)
                break
        else:
            clusters.append([i])
    return clusters


def entropy(clusters, n):
    """Shannon entropy over meaning-cluster sizes, normalised to 0..1."""
    if n < 2:
        return 0.0
    h = -sum((len(c) / n) * math.log(len(c) / n) for c in clusters)
    # abs, not max(...,0): every term of h is non-negative, so the only thing
    # being corrected is the -0.0 a single cluster produces. max() would not fix
    # it - -0.0 == 0.0, so max returns its first argument unchanged.
    return abs(h / math.log(n))


def split_clauses(text):
    """-> [(start, end)] offsets of each clause. Sentences, then long
    sentences broken on commas. Offsets so the UI can highlight without
    reflowing the original text.
    """
    parts = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not s:
            continue
        if len(s.split()) > 12 and "," in s:
            parts += [p for p in re.split(r",\s*", s) if p.strip()]
        else:
            parts.append(s)

    spans, cur = [], 0
    for p in parts:
        i = text.find(p, cur)
        if i < 0:
            continue
        spans.append((i, i + len(p)))
        cur = i + len(p)
    return spans


def verdict(h):
    if h < SHAKY:
        return "grounded"
    return "shaky" if h < CONFABULATED else "likely confabulated"


def score(question, n=N_SAMPLES, with_spans=True):
    """Full pipeline: primary answer, N samples, meaning-clusters, entropy,
    and per-clause support. One dict, ready to serialise.
    """
    primary = answer(question, seed=0, temperature=0.0)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        samples = list(ex.map(lambda i: answer(question, i + 1, TEMP), range(n)))

    clusters = cluster(question, samples)
    h = entropy(clusters, n)

    result = {
        "question": question,
        "answer": primary,
        "entropy": round(h, 3),
        "verdict": verdict(h),
        "n_clusters": len(clusters),
        "n_samples": n,
        "clusters": [[samples[i] for i in c] for c in clusters],
        "spans": [],
    }
    if not with_spans:
        return result

    # Reuse the same judge for span-level support: no second technique needed.
    # Averaging the probability rather than counting binary votes gives the UI a
    # smooth 0-1 ramp instead of six possible values.
    spans = split_clauses(primary)
    jobs = [(si, s) for si in range(len(spans)) for s in samples]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        votes = list(
            ex.map(
                lambda j: (j[0], p_yes(SUPPORT_TMPL.format(ref=j[1], claim=primary[slice(*spans[j[0]])]))),
                jobs,
            )
        )

    for si, (start, end) in enumerate(spans):
        hits = [v for i, v in votes if i == si]
        result["spans"].append(
            {
                "start": start,
                "end": end,
                "text": primary[start:end],
                "support": round(sum(hits) / len(hits), 2) if hits else 1.0,
            }
        )
    return result


# ---------------------------------------------------------------- web

INDEX = HERE / "index.html"


def serve(port=8000):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            for route, name in (("/api/eval", "eval_results.json"), ("/api/traces", "trace_results.json")):
                if self.path.startswith(route):
                    p = HERE / name
                    return self._json(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            q = body.get("question", "").strip()
            if not q:
                return self._json({"error": "no question"}, 400)
            try:
                if body.get("mode") == "answer":
                    return self._json(score(q))
                r = inspect(q)
                if body.get("repair", True) and "error" not in r:
                    r["repair"] = repair(q, r)
                self._json(r)
            except Exception as e:  # surface backend errors in the UI, not the console
                self._json({"error": str(e)}, 500)

        def log_message(self, *a):
            pass

    print(f"GlassBox on http://127.0.0.1:{port}  (model={MODEL_PATH.name})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


# ---------------------------------------------------------------- checks


# ---------------------------------------------------------------- reasoning traces
#
# Gemma 4 emits a reasoning trace before answering. Answer-level entropy tells you
# THAT a question is shaky; step-level entropy tells you WHERE the reasoning went
# off. Every sample plans in numbered steps, so the trace parses rather than
# needing a heuristic.

TRACE_SPLIT = re.compile(r"<channel\|>")
STEP_SPLIT = re.compile(r"^\s*\d+\.\s+", re.M)
# Swept against 14 labelled questions by `python traces.py --report`: every value
# in [0.36, 0.57] scores 86% / 83% / 83%, so accuracy alone does not pick one.
#
# 0.40 is chosen inside that plateau for a second reason: a step carrying a known
# factual error measured 0.406 on one machine, and anything above that misses it
# while gaining nothing. The plateau's centre (0.46) and the old 0.45 both sit on
# the wrong side of that. An earlier provisional 0.30 was a real regression - 79%,
# turning a true negative into a false positive - and rescued no recall at all.
#
# llama.cpp generation is not bit-identical across CPUs, so a borderline step can
# shift sides on another machine. Re-run the sweep where you intend to demo.
FORK = 0.40
# Entropy over a handful of aligned traces is a coarse estimate - with 4 traces
# it can only take a few values, so one sample landing differently swings it
# past a threshold. More traces is the honest fix; the Arc box has the headroom.
TRACE_SAMPLES = 6
ALIGN_MIN = 0.35  # below this, a trace simply never made this move
MAX_STEPS = 6  # forks land early; later steps are answer-formatting boilerplate

STEP_TMPL = """Two reasoning steps come from two attempts at the same question. Do they make the same move - the same claim, the same decision, the same next action?

Question: {q}
Step A: {a}
Step B: {b}

Reply with exactly one word, YES or NO."""

# Alignment and scoring are different questions, and using one prompt for both
# breaks alignment: "is element 111" vs "is element 118" scores NO as a
# claim-match - correctly - which makes the RIGHT counterpart look no better
# than an unrelated step, so the search returns noise. Alignment must ask about
# role, deliberately ignoring whether the two steps agree.
ALIGN_TMPL = """Two reasoning steps come from two attempts at the same question. Do they play the same role - addressing the same sub-question or performing the same part of the reasoning?

Answer YES even if they reach DIFFERENT conclusions. Only the role matters here, not whether they agree.

Question: {q}
Step A: {a}
Step B: {b}

Reply with exactly one word, YES or NO."""

# Most steps are procedure ("I will check my knowledge base"), and the model
# rewords those freely - they scored entropy 1.0 while the step carrying the
# actual factual error scored 0.69. Ranking on raw entropy therefore points at
# the wrong step. Only steps that assert something can be wrong, so we score
# those and mark the rest as procedure.
CLAIM_TMPL = """Could this reasoning step be factually WRONG? Answer YES only if it commits to a substantive fact about the world - a name, a number, a date, an identity, a value.

Answer NO for anything that merely:
- restates or rephrases the question
- names the topic without asserting anything about it
- describes a plan, a procedure, or how to format the response
- says the model will look something up, or checks whether it knows

Step: "Ununoctium is element 118."
Could this be wrong? YES

Step: "Identify the Subject: The Mona Lisa."
Could this be wrong? NO

Step: "The telephone was patented in 1876."
Could this be wrong? YES

Step: "Final Answer Construction."
Could this be wrong? NO

Step: "{s}"
Could this be wrong?"""
# 0.5 let through steps that merely mention an entity. A claim step should be
# unambiguous, so require the judge to be confident rather than merely leaning.
CLAIM_THRESHOLD = 0.90


def think(question, seed, temperature, max_tokens=THINK_TOKENS):
    """One reasoning pass. Returns the raw text, trace and answer together."""
    key = f"think|{MODEL_PATH.name}|{seed}|{temperature}|{max_tokens}|{question}"
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    with _model_lock:
        llm = _llm()
        out = llm.create_chat_completion(
            messages=[{"role": "system", "content": THINK}, {"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        _last_toks.pop(id(llm), None)  # generation rewrote the KV cache
    text = out["choices"][0]["message"]["content"]

    with _cache_lock:
        _cache[key] = text
        _save()
    return text


def parse_trace(text):
    """-> (steps, answer). Steps keep their prose; the answer is what the user sees."""
    parts = TRACE_SPLIT.split(text, maxsplit=1)
    trace, answer = (parts[0], parts[1]) if len(parts) == 2 else (text, "")
    trace = trace.split("\n", 1)[-1] if trace.startswith("<|channel>") else trace
    steps = [" ".join(s.split()) for s in STEP_SPLIT.split(trace)[1:]]
    return [s for s in steps if s], " ".join(answer.split())


def is_claim(step):
    return p_yes(CLAIM_TMPL.format(s=step)) >= CLAIM_THRESHOLD


def counterpart(question, step, trace):
    """The step in `trace` that makes the same move as `step`, or the closest one.

    Alignment is by content, not position. Traces do not share a step ordering -
    one plans "Identify the Subject" where another writes "Recall knowledge
    about ...", so comparing index 2 against index 2 pits different kinds of step
    against each other and manufactures a disagreement out of nothing.
    """
    return max(((p_yes(ALIGN_TMPL.format(q=question, a=step, b=c)), c) for c in trace), default=(0.0, ""))


def step_entropy(question, primary, others, max_steps=MAX_STEPS):
    """For each assertion in the primary trace, find its counterpart in every
    other trace, then take semantic entropy over that aligned group.
    """
    out = []
    for i, step in enumerate(primary[:max_steps]):
        claim = is_claim(step)
        if not claim:  # procedure steps are reworded freely; scoring them is noise
            out.append({"index": i + 1, "text": step, "claim": False, "entropy": None,
                        "n_clusters": None, "n_aligned": 0, "absent": 0, "variants": []})
            continue

        # A trace that never addressed this point has no counterpart to compare.
        # Forcing max() to return its closest step anyway drags an unrelated
        # procedure step into the group and reports it as a competing reading.
        matches = [counterpart(question, step, t) for t in others if t]
        aligned = [step] + [c for p, c in matches if p >= ALIGN_MIN]
        absent = sum(1 for p, _ in matches if p < ALIGN_MIN)
        clusters = []
        for j, v in enumerate(aligned):
            for c in clusters:
                if p_yes(STEP_TMPL.format(q=question, a=aligned[c[0]], b=v)) >= JUDGE_THRESHOLD:
                    c.append(j)
                    break
            else:
                clusters.append([j])
        out.append(
            {
                "index": i + 1,
                "text": step,
                "claim": True,
                "entropy": round(entropy(clusters, len(aligned)), 3),
                "n_clusters": len(clusters),
                "n_aligned": len(aligned),
                "absent": absent,  # traces that never made this move at all
                "variants": [aligned[c[0]] for c in clusters[1:]],  # the divergent readings
            }
        )
    return out


def answer_spread(question, primary, answers):
    """Semantic entropy over the final answers themselves.

    The backstop for a real gap: some questions produce a trace that is pure
    procedure - the model plans, hedges, and only commits in the answer. Those
    score zero assertions, so step analysis has nothing to look at and the error
    walks straight through. Clustering the answers catches exactly that case.
    """
    pool = [a for a in [primary] + answers if a]
    if len(pool) < 2:
        return None
    clusters = []
    for j, a in enumerate(pool):
        for c in clusters:
            if p_yes(JUDGE_TMPL.format(q=question, a=pool[c[0]], b=a)) >= JUDGE_THRESHOLD:
                c.append(j)
                break
        else:
            clusters.append([j])
    return {
        "entropy": round(entropy(clusters, len(pool)), 3),
        "n_clusters": len(clusters),
        "n_aligned": len(pool),
        "variants": [pool[c[0]] for c in clusters[1:]],
    }


def inspect(question, n=TRACE_SAMPLES):
    """Sample N reasoning traces, score every assertion, and locate the fork."""
    primary_steps, primary_answer = parse_trace(think(question, seed=0, temperature=0.0))
    parsed = [parse_trace(think(question, i + 1, TEMP)) for i in range(n)]
    others = [t for t, _ in parsed if t]
    other_answers = [a for _, a in parsed]
    if not primary_steps:
        primary_steps = others.pop(0) if others else []
    if not primary_steps:
        return {"question": question, "error": "no parseable reasoning trace"}

    steps = step_entropy(question, primary_steps, others)
    spread = answer_spread(question, primary_answer, other_answers)
    if spread:
        # The answer is the last assertion, and on procedure-only traces it is the
        # ONLY one. Appending it as a final step means those questions still get
        # scored instead of silently passing.
        steps.append({
            "index": len(steps) + 1, "text": primary_answer or "(no answer parsed)",
            "claim": True, "is_answer": True, "absent": 0, **spread,
        })
    claims = [s for s in steps if s["claim"]]
    fork = next((s["index"] for s in claims if s["entropy"] >= FORK), None)
    traces = [primary_steps] + others
    return {
        "question": question,
        "answer": primary_answer,
        "steps": steps,
        "fork": fork,
        "answer_entropy": spread["entropy"] if spread else None,
        "n_traces": len(traces),
        "depth": len(steps),
        "n_claims": len(claims),
        # averaged over claim steps only - procedure steps carry no signal
        "mean_entropy": round(sum(s["entropy"] for s in claims) / len(claims), 3) if claims else 0.0,
    }


# ---------------------------------------------------------------- repair
#
# Detection is only half of a guardrail. Once the fork is located we hand the
# model its OWN competing readings and ask it to adjudicate, then re-run the
# reasoning from that point with the chosen step substituted in.
#
# The bet is that recognition is easier than recall: free-recalling an element
# number is a lookup the model half-remembers, whereas picking the right option
# from a short list is a judgement it can make. Nothing external is consulted -
# the correction comes entirely from the spread the model already produced.

# The abstain option is not politeness, it is the whole safety property. Forced to
# choose, the model chooses - even when every candidate is wrong. Measured: offered
# "element 111" and "element 112" for ununoctium (truly 118), it picked 112 and
# confidently re-derived an answer around it. Swapping one wrong claim for another
# is strictly worse than not repairing, because the output now looks corrected.
ADJUDICATE_TMPL = """While reasoning about a question, you produced these competing versions of one step. They cannot all be right.

Question: {q}

{options}
{none_option}. None of these is correct.

Which option is correct? If you are not confident that one of them is right, choose {none_option}. Reply with ONLY the number."""

TRACE_HEAD = "<|channel>thought\nThinking Process:\n\n"


def adjudicate(question, readings):
    """Pick among the model's own competing readings.

    -> index, or None for "none of these" / unparseable. None means no repair is
    attempted, which is the correct outcome when the truth was never in the
    candidate set.
    """
    options = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(readings))
    prompt = ADJUDICATE_TMPL.format(q=question, options=options, none_option=len(readings) + 1)
    out = _gen(prompt, seed=0, temperature=0.0, max_tokens=8)
    digits = re.search(r"\d+", out)
    if not digits:
        return None
    pick = int(digits.group()) - 1
    return pick if 0 <= pick < len(readings) else None  # the abstain index falls through


def repair(question, insp):
    """Re-run the reasoning from the fork with the adjudicated step substituted.

    Returns None when there is nothing to repair, so callers can treat "no fork"
    and "fork resolved to the original reading" as the same non-event.
    """
    # Target the most divergent claim step, not strictly the flagged fork. FORK is
    # still provisional, and gating repair on it means a mis-set threshold
    # silently disables repair entirely. Wherever the traces disagree there is
    # something to adjudicate, flagged or not.
    # The appended answer row is not a reasoning step - there is nothing after it
    # to re-run, so repair stays on the trace itself.
    #
    # Drop procedural readings from the group rather than rejecting the whole step.
    #
    # Scoring only the primary text let groups through whose *variants* were
    # procedure ("Analyze the Request", "Final Answer Construction"). Asking which
    # of those is "correct" is a malformed question, so the model declines - and
    # that decline says nothing about whether it can recognise a right answer. That
    # confound invalidated our first repair measurement.
    #
    # Requiring every reading to be a claim over-corrects: one procedural stray in
    # an otherwise real group ("element 111" vs "element 112" vs "Search for the
    # value...") would discard a genuine adjudication. Filter the readings instead,
    # and only keep steps where two or more real claims survive to disagree.
    candidates = []
    for s in insp["steps"]:
        if not s["claim"] or not s["variants"] or s.get("is_answer"):
            continue
        keep = [t for t in [s["text"]] + s["variants"] if is_claim(t)]
        if len(keep) >= 2:
            candidates.append((s, keep))
    if not candidates:
        return None
    step, readings = max(candidates, key=lambda c: c[0]["entropy"])

    pick = adjudicate(question, readings)
    if pick is None:
        # It looked at its own competing readings and rejected all of them. That is
        # a result worth surfacing, not a silent no-op: the model can tell it does
        # not know even where it cannot recall the right answer, and the guardrail
        # fails closed instead of inventing a correction.
        return {"step": step["index"], "entropy": step["entropy"], "declined": True,
                "readings": readings, "answer_before": insp["answer"]}
    chosen = readings[pick]

    # Rebuild the trace: everything before the fork verbatim, then the chosen
    # reading, then let the model carry on from there.
    prefix = [s["text"] for s in insp["steps"] if s["index"] < step["index"]] + [chosen]
    body = TRACE_HEAD + "".join(f"{i + 1}.  {s}\n" for i, s in enumerate(prefix))

    # Consult the cache BEFORE touching the model. _llm() used to be called first,
    # purely to render the prompt, which meant every repair loaded 2.9GB even on a
    # pure cache hit - enough to break replay on a weightless clone and with it the
    # notebook's "no model download" claim.
    key = f"repair|{MODEL_PATH.name}|{body}"
    with _cache_lock:
        cached = _cache.get(key)

    if cached is None:
        with _model_lock:
            llm = _llm()
            seed_prompt = _prompt_for(
                llm, [{"role": "system", "content": THINK}, {"role": "user", "content": question}]
            )
            out = llm.create_completion(
                seed_prompt + body, temperature=0.0, max_tokens=THINK_TOKENS, seed=0
            )
            cached = out["choices"][0]["text"]
            _last_toks.pop(id(llm), None)
        with _cache_lock:
            _cache[key] = cached
            _save()

    _, answer = parse_trace(body + cached)
    return {
        "step": step["index"],
        "entropy": step["entropy"],
        "flagged": insp.get("fork") == step["index"],
        "readings": readings,
        "chosen": chosen,
        "chosen_index": pick,
        "was_original": pick == 0,
        "answer_before": insp["answer"],
        "answer_after": answer or cached.strip()[:600],
    }


# (question, answer A, answer B, do they state the same fact?)
# The hard cases are same-topic/different-fact - that is exactly what a
# confabulating model produces, so the judge has to get those right.
JUDGE_PAIRS = [
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted it.", "It was painted by Leonardo da Vinci.", True),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted it.", "Michelangelo painted it.", False),
    ("What is the longest river in Africa?", "The Nile.", "The Nile is the longest river in Africa.", True),
    ("What is the longest river in Africa?", "The Nile.", "The Congo River.", False),
    ("Why did Tesla decline his 1917 Nobel Prize?", "Tesla refused due to plagiarism accusations.",
     "Tesla refused over a dispute about wireless power transmission.", False),
    ("Why did Tesla decline his 1917 Nobel Prize?", "He declined because of a feud with Edison.",
     "He turned it down due to his rivalry with Thomas Edison.", True),
    ("Why did Tesla decline his 1917 Nobel Prize?",
     "He refused over disagreements with the Nobel committee about his work's scope.",
     "He refused because of accusations of plagiarism.", False),
    ("When was the telephone patented?", "1876.", "It was patented in 1876 by Bell.", True),
    ("When was the telephone patented?", "1876.", "1861.", False),
    ("What is the capital of Australia?", "Canberra.", "Sydney.", False),
    ("What is the capital of Australia?", "Canberra.", "The capital is Canberra, in the ACT.", True),
    ("Who won the 1931 Ceylon chess championship?", "It was won by A. Perera.", "The winner was S. Fernando.", False),
]


def judgecheck():
    """Measure the judge and pick JUDGE_THRESHOLD from evidence, not taste.

    The judge is the crux: if it cannot tell two different answers apart, every
    question clusters into one and the entropy is uniformly zero.
    """
    scored = [(p_yes(JUDGE_TMPL.format(q=q, a=a, b=b)), same, a, b) for q, a, b, same in JUDGE_PAIRS]
    grid = [(i / 100, sum((p >= i / 100) == same for p, same, *_ in scored)) for i in range(1, 100)]
    hits = max(h for _, h in grid)
    # Take the middle of the winning range, not its edge. Probabilities near the
    # boundary drift by a couple of points between runs, and a threshold sitting
    # right against one of them would flip on noise.
    winners = [t for t, h in grid if h == hits]
    t = round((min(winners) + max(winners)) / 2, 2)
    for p, same, a, b in sorted(scored, key=lambda r: -r[0]):
        ok = " " if (p >= t) == same else "X"
        print(f"  {ok} want {'SAME' if same else 'DIFF'}  p_yes {p:6.3f}   {a[:34]!r} vs {b[:34]!r}")
    print(f"\njudge: {hits}/{len(scored)} at threshold {t:.2f}   (judge model: {JUDGE_PATH.name})")
    print(f"-> set JUDGE_THRESHOLD = {t:.2f} in glassbox.py")
    return t, hits


def demo():
    """Offline self-check: the maths and the parsing, no model needed."""
    global CACHE_PATH, _cache
    close = lambda a, b: math.isclose(a, b, abs_tol=1e-9)
    assert close(entropy([[0, 1, 2, 3, 4]], 5), 0.0), "unanimous samples must score 0"
    assert math.copysign(1, entropy([[0, 1, 2, 3, 4]], 5)) > 0, "must be +0.0, not -0.0"
    assert close(entropy([[0], [1], [2], [3], [4]], 5), 1.0), "total disagreement must score 1"
    assert 0.4 < entropy([[0, 1, 2], [3], [4]], 5) < 0.8
    assert close(entropy([[0]], 1), 0.0), "n=1 has no spread to measure"

    t = "Canberra is the capital. It was chosen in 1908 as a compromise, because Sydney and Melbourne both wanted the honour."
    spans = split_clauses(t)
    assert [t[a:b] for a, b in spans][0] == "Canberra is the capital."
    assert len(spans) >= 3, "the long second sentence should split on commas"
    for a, b in spans:
        assert t[a:b] in t and a < b, "offsets must index the original text"
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)), "spans must not overlap"

    assert verdict(0.0) == "grounded" and verdict(1.0) == "likely confabulated"

    # _save must merge with the file, not overwrite it. A long-running serve held a
    # pre-pull snapshot and silently re-dropped 274 pulled entries on every write.
    import tempfile

    real_path, real_cache = CACHE_PATH, _cache
    try:
        with tempfile.TemporaryDirectory() as d:
            CACHE_PATH = pathlib.Path(d) / "c.json"
            CACHE_PATH.write_text(json.dumps({"theirs": 1, "shared": "disk"}), encoding="utf-8")
            _cache = {"ours": 2, "shared": "mine"}
            _save()
            back = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            assert back == {"theirs": 1, "ours": 2, "shared": "mine"}, f"_save lost entries: {back}"
            assert _cache["theirs"] == 1, "_save must also refresh this process's view"
    finally:
        CACHE_PATH, _cache = real_path, real_cache

    print("ok - entropy, clause offsets, verdicts and cache merge all check out")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "test":
        demo()
    elif cmd == "judgecheck":
        judgecheck()
    elif cmd == "inspect":
        print(json.dumps(inspect(" ".join(sys.argv[2:])), indent=2))
    elif cmd == "serve":
        serve()
    elif cmd == "ask":
        print(json.dumps(score(" ".join(sys.argv[2:])), indent=2))
    else:
        sys.exit(__doc__)
