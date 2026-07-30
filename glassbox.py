"""GlassBox - semantic-entropy hallucination detection for Gemma.

Ask the same question N times. If Gemma knows the answer, every sample means
the same thing (1 cluster -> entropy 0). If it is confabulating, the samples
scatter (N clusters -> entropy 1). Gemma itself is the equivalence judge.

This needs open weights: N-sampling and self-judging are only cheap when the
model runs locally. A closed API bills you 6x and hides the spread.

    python glassbox.py test          # self-check, needs no model at all
    python glassbox.py serve         # http://127.0.0.1:8000
    python glassbox.py ask "who invented the safety pin?"
"""

import json
import math
import pathlib
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).parent
MODEL_PATH = HERE / "models" / "gemma-3-1b-it-Q4_K_M.gguf"
# The judge only ever runs a single forward pass - no tokens are generated - so a
# bigger model costs far less here than it would for answering. Point this at the
# same file as MODEL_PATH to run everything on one model.
JUDGE_PATH = HERE / "models" / "gemma-3-4b-it-Q4_K_M.gguf"

N_SAMPLES = 5
MAX_TOKENS = 64  # short answers judge better AND highlight better
TEMP = 0.9
N_CTX = 2048  # prompts here are tiny; small context keeps CPU prefill quick
WORKERS = 4  # callers stay threaded, but the model lock serialises - see _llm()

# P(YES) above this counts as "same meaning". judgecheck tunes it - do not guess.
JUDGE_THRESHOLD = 0.96
# Verdict cut-offs from eval.py's sweep over 24 labelled questions - do not guess.
# CONFABULATED sits mid-way through the winning range [0.32, 0.59]; grounded
# questions topped out at 0.311 and caught confabulations started at 0.59.
SHAKY, CONFABULATED = 0.15, 0.45

GEMMA_TURN = "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"

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
        CACHE_PATH.write_text(json.dumps(_cache), encoding="utf-8")
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
        toks = llm.tokenize(GEMMA_TURN.format(prompt).encode(), add_bos=True, special=True)
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
        CACHE_PATH.write_text(json.dumps(_cache), encoding="utf-8")
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
            if self.path.startswith("/api/eval"):
                p = HERE / "eval_results.json"
                return self._json(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            q = json.loads(self.rfile.read(n) or b"{}").get("question", "").strip()
            if not q:
                return self._json({"error": "no question"}, 400)
            try:
                self._json(score(q))
            except Exception as e:  # surface backend errors in the UI, not the console
                self._json({"error": str(e)}, 500)

        def log_message(self, *a):
            pass

    print(f"GlassBox on http://127.0.0.1:{port}  (model={MODEL_PATH.name})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


# ---------------------------------------------------------------- checks


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
    print("ok - entropy, clause offsets and verdicts all check out")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "test":
        demo()
    elif cmd == "judgecheck":
        judgecheck()
    elif cmd == "serve":
        serve()
    elif cmd == "ask":
        print(json.dumps(score(" ".join(sys.argv[2:])), indent=2))
    else:
        sys.exit(__doc__)
