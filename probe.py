"""Semantic entropy from ONE forward pass, instead of seven reasoning traces.

Implements the idea from Semantic Entropy Probes (Kossen et al., 2024,
arXiv:2406.15927): the model's hidden state already encodes whether it is about
to confabulate, so a small linear probe trained on that state can approximate
semantic entropy without any sampling at all.

Why it matters here: seven traces is the honest cost of measuring disagreement,
but it is also the main reason nobody would run this in production. A probe turns
GlassBox into a cascade - screen every request in one pass, and spend the seven
traces only on the ones the probe flags.

Deviation from the paper, stated plainly: they probe intermediate layers of a
model they control end to end. Through llama.cpp we read the final-layer
embedding of the last token, which is the accessible analogue. We also have
tens of labelled examples where the paper has thousands, so everything below is
leave-one-out cross-validated and reported against a majority baseline. Treat it
as a feasibility result, not a trained artefact.

    python probe.py            # trains + LOO-CV, writes probe.json
"""
import json
import pathlib
import sys

import numpy as np

from glassbox import CONFABULATED, MODEL_PATH, N_CTX

HERE = pathlib.Path(__file__).parent
EMB_CACHE = HERE / "probe_embeddings.json"
OUT = HERE / "probe.json"

_llm = None


def embed(text):
    """Final-layer hidden state of the last token."""
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        _llm = Llama(model_path=str(MODEL_PATH), n_ctx=N_CTX, embedding=True, verbose=False)
    v = _llm.create_embedding(text)["data"][0]["embedding"]
    return np.asarray(v[-1] if isinstance(v[0], list) else v, dtype=np.float32)


def labelled():
    """(text, entropy) pairs from whichever result files exist.

    The probe sees exactly what a single-pass deployment would see: the question
    and the model's greedy answer. It never sees the samples.
    """
    rows, seen = [], set()
    for name, field in (("eval_results.json", "entropy"),
                        ("bulk_results.json", "entropy"),
                        ("trace_results.json", "mean_entropy")):
        p = HERE / name
        if not p.exists():
            continue
        for r in json.loads(p.read_text(encoding="utf-8")).get("rows", []):
            if r["question"] in seen:  # same question labelled twice is not two datapoints
                continue
            seen.add(r["question"])
            rows.append((f"Q: {r['question']}\nA: {r['answer']}", r[field]))
    return rows


def features(rows):
    cache = json.loads(EMB_CACHE.read_text(encoding="utf-8")) if EMB_CACHE.exists() else {}
    X, y, fresh = [], [], 0
    for text, ent in rows:
        if text not in cache:
            cache[text] = embed(text).tolist()
            fresh += 1
        X.append(cache[text])
        y.append(1 if ent >= CONFABULATED else 0)
    if fresh:
        EMB_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        print(f"embedded {fresh} new texts ({len(cache)} cached)")
    return np.asarray(X, dtype=np.float32), np.asarray(y)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    rows = labelled()
    if len(rows) < 12:
        sys.exit(f"only {len(rows)} labelled rows - run eval.py and traces.py first")
    X, y = features(rows)
    print(f"\n{len(y)} examples, {X.shape[1]} dims, {y.sum()} positive / {len(y)-y.sum()} negative")
    if y.sum() < 3 or (len(y) - y.sum()) < 3:
        sys.exit("need at least 3 of each class to say anything")

    # Strong L2: 1536 dims against a few dozen rows will memorise anything it is
    # allowed to. The regularisation is doing more work here than the model is.
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        preds[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

    baseline = max(y.mean(), 1 - y.mean())  # always guess the commoner class
    acc = ((preds >= 0.5).astype(int) == y).mean()
    auroc = roc_auc_score(y, preds) if 0 < y.sum() < len(y) else float("nan")
    print(f"\nleave-one-out accuracy : {acc:.0%}")
    print(f"majority baseline      : {baseline:.0%}")
    print(f"leave-one-out AUROC    : {auroc:.3f}   (0.5 = no signal)")

    beats = acc > baseline + 0.02 and auroc > 0.6
    print("\n" + ("SIGNAL: the hidden state predicts confabulation without sampling."
                  if beats else
                  "NO CLEAR SIGNAL at this sample size - report it as a negative result."))

    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X), y)
    OUT.write_text(json.dumps({
        "note": "single-pass semantic-entropy probe, after Kossen et al. arXiv:2406.15927",
        "n": len(y), "loo_accuracy": round(float(acc), 3),
        "loo_auroc": round(float(auroc), 3), "baseline": round(float(baseline), 3),
        "usable": bool(beats),
        "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
    }), encoding="utf-8")
    print(f"\nwrote {OUT.name}")
    print(f"\nSLIDE: one forward pass predicts confabulation at {acc:.0%} "
          f"(AUROC {auroc:.2f}) vs {baseline:.0%} baseline, n={len(y)}")


if __name__ == "__main__":
    main()
