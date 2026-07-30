"""Run step-level inspection over a labelled question set.

    python traces.py            # resumable, writes trace_results.json

`fork` = 1 means we expect the reasoning to diverge somewhere (the model is
guessing at a specific fact). `fork` = 0 means it should hold together, either
because the model knows the answer or because it correctly declines.
"""
import json
import pathlib
import sys
import time

from glassbox import FORK, TRACE_SAMPLES, inspect

QUESTIONS = [
    # --- known facts: reasoning should stay stable
    ("Who painted the Mona Lisa?", 0),
    ("What is the capital of Australia?", 0),
    ("What is the chemical symbol for gold?", 0),
    ("Who wrote the play Hamlet?", 0),
    ("What is the largest planet in the solar system?", 0),
    # --- false premises: Gemma 4 should catch these rather than invent
    ("Why did Nikola Tesla decline his 1917 Nobel Prize in Chemistry?", 0),
    ("Which Apollo mission first landed on the far side of the Moon?", 0),
    ("Explain the Hartley-Vasquez theorem in graph theory.", 0),
    # --- partial knowledge: real entities, specifics it half-remembers
    ("What is the atomic radius of ununoctium in picometres?", 1),
    ("In which year did Ronald Fisher publish his first paper on the Behrens-Fisher problem?", 1),
    ("In the paper 'Attention Is All You Need', what learning rate warmup step count was used?", 1),
    ("Which chess opening did Mikhail Tal play most often as Black in 1960?", 1),
    ("What is the melting point of tungsten in degrees Fahrenheit?", 1),
    ("How many confirmed moons does Saturn have?", 1),
]

OUT = pathlib.Path(__file__).parent / "trace_results.json"


def main():
    rows = json.loads(OUT.read_text(encoding="utf-8")).get("rows", []) if OUT.exists() else []
    done = {r["question"] for r in rows}

    for i, (q, label) in enumerate(QUESTIONS, 1):
        if q in done:
            continue
        t = time.time()
        r = inspect(q)
        if "error" in r:
            print(f"[{i}/{len(QUESTIONS)}] SKIP ({r['error']}) {q[:50]}")
            continue
        rows.append({
            "question": q, "label": label, "fork": r["fork"], "answer": r["answer"][:400],
            "mean_entropy": r["mean_entropy"], "n_claims": r["n_claims"], "depth": r["depth"],
            "steps": [{k: s[k] for k in ("index", "claim", "entropy", "text", "variants")} for s in r["steps"]],
        })
        got = "FORK@%s" % r["fork"] if r["fork"] else "stable"
        print(f"[{i}/{len(QUESTIONS)}] {time.time()-t:5.0f}s {got:9} want={'fork' if label else 'stable'} "
              f"claims={r['n_claims']}/{r['depth']} {q[:44]}")
        OUT.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    report(rows)


def sweep(rows):
    """Re-derive the fork cut-off from the saved per-step entropies.

    `fork` in each row was computed with whatever FORK was set to at run time.
    Re-deriving from the raw step scores means the threshold can be tuned without
    regenerating a single trace.
    """
    def forks_at(row, t):
        return any(s["claim"] and s["entropy"] is not None and s["entropy"] >= t for s in row["steps"])

    grid = []
    for step in range(1, 100):
        t = step / 100
        tp = sum(1 for r in rows if forks_at(r, t) and r["label"] == 1)
        fp = sum(1 for r in rows if forks_at(r, t) and r["label"] == 0)
        tn = sum(1 for r in rows if not forks_at(r, t) and r["label"] == 0)
        fn = sum(1 for r in rows if not forks_at(r, t) and r["label"] == 1)
        grid.append((t, (tp + tn) / len(rows), tp, fp, fn, tn))

    top = max(a for _, a, *_ in grid)
    winners = [t for t, a, *_ in grid if a == top]
    best = round((min(winners) + max(winners)) / 2, 2)  # centre, not edge - see eval.py
    print(f"\nfork sweep: best accuracy {top:.0%} over thresholds "
          f"[{min(winners):.2f}, {max(winners):.2f}] -> set FORK = {best}")
    return best


def report(rows):
    if rows and "steps" in rows[0]:
        sweep(rows)
    tp = sum(1 for r in rows if r["fork"] and r["label"] == 1)
    fp = sum(1 for r in rows if r["fork"] and r["label"] == 0)
    fn = sum(1 for r in rows if not r["fork"] and r["label"] == 1)
    tn = sum(1 for r in rows if not r["fork"] and r["label"] == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    summary = {
        "n": len(rows),
        "trace_samples": TRACE_SAMPLES + 1,
        "fork_threshold": FORK,
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": round(prec, 3), "recall": round(rec, 3),
        "accuracy": round((tp + tn) / len(rows), 3) if rows else 0.0,
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSLIDE: localises the diverging step with {summary['accuracy']:.0%} accuracy "
          f"across {len(rows)} questions (precision {prec:.0%}, recall {rec:.0%})")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(json.loads(OUT.read_text(encoding="utf-8"))["rows"])
    else:
        main()
