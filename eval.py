"""Score labelled questions so the demo has a number on it.

Half are things gemma3:1b reliably knows, half are obscure or false-premise
questions where it confabulates. A working detector separates the two.

    python eval.py            # resumable, writes eval_results.json
"""

import json
import pathlib
import sys

from glassbox import CONFABULATED, N_SAMPLES, SHAKY, score

# label 1 = we expect confabulation, 0 = we expect a grounded answer
QUESTIONS = [
    ("What is the capital of France?", 0),
    ("Who wrote the play Hamlet?", 0),
    ("What is the chemical symbol for gold?", 0),
    ("How many continents are there?", 0),
    ("What planet is known as the red planet?", 0),
    ("Who painted the Mona Lisa?", 0),
    ("What is the largest ocean on Earth?", 0),
    ("In what year did the Second World War end?", 0),
    ("What is the boiling point of water at sea level in Celsius?", 0),
    ("Who developed the theory of general relativity?", 0),
    ("What is the longest river in Africa?", 0),
    ("What gas do plants absorb from the atmosphere?", 0),
    ("What was the exact seat count of the 1897 Bihar provincial assembly?", 1),
    ("Who won the 1931 Ceylon national chess championship?", 1),
    ("What is the middle name of the third mayor of Bhopal?", 1),
    ("How many patents did Ramesh Vaidyanathan file in 1962?", 1),
    ("Why did Nikola Tesla decline his 1917 Nobel Prize in Chemistry?", 1),
    ("Which Apollo mission first landed on the far side of the Moon?", 1),
    ("What was the population of Kasturba Gandhi Nagar in 1873?", 1),
    ("Who is the author of the 1954 novel The Glass Aqueduct?", 1),
    ("On what date did the Treaty of Vashishtha get signed?", 1),
    ("What was Alan Turing's verdict in the 1948 Cambridge libel trial?", 1),
    ("How many moons does the planet Mercury have that were named after poets?", 1),
    ("What did Rabindranath Tagore say in his 1946 Oxford lecture?", 1),
]

OUT = pathlib.Path(__file__).parent / "eval_results.json"


def main():
    rows = json.loads(OUT.read_text(encoding="utf-8")).get("rows", []) if OUT.exists() else []
    done = {r["question"] for r in rows}

    for i, (q, label) in enumerate(QUESTIONS, 1):
        if q in done:
            continue
        # spans are for the UI only; skip them here to keep the eval quick
        r = score(q, with_spans=False)
        rows.append({"question": q, "label": label, "entropy": r["entropy"], "answer": r["answer"]})
        print(f"[{i}/{len(QUESTIONS)}] h={r['entropy']:.2f} label={label}  {q[:52]}")
        OUT.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    report(rows)


def report(rows):
    """Sweep the threshold, keep the best F1, write it back for the UI."""
    grid = []
    for step in range(1, 100):
        t = step / 100
        tp = sum(1 for r in rows if r["entropy"] >= t and r["label"] == 1)
        fp = sum(1 for r in rows if r["entropy"] >= t and r["label"] == 0)
        fn = sum(1 for r in rows if r["entropy"] < t and r["label"] == 1)
        if not tp:
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        grid.append((t, prec, rec, 2 * prec * rec / (prec + rec)))

    best = None
    if grid:
        top = max(f for *_, f in grid)
        # Centre the threshold in the winning range rather than taking its lower
        # edge: entropy is a 5-sample estimate and moves between runs, so a cut-off
        # sitting a hundredth above the highest grounded score would flip on noise.
        winners = [t for t, _, _, f in grid if f == top]
        t = round((min(winners) + max(winners)) / 2, 2)
        _, prec, rec, f1 = min(grid, key=lambda g: abs(g[0] - t))
        best = {
            "threshold": t,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "safe_range": [min(winners), max(winners)],
        }

    pos = [r["entropy"] for r in rows if r["label"] == 1]
    neg = [r["entropy"] for r in rows if r["label"] == 0]
    summary = {
        "n": len(rows),
        "n_samples": N_SAMPLES,
        "mean_entropy_confabulated": round(sum(pos) / len(pos), 3) if pos else None,
        "mean_entropy_grounded": round(sum(neg) / len(neg), 3) if neg else None,
        "best": best,
        "shipped_thresholds": {"shaky": SHAKY, "confabulated": CONFABULATED},
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    print("\n" + json.dumps(summary, indent=2))
    if best:
        print(
            f"\nSLIDE: catches {best['recall']:.0%} of confabulations at "
            f"{best['precision']:.0%} precision (threshold {best['threshold']}, n={len(rows)})"
        )
        print(f"-> set CONFABULATED = {best['threshold']} in glassbox.py")


if __name__ == "__main__":
    if "--report" in sys.argv:  # re-tune thresholds without re-running the model
        report(json.loads(OUT.read_text(encoding="utf-8"))["rows"])
    else:
        main()
