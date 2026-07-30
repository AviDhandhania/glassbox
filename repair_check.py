"""Does the model repair its own fork? Recognition-vs-recall, measured.

For each question: inspect the reasoning, find the most divergent claim step,
show the model its own competing readings, let it pick, then re-run the
reasoning from that step.

The `truth` field is what a correct answer must contain - it is only used to
print a hint next to the result, never to choose anything.

    python repair_check.py            # resumable, writes repair_results.json
"""
import json
import pathlib
import sys
import time

from glassbox import inspect, repair

QUESTIONS = [
    ("What is the atomic radius of ununoctium in picometres?", "118"),
    ("In which year did Ronald Fisher publish his first paper on the Behrens-Fisher problem?", None),
    ("In the paper 'Attention Is All You Need', what learning rate warmup step count was used?", "4000"),
    ("What is the melting point of tungsten in degrees Fahrenheit?", "6192"),
    ("How many confirmed moons does Saturn have?", None),
    ("Which chess opening did Mikhail Tal play most often as Black in 1960?", None),
    ("What is the chemical symbol for tungsten?", "W"),
    ("Who painted the Mona Lisa?", "Leonardo"),
]

OUT = pathlib.Path(__file__).parent / "repair_results.json"


def main():
    rows = json.loads(OUT.read_text(encoding="utf-8")).get("rows", []) if OUT.exists() else []
    done = {r["question"] for r in rows}

    for i, (q, truth) in enumerate(QUESTIONS, 1):
        if q in done:
            continue
        t = time.time()
        insp = inspect(q)
        if "error" in insp:
            print(f"[{i}/{len(QUESTIONS)}] SKIP {insp['error']}")
            continue
        rep = repair(q, insp)
        row = {
            "question": q, "truth": truth, "fork": insp["fork"],
            "mean_entropy": insp["mean_entropy"], "n_claims": insp["n_claims"],
            "repair": rep,
        }
        rows.append(row)
        OUT.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

        print(f"\n[{i}/{len(QUESTIONS)}] {time.time()-t:.0f}s  {q}")
        if not rep:
            print("     no divergent claim step - nothing to repair")
            continue
        print(f"     step {rep['step']} (h={rep['entropy']}), {len(rep['readings'])} readings, "
              f"picked #{rep['chosen_index']+1}{' (its original)' if rep['was_original'] else ' - CHANGED ITS MIND'}")
        for j, r in enumerate(rep["readings"]):
            print(f"       {'>' if j == rep['chosen_index'] else ' '} {r[:96]}")
        print(f"     BEFORE: {rep['answer_before'][:150]}")
        print(f"     AFTER : {rep['answer_after'][:150]}")
        if truth:
            was = truth.lower() in rep["answer_before"].lower()
            now = truth.lower() in rep["answer_after"].lower()
            print(f"     contains {truth!r}: before={was} after={now}"
                  f"{'   <== REPAIRED' if now and not was else ''}"
                  f"{'   <== BROKE IT' if was and not now else ''}")

    report(rows)


def report(rows):
    reps = [r for r in rows if r["repair"]]
    changed = [r for r in reps if not r["repair"]["was_original"]]
    scored = [r for r in reps if r["truth"]]
    fixed = [r for r in scored
             if r["truth"].lower() in r["repair"]["answer_after"].lower()
             and r["truth"].lower() not in r["repair"]["answer_before"].lower()]
    broke = [r for r in scored
             if r["truth"].lower() in r["repair"]["answer_before"].lower()
             and r["truth"].lower() not in r["repair"]["answer_after"].lower()]
    summary = {
        "n": len(rows),
        "had_divergent_step": len(reps),
        "model_changed_its_mind": len(changed),
        "checkable": len(scored),
        "repaired": len(fixed),
        "regressed": len(broke),
        "repaired_questions": [r["question"] for r in fixed],
        "regressed_questions": [r["question"] for r in broke],
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSLIDE: shown its own competing readings, the model changed its answer on "
          f"{len(changed)}/{len(reps)} forks; {len(fixed)} became correct, {len(broke)} regressed")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(json.loads(OUT.read_text(encoding="utf-8"))["rows"])
    else:
        main()
