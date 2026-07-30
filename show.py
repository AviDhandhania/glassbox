"""Pretty-print a reasoning inspection in the terminal.

    python show.py "What is the atomic radius of ununoctium in picometres?"
    python show.py --repair "..."     # also adjudicate the diverging step
"""
import sys

from glassbox import FORK, inspect, repair

DEFAULT_Q = "What is the atomic radius of ununoctium in picometres?"


def main(argv):
    want_repair = "--repair" in argv
    question = " ".join(a for a in argv if a != "--repair") or DEFAULT_Q

    r = inspect(question)
    if "error" in r:
        sys.exit(r["error"])

    print(f"\nQ: {r['question']}")
    print(f"A: {r['answer'][:300]}\n")
    for s in r["steps"]:
        if not s["claim"]:
            print(f"  {s['index']}. [procedure]        {s['text'][:74]}")
            continue
        flag = "<<< FORK" if s["index"] == r["fork"] else ""
        gap = f" (+{s['absent']} traces skipped this)" if s.get("absent") else ""
        tag = "ANSWER" if s.get("is_answer") else f"{s['index']}."
        print(f"  {tag} h={s['entropy']:<5} {s['n_clusters']} of {s['n_aligned']} readings{gap}  "
              f"{s['text'][:58]} {flag}")
        for v in s["variants"]:
            print(f"        also: {v[:88]}")

    print(f"\nfork at step {r['fork']}   |   {r['n_claims']} claim steps of {r['depth']}   |   "
          f"mean claim entropy {r['mean_entropy']}   |   {r['n_traces']} traces   |   fork cut-off {FORK}")

    if not want_repair:
        return
    rep = repair(question, r)
    if not rep:
        print("\nrepair: no step with two or more competing factual claims")
    elif rep.get("declined"):
        print(f"\nrepair: step {rep['step']} - shown its own {len(rep['readings'])} readings, "
              f"the model rejected ALL of them. Answer left alone and flagged.")
        for t in rep["readings"]:
            print(f"    {t[:96]}")
    else:
        print(f"\nrepair: step {rep['step']} - picked #{rep['chosen_index'] + 1}"
              f"{' (its original)' if rep['was_original'] else ' - CHANGED ITS MIND'}")
        print(f"  BEFORE: {rep['answer_before'][:200]}")
        print(f"  AFTER : {rep['answer_after'][:200]}")


if __name__ == "__main__":
    main(sys.argv[1:])
