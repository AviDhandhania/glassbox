"""Pretty-print a reasoning inspection in the terminal.

    python show.py "What is the atomic radius of ununoctium in picometres?"
"""
import sys

from glassbox import FORK, inspect

r = inspect(" ".join(sys.argv[1:]) or "What is the atomic radius of ununoctium in picometres?")
if "error" in r:
    sys.exit(r["error"])

print(f"\nQ: {r['question']}")
print(f"A: {r['answer'][:300]}\n")
for s in r["steps"]:
    if not s["claim"]:
        print(f"  {s['index']}. [procedure]        {s['text'][:74]}")
        continue
    flag = "<<< FORK" if s["index"] == r["fork"] else ""
    print(f"  {s['index']}. h={s['entropy']:<5} {s['n_clusters']} readings  {s['text'][:64]} {flag}")
    for v in s["variants"]:
        print(f"        also: {v[:88]}")
print(f"\nfork at step {r['fork']}   |   {r['n_claims']} claim steps of {r['depth']}   |   "
      f"mean claim entropy {r['mean_entropy']}   |   {r['n_traces']} traces   |   fork cut-off {FORK}")
