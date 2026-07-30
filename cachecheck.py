"""Can the demo replay from cache alone, with no model weights present?

The Kaggle notebook and a fresh clone both depend on this: every generation and
judgement is keyed by its exact prompt text, so ANY prompt edit silently
invalidates the cached entries and the demo starts demanding a 2.9GB download.

Run this before committing cache.json. It fakes away the weights, so a cache
miss fails loudly here instead of in front of a judge.

    python cachecheck.py
"""
import pathlib
import sys

import glassbox as g

PRESETS = [
    "What is the atomic radius of ununoctium in picometres?",
    "Who painted the Mona Lisa?",
    "Why did Nikola Tesla decline his 1917 Nobel Prize in Chemistry?",
    "In which year did Ronald Fisher publish his first paper on the Behrens-Fisher problem?",
]

g.MODEL_PATH = pathlib.Path("no-weights-on-purpose.gguf")
g.JUDGE_PATH = pathlib.Path("no-weights-on-purpose.gguf")

ok = True
for q in PRESETS:
    try:
        r = g.inspect(q)
        rep = g.repair(q, r)
        print(f"  cached  fork={str(r['fork']):>4}  claims={r['n_claims']}  "
              f"repair={'yes' if rep else 'none'}  {q[:52]}")
    except Exception as e:
        ok = False
        print(f"  MISS    {q[:52]}\n          {type(e).__name__}: {str(e)[:110]}")

print()
if ok:
    print("all presets replay from cache with no weights - safe to commit cache.json")
else:
    print("SOME PRESETS NEED THE MODEL.")
    print("Re-run them once with the weights present so the current prompts get cached:")
    for q in PRESETS:
        print(f'    python show.py "{q}"')
    print("then re-run this check before committing.")
    sys.exit(1)
