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

def _no_weights(path=None):
    """Stand in for _llm() so any cache miss fails here instead of loading 2.9GB.

    Repointing MODEL_PATH/JUDGE_PATH cannot do this: every cache key embeds
    MODEL_PATH.name (see think/_gen/p_yes/repair), so renaming the model renames
    every key and the check misses everything by construction - it would fail
    even against a perfectly warm cache. Blocking the loader leaves the keys
    untouched, which is what actually needs testing.
    """
    raise RuntimeError("cache miss - this preset would need the model")


g._llm = _no_weights

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
