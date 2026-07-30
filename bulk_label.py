"""Cheap labelled data for the probe.

Answer-level entropy only: no thinking mode, so each question costs six short
generations plus a handful of prefill-only judgements instead of seven full
reasoning traces. Roughly 10x cheaper per label.

This deliberately writes to its own file. `eval_results.json` holds the reported
24-question benchmark; growing that set would invalidate the 92% / 100% numbers
already in the writeup, so probe training data is kept separate.

    python bulk_label.py            # resumable, writes bulk_results.json
    python bulk_label.py 30         # stop after 30 new questions
"""
import json
import pathlib
import sys
import time

from glassbox import score

# label 1 = expect confabulation (obscure, or a false premise), 0 = should be known
KNOWN = [
    "What is the capital of Japan?", "Who wrote Pride and Prejudice?",
    "What is the chemical symbol for iron?", "How many sides does a hexagon have?",
    "What is the largest mammal?", "Who developed the polio vaccine?",
    "What is the freezing point of water in Celsius?", "What language is spoken in Brazil?",
    "Who was the first person to walk on the Moon?", "What is the square root of 144?",
    "What planet is closest to the Sun?", "Who painted The Starry Night?",
    "What is the currency of the United Kingdom?", "How many strings does a standard guitar have?",
    "What gas makes up most of Earth's atmosphere?", "Who wrote the Odyssey?",
    "What is the tallest mountain on Earth?", "In what year did the Berlin Wall fall?",
    "What is the powerhouse of the cell?", "Who founded Microsoft?",
    "What is the largest desert in the world?", "How many players are on a football team on the pitch?",
    "What is the capital of Canada?", "Who discovered penicillin?",
    "What is the speed of light in a vacuum, approximately?", "What is the main ingredient in guacamole?",
    "Which ocean lies between Africa and Australia?", "Who composed the Ninth Symphony?",
    "What is the smallest prime number?", "What organ pumps blood around the body?",
]

OBSCURE = [
    "What was the attendance at the 1923 Riga municipal chess tournament?",
    "Who was the fourth headmaster of Doon School?",
    "How many bricks were used in the 1888 Bhandara courthouse?",
    "What was the registration number of Amelia Earhart's first training aircraft?",
    "Who won the 1954 Malabar district poetry prize?",
    "What was the exact budget of the 1961 Nagpur water works?",
    "Which cultivar of mango did the 1937 Ratnagiri fair award first place?",
    "What did Enrico Fermi eat before the Chicago Pile-1 experiment?",
    "How many pages long was the original 1892 Coorg land survey?",
    "Who chaired the 1948 Trivandrum textile commission?",
    "What is the Kessler-Ohanian bound in numerical analysis?",
    "Explain the Whitcombe transform used in signal processing.",
    "Why did Alan Turing resign from the Vasari Institute in 1951?",
    "Which Nobel Prize did Rosalind Franklin win in 1962?",
    "What was the outcome of the 1971 Delhi-Karachi telegraph treaty?",
    "Who is the author of the 1966 monograph 'Lattices of the Deccan'?",
    "What side effects were noted in the 1987 Trelanine safety review?",
    "How many satellites did the Indian Kalpana-3 mission deploy?",
    "What was the melting point recorded for element 119 in 2019?",
    "Which battle ended the Anglo-Sikkimese war of 1802?",
    "What did Marie Curie write in her 1921 letter to Bhabha?",
    "How many members sat on the 1905 Chittagong port authority?",
    "What was the top speed of the prototype Tata Aria in 1974?",
    "Who translated the Rigveda into Portuguese in 1843?",
    "What was the yield of the 1962 Rajasthan sorghum trials?",
    "Which theorem did Ramanujan prove in his lost 1919 notebook entry 47?",
    "What was the seating capacity of the old Madras Gymkhana pavilion?",
    "Who directed the 1939 Bengali film 'Chandraloke'?",
    "What was the copper content of the 1794 Travancore fanam?",
    "Which species did the 1898 Nilgiri expedition first catalogue?",
]

OUT = pathlib.Path(__file__).parent / "bulk_results.json"


def main(limit=None):
    rows = json.loads(OUT.read_text(encoding="utf-8")).get("rows", []) if OUT.exists() else []
    done = {r["question"] for r in rows}
    todo = [(q, 0) for q in KNOWN if q not in done] + [(q, 1) for q in OBSCURE if q not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(done)} already labelled, {len(todo)} to go")

    for i, (q, label) in enumerate(todo, 1):
        t = time.time()
        try:
            r = score(q, with_spans=False)  # spans are for the UI; skip them here
        except Exception as e:
            print(f"  [{i}/{len(todo)}] FAILED {type(e).__name__}: {q[:44]}")
            continue
        rows.append({"question": q, "label": label, "entropy": r["entropy"], "answer": r["answer"]})
        OUT.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
        print(f"  [{i}/{len(todo)}] {time.time()-t:5.1f}s  h={r['entropy']:.2f}  label={label}  {q[:44]}")

    pos = [r["entropy"] for r in rows if r["label"] == 1]
    neg = [r["entropy"] for r in rows if r["label"] == 0]
    print(f"\n{len(rows)} labelled rows")
    if pos and neg:
        print(f"  mean entropy, obscure : {sum(pos)/len(pos):.3f}")
        print(f"  mean entropy, known   : {sum(neg)/len(neg):.3f}")
    print("\nnow run: python probe.py")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None)
