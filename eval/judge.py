"""
Terminal judging tool for one judge sheet. Progress is saved after every
answer, so you can stop (q) and resume any time.

    python -m eval.judge eval/judging/judge_A.tsv

Keys: 0 / 1 / 2 = grade, s = skip, b = back, q = save and quit.
"""
from __future__ import annotations

import csv
import sys
import textwrap
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    fields = list(rows[0].keys())
    print(open(Path(__file__).parent / "judging" / "GUIDELINES.txt").read() if
          (Path(__file__).parent / "judging" / "GUIDELINES.txt").exists() else "")

    def save():
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            w.writerows(rows)

    i = next((k for k, r in enumerate(rows) if r["grade"] == ""), len(rows))
    while i < len(rows):
        r = rows[i]
        done = sum(1 for x in rows if x["grade"] != "")
        print("\n" + "=" * 78)
        print(f"[{done}/{len(rows)}]  {r['qid']}: {r['query']}")
        print(f"  {r['url']}\n  {r['heading']}")
        print(textwrap.indent(textwrap.fill(r["snippet"], 74), "  | "))
        k = input("grade 0/1/2 (s skip, b back, q quit): ").strip().lower()
        if k in ("0", "1", "2"):
            r["grade"] = k
            save()
            i += 1
        elif k == "s":
            i += 1
        elif k == "b":
            i = max(0, i - 1)
        elif k == "q":
            break
    save()
    print(f"saved {path}")


if __name__ == "__main__":
    main()
