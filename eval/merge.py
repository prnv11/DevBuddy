"""
Merge the judge sheets into eval/qrels.tsv and report agreement.

    python -m eval.merge

* Cohen's kappa (unweighted and linear-weighted) on the overlap queries,
  per judge pair, plus the confusion counts.
* Disagreements on overlap queries are resolved by taking the lower grade
  when they differ by 1, and are listed in eval/judging/disagreements.tsv
  (differences of 2 should be discussed and fixed in the sheets).
"""
from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict

from eval.common import EVAL_DIR, cohen_kappa, write_qrels


def main():
    jdir = EVAL_DIR / "judging"
    grades: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for p in sorted(jdir.glob("judge_*.tsv")):
        judge = p.stem.split("_", 1)[1]
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["grade"].strip() in ("0", "1", "2"):
                    grades[(r["qid"], r["chunk_id"])][judge] = int(r["grade"])
    if not grades:
        raise SystemExit("no grades found in eval/judging/judge_*.tsv")

    report = {"pairs": {}, "judged_pairs": len(grades)}
    judges = sorted({j for g in grades.values() for j in g})
    for a, b in itertools.combinations(judges, 2):
        x, y = [], []
        for g in grades.values():
            if a in g and b in g:
                x.append(g[a]); y.append(g[b])
        if x:
            report["pairs"][f"{a}-{b}"] = {
                "n": len(x), "agreement": sum(i == j for i, j in zip(x, y)) / len(x),
                "kappa": cohen_kappa(x, y), "kappa_linear": cohen_kappa(x, y, weights="linear"),
                "kappa_binary": cohen_kappa([int(i > 0) for i in x], [int(j > 0) for j in y], labels=(0, 1)),
            }
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    dis = []
    for (qid, cid), g in grades.items():
        vals = list(g.values())
        if len(set(vals)) > 1:
            dis.append((qid, cid, json.dumps(g)))
        qrels[qid][cid] = min(vals) if max(vals) - min(vals) == 1 else round(sum(vals) / len(vals))
    write_qrels(qrels, EVAL_DIR / "qrels.tsv", header=f"merged human judgments from {', '.join(judges)}")
    with open(jdir / "disagreements.tsv", "w", encoding="utf-8") as f:
        f.write("qid\tchunk_id\tgrades\n")
        for row in sorted(dis):
            f.write("\t".join(row) + "\n")
    report["disagreements"] = len(dis)
    (EVAL_DIR / "results").mkdir(exist_ok=True)
    (EVAL_DIR / "results" / "agreement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
