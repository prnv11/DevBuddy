"""
Build the judging pool and the judge sheets.

    python -m eval.pool [--depth 10] [--judges A B C] [--overlap 20]

1. Runs every system on every query and takes the top `depth` from each.
2. Merges them into one pool per query (eval/pool.tsv). Pool order is
   shuffled per query so judges cannot tell which system found what.
3. Splits the queries between the judges (default: two); `overlap` queries are given to two judges
   so inter-annotator agreement (Cohen's kappa) can be measured.
   -> eval/judging/judge_<X>.tsv   (grade column empty; fill 0 / 1 / 2)
4. Writes eval/qrels_author.tsv: provisional judgments from the gold
   patterns in queries.jsonl, used until human judgments exist.

Pooling bias: documents only a non-pooled system retrieves are unjudged and
count as non-relevant. Every result table therefore reports judged@10.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from eval.common import EVAL_DIR, author_qrels, load_queries, write_qrels
from ir.engine import Engine

GUIDELINES = """\
Judge each (query, chunk) pair on what the chunk says, using its URL if needed.
  2 = answers the query (for "equivalent of X" queries: shows the counterpart
      concept in the target framework)
  1 = related / partially useful (right topic but does not answer; or shows the
      source-framework concept for an "equivalent" query)
  0 = not relevant
Do not look at which system retrieved it. Judge independently of other judges.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--judges", nargs="+", default=["A", "B"])
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite judge sheets that already contain grades")
    a = ap.parse_args()

    jdir = EVAL_DIR / "judging"
    graded = [p for p in jdir.glob("judge_*.tsv")
              if any(r.get("grade", "").strip() for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))]
    if graded and not a.force:
        raise SystemExit(f"refusing to overwrite judge sheets that already contain grades: "
                         f"{', '.join(p.name for p in graded)} (use --force)")

    eng = Engine(a.data)
    queries = load_queries()
    ids = [c["id"] for c in eng.chunks]

    qa = author_qrels(queries, ids)
    write_qrels(qa, EVAL_DIR / "qrels_author.tsv",
                header="PROVISIONAL judgments derived from the gold patterns in queries.jsonl.\n"
                       "Replace with human judgments: eval.pool -> eval.judge -> eval.merge")
    empty = [q["qid"] for q in queries if not qa[q["qid"]]]
    if empty:
        print("WARNING: gold patterns match nothing for", empty)

    systems = a.systems or eng.systems() + ["bm25f+pr", "bm25f+hits"]
    pool: dict[str, dict[str, set]] = {q["qid"]: {} for q in queries}
    for s in systems:
        for q in queries:
            for d, _ in eng.run(s, q["query"], k=a.depth):
                pool[q["qid"]].setdefault(ids[d], set()).add(s)
    rng = random.Random(13)
    with open(EVAL_DIR / "pool.tsv", "w", encoding="utf-8") as f:
        f.write("qid\tchunk_id\tsystems\n")
        for qid, docs in pool.items():
            for cid, ss in sorted(docs.items()):
                f.write(f"{qid}\t{cid}\t{','.join(sorted(ss))}\n")
    sizes = [len(v) for v in pool.values()]
    print(f"pool: {sum(sizes)} pairs, {min(sizes)}-{max(sizes)} per query "
          f"(mean {sum(sizes) / len(sizes):.1f}) from {len(systems)} systems at depth {a.depth}")

    # assign queries to judges: round-robin, plus an overlap set judged twice
    qids = [q["qid"] for q in queries]
    order = qids[:]
    rng.shuffle(order)
    J = a.judges
    assign = {j: [] for j in J}
    for i, qid in enumerate(order):
        assign[J[i % len(J)]].append(qid)
    overlap = order[: a.overlap]
    for i, qid in enumerate(overlap):
        primary = J[order.index(qid) % len(J)]
        second = J[(J.index(primary) + 1) % len(J)]
        assign[second].append(qid)

    qtext = {q["qid"]: q["query"] for q in queries}
    by_id = {c["id"]: c for c in eng.chunks}
    out = EVAL_DIR / "judging"
    out.mkdir(exist_ok=True)
    (out / "GUIDELINES.txt").write_text(GUIDELINES)
    (out / "overlap_qids.txt").write_text("\n".join(sorted(overlap)) + "\n")
    for j, qs in assign.items():
        with open(out / f"judge_{j}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["qid", "query", "chunk_id", "url", "heading", "snippet", "grade"])
            for qid in sorted(qs):
                docs = list(pool[qid])
                rng.shuffle(docs)
                for cid in docs:
                    c = by_id[cid]
                    snippet = (c["text"][:300] + (" [code] " + c["code"][:200] if c["code"] else "")).replace("\t", " ").replace("\n", " ")
                    w.writerow([qid, qtext[qid], cid, c["url"], " > ".join(c["heading_path"]), snippet, ""])
        print(f"judge {j}: {len(qs)} queries, {sum(len(pool[q]) for q in qs)} pairs -> {out / f'judge_{j}.tsv'}")


if __name__ == "__main__":
    main()
