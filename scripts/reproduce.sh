#!/usr/bin/env bash
# Rebuild every number in eval/results/ from the crawled corpus in data/.
# Crawl first (live):  python -m ir.crawler --out data
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q tests/                 # algorithm checks (BM25, PageRank vs networkx, kappa, ...)
python -m eval.pool                        # pool + judge sheets + provisional qrels (refuses to overwrite grades)
python -m eval.run part1                   # BM25 / BM25F / dense / hybrid, tuned on dev, reported on test
python -m eval.alignment                   # part 2: clustering + concept alignment
python -m eval.run part3                   # PageRank / personalised PageRank / HITS
python -m eval.feedback                    # part 4: Rocchio + PRF on the residual collection
if ls eval/judging/judge_*.tsv >/dev/null 2>&1 && grep -qP '\t[012]$' eval/judging/judge_*.tsv; then
  python -m eval.merge                     # kappa + eval/qrels.tsv (then re-run the steps above)
fi
