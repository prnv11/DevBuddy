# DevBuddy IR extensions

Four additions on one shared corpus, all measured: **(0)** a polite crawler,
**(1)** a self-built zoned index with BM25/BM25F, dense and hybrid retrieval and an evaluation
harness, **(2)** clustering for Flask ↔ FastAPI concept alignment, **(3)** PageRank / HITS,
**(4)** Rocchio relevance feedback, wired into the dashboard.

> **Read this first: the numbers below use *provisional* judgments.**
> They come from gold answer patterns written with each query (`eval/queries.jsonl`), not from
> the pooled human judging the plan calls for. The harness for human judging is complete
> (pool → judge sheets → kappa → `eval/qrels.tsv`). Once `eval/qrels.tsv` exists, every script
> uses it automatically. Re-run `scripts/reproduce.sh` and replace these numbers before you
> report them.

---

## How to run

```bash
pip install -r requirements.txt
python -m ir.crawler --out data            # step 0, live crawl, ~4 min at 1 s/host
bash scripts/reproduce.sh                  # tests + every experiment -> eval/results/
uvicorn app.main:app --reload              # dashboard with real retrieval + 👍/👎 refine
```

Human judging, split between the two authors (about 2–3 h each):

```bash
python -m eval.pool                        # pool top-10 of every system -> eval/judging/judge_A.tsv, judge_B.tsv
python -m eval.judge eval/judging/judge_A.tsv   # Pranav; Aastha uses judge_B.tsv (resumable: 0/1/2, s, b, q)
python -m eval.merge                       # Cohen's kappa on the 20 shared queries -> eval/qrels.tsv
bash scripts/reproduce.sh                  # all tables now use human judgments (+ judged@10)
```

`eval.pool` refuses to overwrite sheets that already contain grades. Install
`sentence-transformers` **before** pooling, so MiniLM results are in the pool.

## What is where

| Path | What it is |
|:---|:---|
| `ir/crawler.py`, `ir/extract.py` | Step 0: robots.txt, ≥1 s per host, URL normalisation, trap filters (query strings, old versions, translations, deep or repeating paths), SHA-256 and Jaccard 5-shingle dedup, main-content-only links, heading-bounded chunks with a separate `code` field, raw HTML kept for `--reextract` |
| `ir/text.py` | Code-aware tokenizer: `request.args` → `request.args`, `request`, `args` (same position) |
| `ir/index.py` | Inverted index per zone (title, heading, body, code) with `(doc, tf, positions)` postings, df and lengths; BM25, BM25F, weighted BM25F; term-at-a-time accumulators + heap; `framework:`, `section:`, `+term`, `-term`, `code:` filters |
| `ir/dense.py` | MiniLM (sentence-transformers) and LSA retrievers; reciprocal rank fusion |
| `ir/cluster.py` | Spherical k-means, HAC, cluster labels, centroid / Hungarian alignment, no-match threshold, cluster pruning, cluster-personalised PageRank |
| `ir/links.py` | PageRank (dangling → uniform, tol 1e-8), personalised PageRank, query-time HITS |
| `ir/feedback.py` | Rocchio (α=1, β=0.75, γ=0.15, negatives clipped, top-k expansion) on terms and on embeddings; gated PRF |
| `ir/engine.py`, `ir/answer.py` | Every system behind one `run(system, query)` call; answer composition with citations (Groq if `GROQ_API_KEY` is set, otherwise extractive) |
| `eval/` | Queries (90), concept pairs (30 + 7 no-match), pooling, judging, merging + kappa, metrics, significance, all experiments |
| `eval/results/` | Generated tables (`part1-4.md`), JSON, `pagerank_convergence.png`, `concept_dendrogram.png` |
| `tools/` | Offline docs mirror (only needed where the live sites are unreachable) |
| `tests/` | 17 tests, including PageRank/PPR/HITS against networkx, kappa against scikit-learn, BM25 against the formula |

## Step 0: data (live crawl)

211 pages (FastAPI 140, Flask 71), **2,967 chunks** (1,779 with code), 493 content-link edges,
157 s. Minimum gap between requests to one host: 1.00 s on both sites. Skipped: 388 off-site,
22 denied sections, 12 out of scope, one 404. Chunk ids are readable and stable,
e.g. `flask/api#flask.Request.args`.

## Results (test split, 60 queries; tuned on 30 dev queries)

**Part 1: baselines.** The expected three-way pattern holds:

| nDCG@10 | overall | exact API | how-to | cross-framework |
|:---|---:|---:|---:|---:|
| BM25F (tuned) | 0.434 | **0.711** | 0.446 | 0.130 |
| Dense MiniLM | 0.341 | 0.400 | 0.407 | **0.206** |
| Hybrid RRF (BM25F + MiniLM) | **0.472** | 0.710 | **0.509** | 0.179 |

* BM25F wins on exact API names. On that query type MiniLM loses to it on all 19 queries
  where they differ (−0.310, p < 0.001).
* MiniLM wins on cross-framework paraphrases (+0.076 over BM25F, p = 0.019).
* The hybrid wins overall: +0.037 over BM25F (p = 0.005, wins/losses 37/11), and it is
  significantly better on how-to (+0.063, p = 0.001) and cross-framework (+0.049, p < 0.001) queries.
* Plain BM25 ≈ BM25F (+0.004, n.s.). The tuned BM25F (k1 = 2.1, b = 0.5) is slightly below the
  untuned one on test (0.434 vs 0.444): a mild overfit to 30 dev queries.
* LSA behaves like a weaker MiniLM (0.375 overall; hybrid with LSA 0.451).
* Latency: BM25F about 1.5 ms per query, hybrid with MiniLM about 15 ms.

**Part 2: alignment.** On 20 held-out concept pairs, the configuration selected on the seeds
(LSA, k-means, evidence routing, 5 clusters searched) gets hit@1 = 0.45, hit@3 = 0.75 and
MRR = 0.612. The BM25F baseline gets 0.45 / 0.60 / 0.527, and the MiniLM baseline 0.30 / 0.45 / 0.398.
That is +0.085 MRR, but **not significant** (p = 0.36, n = 20).

* Cluster quality: HAC beats k-means on FastAPI against the URL-section labels (ARI 0.27 vs 0.13–0.19).
  Flask labels are too coarse to separate methods (4 sections, random purity already 0.82).
* MiniLM is not a better shared space for alignment than tf-idf or LSA here.
* Seed tuning chose evidence routing in 11 of 12 variants, and never chose the Hungarian 1:1 mapping.
* Pruning: searching 5 of 60 clusters (7 % of FastAPI chunks) keeps hit@10 at 0.85, the same
  as an exhaustive search. On this corpus that saves only about 3 ms per query (2.3 vs 5.1 ms).
* No-match threshold (τ = 0.153): its 1 "no match" prediction was correct (precision 1.0), but
  it caught only 1 of 5 real no-match concepts (recall 0.2). `flask.g`, for example, gets a
  confident wrong match.
* The dendrogram's first cross-framework merges are real concept pairs (errorhandler ↔ HTTPException,
  Celery ↔ BackgroundTasks, gevent/async ↔ async, static folder ↔ StaticFiles), plus one
  spurious pair (installation ↔ an "AI skills" page).
* Typical failures: `request.args` and `before_request` are routed to FastAPI's *Request class*
  cluster, because they share the word "request".

**Part 3: link analysis.** Mostly a negative result, as expected.

* Dev tuning sets β = 0 for global PageRank and both personalised PageRanks.
* HITS gets β = 0.02: +0.009 nDCG (bootstrap p = 0.043, but Wilcoxon p = 0.14, and this is one
  of 8 comparisons). Treat it as no reliable gain.
* With β *fixed* at 0.1, cluster-personalised PageRank gives +0.016 (p = 0.003). That value was
  not chosen on dev, so it is a post-hoc observation and must not be reported as the tuned
  result. It does justify tuning β on a larger dev set.
* The implementation matches `networkx.pagerank` to 8.9e-10 (personalised: 3.4e-9) and converges
  in 93 iterations.
* Why links don't help: 493 content edges, and 63 pages have no content out-links. PageRank
  concentrates on hubs such as `flask/api/` (0.088). The 19,709 navigation edges would flatten
  the scores (Gini 0.349 vs 0.532 for the content graph).
* Surprise: for the query `dependencies`, HITS *hubs* are the FastAPI **reference** pages and the
  authorities are tutorial pages. That is the opposite of the expected roles, because FastAPI's
  reference docstrings link out to the tutorials.

**Part 4: feedback.** On the residual collection (judged documents removed), explicit Rocchio
helps every system:

| system | 1 round | 2 rounds |
|:---|---:|---:|
| BM25F | +0.083 | +0.115 |
| MiniLM | +0.084 | +0.155 |
| Hybrid (BM25F + MiniLM) | +0.090 | +0.121 |

All gains have p ≤ 0.007. PRF without a gate *hurts* slightly (−0.013, n.s.). With the
confidence gate it runs on 15 of 60 queries and gives +0.007 overall (+0.026 on the residual
collection, p = 0.03).
**Caveat:** the provisional judgments mark whole answer pages as relevant, so one 👍 pulls in
sibling chunks that also count as relevant. This probably inflates the feedback gains; check
them again with human judgments.

## Pitfalls to state in the report

* Pooling favours the pooled systems; report judged@10 for any system added later.
* Provisional judgments are page-level and written by the query author, so they are not
  independent judges, and no kappa can be computed for them.
* Alignment test set: n = 20 pairs plus 5 no-match concepts, so effects below about 0.15 MRR
  can't be detected.
* For Flask `g` the gold list says "no counterpart" (as in the plan), while query X04 treats
  FastAPI dependencies as the practical replacement. Decide which framing the report uses.

## Environment notes

The development environment could not reach the docs sites, so the code was first built and
tested against an offline mirror (`tools/build_mirror.py`, `tools/serve_mirror.py`). **All numbers
above come from the live crawl** (`python -m ir.crawler --out data`), run on 20 Nov 2026,
with MiniLM (sentence-transformers all-MiniLM-L6-v2).

## Changes to existing code

* `app/rag/pipeline/router.py` previously returned **three hard-coded answers after a fake
  2.8 s delay**. It now calls the real pipeline.
* `/user/chat` returns cited sources. The new `/user/refine` endpoint runs Rocchio, and the
  dashboard shows 👍/👎 per source, a Refine button and rank-movement markers.
* `TemplateResponse(request, name, ctx)`: the old call style crashes on current Starlette.
* The README's former benchmark table (P@3 0.88, MRR 0.91, …) and its `app/crawler/`, dedup
  and MMR modules have no supporting code in the repository; that section now points here.
* `vectorstore/chroma_db` (OpenAI 1536-d, FastAPI only) is no longer used, and the old
  `retriever.py` (MiniLM 384-d) could not have queried it. It can be deleted.
