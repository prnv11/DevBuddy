# Part 1: lexical, dense and hybrid retrieval

Judgments: **provisional (author gold patterns)**. Test split: 60 queries (api=20, howto=21, cross=19); tuned on dev (30 queries, 14s grid search).

Columns api / howto / cross are nDCG@10 per query type.

| system | nDCG@10 | P@5 | R@10 | MAP | MRR | api | howto | cross | ms/q |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 (k1=1.2, b=0.75) | 0.410 | 0.460 | 0.256 | 0.314 | 0.640 | 0.616 | 0.470 | 0.128 | 1.669 |
| BM25 tuned | 0.434 | 0.450 | 0.248 | 0.312 | 0.700 | 0.696 | 0.453 | 0.136 | 1.745 |
| BM25F (default weights) | 0.445 | 0.457 | 0.251 | 0.329 | 0.704 | 0.703 | 0.480 | 0.136 | 1.596 |
| BM25F tuned | 0.449 | 0.467 | 0.252 | 0.351 | 0.736 | 0.687 | 0.476 | 0.168 | 1.599 |
| Dense LSA | 0.375 | 0.467 | 0.249 | 0.351 | 0.629 | 0.530 | 0.399 | 0.186 | 7.897 |
| Dense MiniLM | 0.339 | 0.393 | 0.201 | 0.260 | 0.593 | 0.388 | 0.412 | 0.206 | 12.706 |
| Hybrid RRF (BM25F + LSA) | 0.459 | 0.470 | 0.269 | 0.366 | 0.722 | 0.692 | 0.484 | 0.186 | 10.399 |
| Hybrid RRF (BM25F + MiniLM) | 0.464 | 0.507 | 0.283 | 0.362 | 0.741 | 0.638 | 0.542 | 0.197 | 14.926 |

## Tuned parameters (dev)

```json
{
 "bm25": {
  "k1": 2.1,
  "b": 0.3
 },
 "bm25f": {
  "k1": 2.1,
  "b": 0.75,
  "weights": {
   "title": 3.0,
   "heading": 2.5,
   "body": 1.0,
   "code": 0.5
  }
 },
 "hybrid_lsa": {
  "rrf_w": 0.8
 },
 "hybrid_minilm": {
  "rrf_w": 0.8
 }
}
```

## Paired significance (test)

| comparison (nDCG@10) | delta | p (bootstrap) | p (Wilcoxon) | wins/losses |
|:---|---:|---:|---:|---:|
| BM25F tuned vs BM25 tuned | +0.015 | 0.3881 | 0.5754 | 27/25 |
| Dense LSA vs BM25F tuned | -0.074 | **0.0017** | 0.0020 | 13/38 |
| Hybrid RRF (BM25F + LSA) vs BM25F tuned | +0.010 | 0.1995 | 0.0908 | 24/17 |
| Hybrid RRF (BM25F + LSA) vs Dense LSA | +0.084 | **0.0001** | 0.0003 | 41/12 |
| Dense MiniLM vs BM25F tuned | -0.111 | **0.0016** | 0.0028 | 17/39 |
| Hybrid RRF (BM25F + MiniLM) vs BM25F tuned | +0.015 | 0.2158 | 0.2071 | 27/21 |
| Hybrid RRF (BM25F + MiniLM) vs Dense MiniLM | +0.126 | **0.0000** | 0.0001 | 44/12 |

### By query type

| comparison (nDCG@10) | delta | p (bootstrap) | p (Wilcoxon) | wins/losses |
|:---|---:|---:|---:|---:|
| [api] Dense LSA vs BM25F tuned | -0.157 | **0.0003** | 0.0049 | 2/15 |
| [api] Hybrid RRF (BM25F + LSA) vs BM25F tuned | +0.005 | 0.5126 | 0.7532 | 5/8 |
| [api] Hybrid RRF (BM25F + LSA) vs Dense LSA | +0.162 | **0.0000** | 0.0031 | 15/2 |
| [api] Dense MiniLM vs BM25F tuned | -0.300 | **0.0000** | 0.0002 | 1/18 |
| [api] Hybrid RRF (BM25F + MiniLM) vs BM25F tuned | -0.050 | **0.0202** | 0.0090 | 2/13 |
| [api] Hybrid RRF (BM25F + MiniLM) vs Dense MiniLM | +0.250 | **0.0000** | 0.0002 | 18/1 |
| [howto] Dense LSA vs BM25F tuned | -0.078 | **0.0358** | 0.0479 | 5/15 |
| [howto] Hybrid RRF (BM25F + LSA) vs BM25F tuned | +0.008 | 0.6327 | 0.2485 | 13/5 |
| [howto] Hybrid RRF (BM25F + LSA) vs Dense LSA | +0.086 | **0.0011** | 0.0080 | 16/5 |
| [howto] Dense MiniLM vs BM25F tuned | -0.065 | 0.2082 | 0.2290 | 7/14 |
| [howto] Hybrid RRF (BM25F + MiniLM) vs BM25F tuned | +0.065 | **0.0004** | 0.0028 | 15/5 |
| [howto] Hybrid RRF (BM25F + MiniLM) vs Dense MiniLM | +0.130 | **0.0093** | 0.0142 | 17/4 |
| [cross] Dense LSA vs BM25F tuned | +0.018 | 0.5639 | 0.9750 | 6/8 |
| [cross] Hybrid RRF (BM25F + LSA) vs BM25F tuned | +0.017 | 0.2077 | 0.3329 | 6/4 |
| [cross] Hybrid RRF (BM25F + LSA) vs Dense LSA | -0.001 | 0.9802 | 0.6092 | 10/5 |
| [cross] Dense MiniLM vs BM25F tuned | +0.038 | 0.3751 | 0.5014 | 9/7 |
| [cross] Hybrid RRF (BM25F + MiniLM) vs BM25F tuned | +0.028 | 0.0788 | 0.0546 | 10/3 |
| [cross] Hybrid RRF (BM25F + MiniLM) vs Dense MiniLM | -0.010 | 0.7964 | 0.8361 | 9/7 |
