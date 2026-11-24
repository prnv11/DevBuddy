# Part 4: relevance feedback (Rocchio) and pseudo-relevance feedback

Judgments: **provisional (author gold patterns)**. Test split (60 queries). Rocchio alpha=1, beta=0.75, gamma=0.15; 20 expansion terms (tuned on dev). The simulated user judges the top 5 with the qrels; scores are on the **residual collection** (judged documents removed from both rankings).

## Explicit feedback

| system | feedback | judged removed | nDCG@10 before | nDCG@10 after | delta nDCG | p (bootstrap) | wins/losses | MAP before | MAP after |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25f | explicit, 1 round | 5 | 0.317 | 0.414 | 0.097 | 0.002 | 32/16 | 0.274 | 0.380 |
| bm25f | explicit, 2 rounds | 10 | 0.237 | 0.376 | 0.139 | 0.001 | 30/13 | 0.205 | 0.330 |
| lsa | explicit, 1 round | 5 | 0.290 | 0.397 | 0.107 | 0.000 | 31/6 | 0.282 | 0.380 |
| lsa | explicit, 2 rounds | 10 | 0.243 | 0.419 | 0.175 | 0.000 | 38/5 | 0.230 | 0.376 |
| hybrid_lsa | explicit, 1 round | 5 | 0.321 | 0.410 | 0.089 | 0.002 | 28/18 | 0.280 | 0.383 |
| hybrid_lsa | explicit, 2 rounds | 10 | 0.255 | 0.367 | 0.112 | 0.004 | 27/16 | 0.228 | 0.345 |
| minilm | explicit, 1 round | 5 | 0.248 | 0.344 | 0.095 | 0.000 | 37/9 | 0.208 | 0.285 |
| minilm | explicit, 2 rounds | 10 | 0.169 | 0.326 | 0.156 | 0.000 | 37/9 | 0.145 | 0.273 |
| hybrid_minilm | explicit, 1 round | 5 | 0.310 | 0.440 | 0.129 | 0.000 | 34/12 | 0.279 | 0.399 |
| hybrid_minilm | explicit, 2 rounds | 10 | 0.225 | 0.375 | 0.150 | 0.000 | 32/14 | 0.217 | 0.346 |

## Pseudo-relevance feedback (BM25F, top 5 assumed relevant)

| variant | applied to | nDCG@10 base | nDCG@10 PRF | delta | p | residual delta | residual p | MAP base | MAP PRF |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PRF, always on | 60/60 | 0.449 | 0.451 | 0.001 | 0.919 | 0.028 | 0.221 | 0.351 | 0.380 |
| PRF, confidence gate 0.52 (dev) | 13/60 | 0.449 | 0.451 | 0.001 | 0.846 | 0.019 | 0.093 | 0.351 | 0.359 |
