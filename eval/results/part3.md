# Part 3: link analysis (PageRank, personalised PageRank, HITS)

Judgments: **provisional (author gold patterns)**. Test split (60 queries); beta tuned on dev. Score = BM25F/max(BM25F) + beta * g(page), g = log-scaled min-max of the page score, re-ranking the BM25F top 100.

| system | beta | nDCG@10 | P@5 | R@10 | MAP | MRR | api | howto | cross |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25F tuned | 0.000 | 0.449 | 0.467 | 0.252 | 0.351 | 0.736 | 0.687 | 0.476 | 0.168 |
| + PageRank | 0.300 | 0.421 | 0.413 | 0.221 | 0.307 | 0.707 | 0.689 | 0.426 | 0.135 |
| + personalised PageRank (user stack) | 0.300 | 0.449 | 0.433 | 0.242 | 0.330 | 0.725 | 0.705 | 0.469 | 0.157 |
| + HITS authority | 0.100 | 0.447 | 0.463 | 0.252 | 0.344 | 0.733 | 0.672 | 0.493 | 0.159 |
| + personalised PageRank (query cluster) | 0.000 | 0.449 | 0.467 | 0.252 | 0.351 | 0.736 | 0.687 | 0.476 | 0.168 |
| + PageRank (beta=0.1 fixed) | 0.100 | 0.437 | 0.460 | 0.244 | 0.337 | 0.726 | 0.679 | 0.460 | 0.158 |
| + personalised PageRank (user stack) (beta=0.1 fixed) | 0.100 | 0.446 | 0.467 | 0.248 | 0.345 | 0.731 | 0.681 | 0.476 | 0.167 |
| + HITS authority (beta=0.1 fixed) | 0.100 | 0.447 | 0.463 | 0.252 | 0.344 | 0.733 | 0.672 | 0.493 | 0.159 |
| + personalised PageRank (query cluster) (beta=0.1 fixed) | 0.100 | 0.453 | 0.480 | 0.257 | 0.353 | 0.734 | 0.683 | 0.503 | 0.155 |

## Significance (test)

| comparison (nDCG@10) | delta | p (bootstrap) | p (Wilcoxon) | wins/losses |
|:---|---:|---:|---:|---:|
| + PageRank vs BM25F tuned | -0.028 | 0.0674 | 0.0935 | 19/27 |
| + personalised PageRank (user stack) vs BM25F tuned | -0.000 | 0.9777 | 0.9782 | 24/22 |
| + HITS authority vs BM25F tuned | -0.002 | 0.8062 | 0.4662 | 11/17 |
| + personalised PageRank (query cluster) vs BM25F tuned | +0.000 | 1.0000 | 1.0000 | 0/0 |
| + PageRank (beta=0.1 fixed) vs BM25F tuned | -0.012 | 0.0933 | 0.1687 | 13/21 |
| + personalised PageRank (user stack) (beta=0.1 fixed) vs BM25F tuned | -0.003 | 0.6896 | 0.5902 | 16/18 |
| + HITS authority (beta=0.1 fixed) vs BM25F tuned | -0.002 | 0.8062 | 0.4662 | 11/17 |
| + personalised PageRank (query cluster) (beta=0.1 fixed) vs BM25F tuned | +0.004 | 0.5459 | 0.6807 | 18/13 |

### By query type

| comparison (nDCG@10) | delta | p (bootstrap) | p (Wilcoxon) | wins/losses |
|:---|---:|---:|---:|---:|
| [api] + PageRank vs BM25F tuned | +0.001 | 0.9669 | 0.6051 | 9/7 |
| [api] + personalised PageRank (user stack) vs BM25F tuned | +0.017 | 0.5212 | 0.3066 | 10/5 |
| [api] + HITS authority vs BM25F tuned | -0.015 | 0.2207 | 0.3452 | 2/3 |
| [api] + personalised PageRank (query cluster) vs BM25F tuned | +0.000 | 1.0000 | 1.0000 | 0/0 |
| [api] + PageRank (beta=0.1 fixed) vs BM25F tuned | -0.008 | 0.5415 | 0.7794 | 4/4 |
| [api] + personalised PageRank (user stack) (beta=0.1 fixed) vs BM25F tuned | -0.007 | 0.5129 | 0.6784 | 5/4 |
| [api] + HITS authority (beta=0.1 fixed) vs BM25F tuned | -0.015 | 0.2207 | 0.3452 | 2/3 |
| [api] + personalised PageRank (query cluster) (beta=0.1 fixed) vs BM25F tuned | -0.004 | 0.5777 | 0.8590 | 5/4 |
| [howto] + PageRank vs BM25F tuned | -0.051 | **0.0214** | 0.0486 | 6/13 |
| [howto] + personalised PageRank (user stack) vs BM25F tuned | -0.007 | 0.7416 | 0.9405 | 10/10 |
| [howto] + HITS authority vs BM25F tuned | +0.017 | 0.3199 | 0.4631 | 7/6 |
| [howto] + personalised PageRank (query cluster) vs BM25F tuned | +0.000 | 1.0000 | 1.0000 | 0/0 |
| [howto] + PageRank (beta=0.1 fixed) vs BM25F tuned | -0.017 | 0.1674 | 0.2553 | 5/11 |
| [howto] + personalised PageRank (user stack) (beta=0.1 fixed) vs BM25F tuned | -0.000 | 0.9913 | 0.9250 | 6/8 |
| [howto] + HITS authority (beta=0.1 fixed) vs BM25F tuned | +0.017 | 0.3199 | 0.4631 | 7/6 |
| [howto] + personalised PageRank (query cluster) (beta=0.1 fixed) vs BM25F tuned | +0.027 | **0.0435** | 0.0392 | 11/2 |
| [cross] + PageRank vs BM25F tuned | -0.034 | 0.2099 | 0.2477 | 4/7 |
| [cross] + personalised PageRank (user stack) vs BM25F tuned | -0.011 | 0.6707 | 0.5337 | 4/7 |
| [cross] + HITS authority vs BM25F tuned | -0.010 | 0.6142 | 0.2411 | 2/8 |
| [cross] + personalised PageRank (query cluster) vs BM25F tuned | +0.000 | 1.0000 | 1.0000 | 0/0 |
| [cross] + PageRank (beta=0.1 fixed) vs BM25F tuned | -0.010 | 0.3588 | 0.3329 | 4/6 |
| [cross] + personalised PageRank (user stack) (beta=0.1 fixed) vs BM25F tuned | -0.001 | 0.9229 | 0.7897 | 5/6 |
| [cross] + HITS authority (beta=0.1 fixed) vs BM25F tuned | -0.010 | 0.6142 | 0.2411 | 2/8 |
| [cross] + personalised PageRank (query cluster) (beta=0.1 fixed) vs BM25F tuned | -0.013 | 0.1491 | 0.1097 | 2/7 |

## PageRank implementation check

* Against `networkx.pagerank` on the same graph: max |difference| = 8.92e-10 (global), 3.41e-09 (personalised to FastAPI).
* Converged in 93 iterations (final L1 change 8.8e-09); plot: `pagerank_convergence.png`.

## Graph: content links vs navigation links

* 211 pages, **493 content edges**, 19709 navigation edges; 63 pages have no content out-links.
* Gini coefficient of PageRank: content graph 0.532 vs nav graph 0.349 (lower = flatter). Navigation links make every page look alike.

Top pages by content-graph PageRank:

1. https://flask.palletsprojects.com/en/stable/api/ (0.0878)
2. https://fastapi.tiangolo.com/advanced/behind-a-proxy/ (0.0440)
3. https://flask.palletsprojects.com/en/stable/deploying/ (0.0265)
4. https://fastapi.tiangolo.com/advanced/sub-applications/ (0.0245)
5. https://fastapi.tiangolo.com/deployment/https/ (0.0216)
6. https://flask.palletsprojects.com/en/stable/reqcontext/ (0.0182)
7. https://flask.palletsprojects.com/en/stable/cli/ (0.0176)
8. https://flask.palletsprojects.com/en/stable/deploying/proxy_fix/ (0.0165)
9. https://flask.palletsprojects.com/en/stable/config/ (0.0154)
10. https://flask.palletsprojects.com/en/stable/appcontext/ (0.0149)

## HITS example: query `dependencies` (base set 128 pages)

Authorities:
* https://fastapi.tiangolo.com/python-types/ (0.2629)
* https://fastapi.tiangolo.com/tutorial/bigger-applications/ (0.2292)
* https://fastapi.tiangolo.com/tutorial/response-model/ (0.2281)
* https://fastapi.tiangolo.com/advanced/additional-responses/ (0.2236)
* https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/ (0.2221)

Hubs:
* https://fastapi.tiangolo.com/reference/fastapi/ (0.7852)
* https://fastapi.tiangolo.com/reference/apirouter/ (0.5701)
* https://fastapi.tiangolo.com/reference/parameters/ (0.072)
* https://fastapi.tiangolo.com/advanced/custom-response/ (0.0679)
* https://fastapi.tiangolo.com/advanced/response-directly/ (0.0663)
