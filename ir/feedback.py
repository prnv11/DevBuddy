"""
Part 4: Rocchio relevance feedback and pseudo-relevance feedback.

    q' = alpha * q0 + beta * mean(relevant) - gamma * mean(non-relevant)
    alpha = 1, beta = 0.75, gamma = 0.15

Lexical side: vectors are tf-idf term weights (L2-normalised per document);
negative weights are clipped to zero and only the original terms plus the
top `n_terms` expansion terms are kept. The new query is scored with
weighted BM25F. Dense side: the same formula on the embedding vectors,
then renormalised. Hybrid: RRF of the two.

Pseudo-relevance feedback assumes the top `n_docs` are relevant and expands
automatically, but only when the ranking is confident (top-1 score clearly
above the top-10 score), to avoid query drift on vague queries.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ir.dense import rrf
from ir.index import parse_query

ALPHA, BETA, GAMMA = 1.0, 0.75, 0.15


def _unit(v: dict[str, float]) -> dict[str, float]:
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / n for t, x in v.items()}


def query_vector(eng, q: str) -> dict[str, float]:
    pq = parse_query(q)
    tf = defaultdict(int)
    for t, _ in pq.terms:
        tf[t] += 1
    return _unit({t: (1 + math.log(c)) * eng.index.idf(t) for t, c in tf.items()})


def rocchio_terms(eng, q: str, rel: list[int], nonrel: list[int], n_terms: int = 25,
                  alpha=ALPHA, beta=BETA, gamma=GAMMA) -> dict[str, float]:
    q0 = query_vector(eng, q)
    new = defaultdict(float)
    for t, w in q0.items():
        new[t] += alpha * w
    for docs, coef in ((rel, beta), (nonrel, -gamma)):
        if not docs:
            continue
        for d in docs:
            for t, w in _unit(eng.index.term_vector(d)).items():
                new[t] += coef * w / len(docs)
    new = {t: w for t, w in new.items() if w > 0}                 # clip negatives
    expansion = sorted((t for t in new if t not in q0), key=lambda t: -new[t])[:n_terms]
    keep = set(q0) | set(expansion)
    return {t: new[t] for t in keep if t in new}


def rocchio_dense(eng, q: str, rel: list[int], nonrel: list[int], dense=None,
                  alpha=ALPHA, beta=BETA, gamma=GAMMA) -> np.ndarray:
    dense = dense or eng.dense
    v = alpha * dense.encode_query(eng.strip_filters(q))
    if rel:
        v = v + beta * dense.vectors[rel].mean(0)
    if nonrel:
        v = v - gamma * dense.vectors[nonrel].mean(0)
    return v / (np.linalg.norm(v) or 1.0)


def feedback_search(eng, q: str, rel: list[int], nonrel: list[int], system: str = "bm25f",
                    k: int = 10, n_terms: int = 25, exclude: set[int] | None = None):
    """Re-rank after feedback. `exclude` removes already-judged docs (residual collection)."""
    allowed = eng.allowed(q)
    if exclude:
        base = allowed if allowed is not None else set(range(len(eng.chunks)))
        allowed = base - exclude
    P = eng.params
    out = {}
    if system in ("bm25f", "hybrid", "hybrid_lsa", "hybrid_minilm"):
        qw = rocchio_terms(eng, q, rel, nonrel, n_terms=n_terms)
        out["lex"] = eng.index.bm25f_weighted(qw, k=max(k, 100), k1=P["k1"], b=P["b"], weights=P["weights"],
                                              allowed=allowed)
    if system in ("lsa", "minilm", "dense", "hybrid", "hybrid_lsa", "hybrid_minilm"):
        dn = {"hybrid": "dense", "hybrid_lsa": "lsa", "hybrid_minilm": "minilm"}.get(system, system)
        dense = getattr(eng, dn)
        v = rocchio_dense(eng, q, rel, nonrel, dense=dense)
        out["dense"] = dense.search(v, k=max(k, 100), allowed=allowed)
    if len(out) == 2:
        return rrf([out["lex"], out["dense"]], [P["rrf_w"], 1 - P["rrf_w"]], k=k)
    return next(iter(out.values()))[:k]


def confidence(run: list[tuple[int, float]]) -> float:
    """Peakedness of a ranking: top-1 score relative to the 10th score."""
    if len(run) < 2:
        return 0.0
    s1, s10 = run[0][1], run[min(9, len(run) - 1)][1]
    return (s1 - s10) / (abs(s1) or 1.0)


def pseudo_relevance_feedback(eng, q: str, k: int = 10, n_docs: int = 5, n_terms: int = 25,
                              min_conf: float = 0.0, system: str = "bm25f"):
    first = eng.run(system, q, k=max(k, 100))
    if confidence(first) < min_conf:
        return first[:k]                 # not confident: keep the original ranking
    rel = [d for d, _ in first[:n_docs]]
    return feedback_search(eng, q, rel, [], system=system, k=k, n_terms=n_terms)
