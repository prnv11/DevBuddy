"""
One object that loads the corpus and exposes every retrieval system by name.

    eng = Engine("data")
    eng.run("bm25f", "framework:flask how do blueprints work", k=10)

Systems
    bm25            BM25 over all zones as one field
    bm25f           BM25F with zone weights (title, heading, body, code)
    lsa | minilm    dense cosine retrieval
    hybrid          RRF(bm25f, dense)             dense = minilm if available else lsa
    hybrid_lsa      RRF(bm25f, lsa)
    bm25f+pr        bm25f re-ranked with global PageRank
    bm25f+ppr       ... with PageRank personalised to the query's framework (user's stack)
    bm25f+pprc      ... with PageRank personalised to the query's concept cluster (part 2)
    bm25f+hits      ... with query-time HITS authority
    bm25f+prf       pseudo-relevance feedback (Rocchio on the top 5, only when confident)
    align           cross-framework concept alignment (part 2); see ir/cluster.py
"""
from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path

import numpy as np

from ir.dense import LSARetriever, MiniLMRetriever, minilm_available, rrf
from ir.index import DEFAULT_ZONE_WEIGHTS, FILTER_RE, InvertedIndex, parse_query
from ir.links import LinkGraph

DEFAULT_PARAMS = {
    "k1": 1.2, "b": 0.75, "weights": dict(DEFAULT_ZONE_WEIGHTS),
    "rrf_w": 0.5,               # weight of the lexical run in RRF
    "beta_pr": 0.1, "beta_ppr": 0.1, "beta_pprc": 0.1, "beta_hits": 0.1,
    "rerank_depth": 100,
    "prf_docs": 5, "prf_terms": 25, "prf_conf": 0.0,
}


def load_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class Engine:
    def __init__(self, data_dir: str | Path = "data", use_minilm: bool | None = None,
                 params: dict | None = None):
        self.dir = Path(data_dir)
        self.chunks = load_jsonl(self.dir / "chunks.jsonl")
        self.pages = load_jsonl(self.dir / "pages.jsonl")
        ipath = self.dir / "index.pkl"
        self.index = InvertedIndex.load(ipath) if ipath.exists() else InvertedIndex.build(self.chunks)
        if self.index.N != len(self.chunks) or self.index.ids[:1] != [self.chunks[0]["id"]]:
            self.index = InvertedIndex.build(self.chunks)
        if not ipath.exists() or self.index.N != len(self.chunks):
            self.index.save(ipath)
        self.id2idx = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.use_minilm = minilm_available() if use_minilm is None else use_minilm
        self.graph = LinkGraph(self.pages)
        self.page_of = np.array([self.graph.idx.get(c["page_url"], -1) for c in self.chunks])
        self.clusters = None        # set by ir.cluster.ConceptAligner.attach()

    # ---------------- lazily built components ----------------
    @cached_property
    def lsa(self) -> LSARetriever:
        return LSARetriever(self.chunks)

    @cached_property
    def minilm(self) -> MiniLMRetriever:
        return MiniLMRetriever(self.chunks, cache=self.dir / "emb_minilm.npy")

    @property
    def dense(self):
        return self.minilm if self.use_minilm else self.lsa

    @cached_property
    def pagerank(self) -> np.ndarray:
        r, self.pr_deltas = self.graph.pagerank()
        return r

    def _ppr(self, frameworks: frozenset) -> np.ndarray:
        cache = self.__dict__.setdefault("_ppr_cache", {})
        if frameworks not in cache:
            cache[frameworks] = self.graph.pagerank(teleport=self.graph.teleport_for(set(frameworks)))[0]
        return cache[frameworks]

    # ---------------- helpers ----------------
    @staticmethod
    def strip_filters(q: str) -> str:
        q = FILTER_RE.sub(lambda m: m.group(2) if m.group(1) not in ("framework", "section") else " ", q)
        return re.sub(r"(?<!\S)[+-](?=\S)", "", q).strip()

    def allowed(self, q: str) -> set[int] | None:
        return self.index.candidates(parse_query(q))

    @staticmethod
    def framework_of(q: str) -> str | None:
        m = re.search(r"framework:(\w+)", q)
        return m.group(1).lower() if m else None

    @staticmethod
    def _logscale(x: np.ndarray) -> np.ndarray:
        lx = np.log(np.clip(x, 1e-12, None))
        lo, hi = lx.min(), lx.max()
        return (lx - lo) / (hi - lo) if hi > lo else np.zeros_like(lx)

    def _rerank(self, base: list[tuple[int, float]], page_score: np.ndarray, beta: float, k: int):
        if not base:
            return []
        top = base[0][1] or 1.0
        g = self._logscale(page_score)
        out = []
        for d, s in base:
            p = self.page_of[d]
            out.append((d, s / top + beta * (g[p] if p >= 0 else 0.0)))
        return sorted(out, key=lambda x: -x[1])[:k]

    def hits_scores(self, q: str, root_k: int = 50) -> np.ndarray:
        base = self.run("bm25", q, k=400)
        pages = []
        for d, _ in base:
            p = int(self.page_of[d])
            if p >= 0 and p not in pages:
                pages.append(p)
            if len(pages) >= root_k:
                break
        auth, _ = self.graph.hits(pages)
        a = np.full(self.graph.N, 1e-6)
        for p, v in auth.items():
            a[p] = max(v, 1e-6)
        return a

    # ---------------- main entry point ----------------
    def run(self, system: str, q: str, k: int = 10, **override) -> list[tuple[int, float]]:
        P = {**self.params, **override}
        ix = self.index
        if system == "bm25":
            return ix.bm25(q, k=k, k1=P["k1"], b=P["b"])
        if system == "bm25f":
            return ix.bm25f(q, k=k, k1=P["k1"], b=P["b"], weights=P["weights"])
        if system in ("lsa", "minilm", "dense"):
            r = getattr(self, system)
            return r.search(self.strip_filters(q), k=k, allowed=self.allowed(q))
        if system in ("hybrid", "hybrid_lsa", "hybrid_minilm"):
            dense = {"hybrid": "dense", "hybrid_lsa": "lsa", "hybrid_minilm": "minilm"}[system]
            lex = self.run("bm25f", q, k=100, **override)
            den = self.run(dense, q, k=100, **override)
            return rrf([lex, den], [P["rrf_w"], 1 - P["rrf_w"]], k=k)
        depth = P["rerank_depth"]
        if system == "bm25f+pr":
            return self._rerank(self.run("bm25f", q, k=depth, **override), self.pagerank, P["beta_pr"], k)
        if system == "bm25f+ppr":
            fw = override.get("stack") or self.framework_of(q)
            pr = self._ppr(frozenset([fw])) if fw else self.pagerank
            return self._rerank(self.run("bm25f", q, k=depth, **override), pr, P["beta_ppr"], k)
        if system == "bm25f+pprc":
            if self.clusters is None:
                raise RuntimeError("attach a ConceptAligner first (ir.cluster)")
            pr = self.clusters.cluster_pagerank(self, q)
            return self._rerank(self.run("bm25f", q, k=depth, **override), pr, P["beta_pprc"], k)
        if system == "bm25f+hits":
            return self._rerank(self.run("bm25f", q, k=depth, **override), self.hits_scores(q),
                                P["beta_hits"], k)
        if system == "bm25f+prf":
            from ir.feedback import pseudo_relevance_feedback
            return pseudo_relevance_feedback(self, q, k=k, n_docs=P["prf_docs"], n_terms=P["prf_terms"],
                                             min_conf=P["prf_conf"])
        raise ValueError(f"unknown system {system!r}")

    def systems(self) -> list[str]:
        s = ["bm25", "bm25f", "lsa", "hybrid_lsa"]
        if self.use_minilm:
            s += ["minilm", "hybrid_minilm"]
        return s

    def show(self, results, n: int = 10):
        for r, (d, s) in enumerate(results[:n], 1):
            c = self.chunks[d]
            print(f"{r:2d}. {s:7.3f}  {c['id']}")
