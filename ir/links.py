"""
Link analysis over the documentation link graph.

Graph: one node per crawled page; an edge p -> q for every *content* link
(links inside the main article only; sidebar/nav/footer links are excluded at
extraction time because they connect every page to every page and flatten
the scores). `nav_graph()` builds the nav-link graph for comparison.

PageRank      r = d * M r + d * (dangling mass) / N + (1 - d) * v
              d = 0.85, dangling pages spread their rank uniformly,
              iterate until ||r_new - r||_1 < 1e-8. v = uniform (global) or a
              personalised teleport vector (user's stack, query's cluster).
HITS          query-time: root set = top-50 BM25 pages, base set = root plus
              in- and out-neighbours; a = A^T h, h = A a, L2-normalised.
"""
from __future__ import annotations

import numpy as np


class LinkGraph:
    def __init__(self, pages: list[dict], use_nav: bool = False):
        self.urls = [p["url"] for p in pages]
        self.idx = {u: i for i, u in enumerate(self.urls)}
        self.framework = [p["framework"] for p in pages]
        self.out: list[list[int]] = []
        for p in pages:
            links = p["nav_links"] if use_nav else p["outlinks"]
            tgt = sorted({self.idx[l.split("#")[0]] for l in links
                          if l.split("#")[0] in self.idx and l.split("#")[0] != p["url"]})
            self.out.append(tgt)
        self.inn: list[list[int]] = [[] for _ in pages]
        for s, ts in enumerate(self.out):
            for t in ts:
                self.inn[t].append(s)
        self.N = len(pages)

    @property
    def edges(self) -> int:
        return sum(map(len, self.out))

    def pagerank(self, d: float = 0.85, tol: float = 1e-8, max_iter: int = 1000,
                 teleport: np.ndarray | None = None) -> tuple[np.ndarray, list[float]]:
        N = self.N
        v = np.full(N, 1.0 / N) if teleport is None else teleport / teleport.sum()
        outdeg = np.array([len(o) for o in self.out], float)
        dangling = outdeg == 0
        src = np.array([s for s, ts in enumerate(self.out) for _ in ts], int)
        dst = np.array([t for ts in self.out for t in ts], int)
        r = np.full(N, 1.0 / N)
        deltas = []
        for _ in range(max_iter):
            contrib = np.zeros(N)
            np.add.at(contrib, dst, r[src] / outdeg[src])
            new = d * contrib + d * r[dangling].sum() / N + (1 - d) * v
            new /= new.sum()
            delta = np.abs(new - r).sum()
            deltas.append(float(delta))
            r = new
            if delta < tol:
                break
        return r, deltas

    def teleport_for(self, frameworks: set[str] | None = None, pages: set[int] | None = None,
                     mix: float = 0.0) -> np.ndarray:
        """Teleport mass on a page subset (optionally mixed with uniform)."""
        v = np.zeros(self.N)
        for i in range(self.N):
            if (frameworks and self.framework[i] in frameworks) or (pages and i in pages):
                v[i] = 1.0
        if v.sum() == 0:
            v[:] = 1.0
        v /= v.sum()
        return (1 - mix) * v + mix / self.N

    def hits(self, root: list[int], max_in: int = 50, iters: int = 100, tol: float = 1e-8):
        base = set(root)
        for p in root:
            base.update(self.out[p])
            base.update(self.inn[p][:max_in])
        base = sorted(base)
        pos = {p: i for i, p in enumerate(base)}
        n = len(base)
        A = np.zeros((n, n))
        for p in base:
            for q in self.out[p]:
                if q in pos:
                    A[pos[p], pos[q]] = 1.0
        h = np.ones(n) / np.sqrt(n)
        a = h.copy()
        for _ in range(iters):
            a_new = A.T @ h
            a_new /= np.linalg.norm(a_new) or 1.0
            h_new = A @ a_new
            h_new /= np.linalg.norm(h_new) or 1.0
            done = np.abs(a_new - a).sum() + np.abs(h_new - h).sum() < tol
            a, h = a_new, h_new
            if done:
                break
        return {p: float(a[pos[p]]) for p in base}, {p: float(h[pos[p]]) for p in base}

    def to_networkx(self):
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(range(self.N))
        G.add_edges_from((s, t) for s, ts in enumerate(self.out) for t in ts)
        return G
