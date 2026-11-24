"""
Part 2: clustering for cross-framework concept alignment.

Pipeline
  1. Shared vector space for both frameworks: tf-idf, LSA (tf-idf + SVD) or
     MiniLM. One space is needed because Flask and FastAPI use different
     vocabulary for the same concept.
  2. Flat clustering per framework: spherical k-means (cosine), k chosen in
     [30, 60] by silhouette; or HAC (average linkage, cosine) cut at k.
     Clusters are labelled with the heaviest tf-idf terms of their members.
  3. Hierarchy: HAC over the cluster centroids of *both* frameworks gives a
     dendrogram (routing -> path params / blueprints / ...).
  4. Alignment: each source cluster is mapped to the target cluster with the
     highest centroid cosine (optionally one-to-one with the Hungarian
     algorithm). Below a similarity threshold tau the concept is "no match".
  5. Query time: best source chunk by BM25F -> its cluster -> matched target
     cluster(s) -> rank only the target chunks inside them (cluster pruning)
     by RRF of (a) cosine to the source evidence vector and (b) BM25F of the
     query restricted to the cluster.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from ir.dense import rrf


def spherical_kmeans(X: np.ndarray, k: int, n_init: int = 5, iters: int = 50, seed: int = 0):
    """k-means on the unit sphere: assign by max cosine, centroid = normalised mean."""
    rng = np.random.default_rng(seed)
    best = (None, None, -np.inf)
    for _ in range(n_init):
        # k-means++ style seeding with cosine distance
        C = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d = 1 - np.max(X @ np.array(C).T, axis=1)
            d = np.clip(d, 0, None) ** 2
            C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
        C = np.array(C)
        labels = None
        for _ in range(iters):
            new = np.argmax(X @ C.T, axis=1)
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                m = X[labels == j]
                C[j] = m.sum(0) if len(m) else X[rng.integers(len(X))]
            C = normalize(C)
        obj = float(np.sum(np.max(X @ C.T, axis=1)))
        if obj > best[2]:
            best = (labels.copy(), C.copy(), obj)
    return best[0], best[1]


def hac(X: np.ndarray, k: int):
    Z = linkage(X, method="average", metric="cosine")
    labels = fcluster(Z, t=k, criterion="maxclust") - 1
    C = normalize(np.array([X[labels == j].mean(0) for j in range(labels.max() + 1)]))
    return labels, C


@dataclass
class AlignResult:
    source_chunk: int | None
    source_cluster: int | None
    target_clusters: list[int]
    similarity: float
    no_match: bool
    ranked: list[tuple[int, float]]
    candidates_scored: int


class ConceptAligner:
    FRAMEWORKS = ("flask", "fastapi")

    def __init__(self, eng, space: str = "lsa", method: str = "kmeans", k: int | dict | None = None,
                 k_range=(30, 35, 40, 45, 50, 55, 60), seed: int = 0):
        self.eng, self.space, self.method, self.seed = eng, space, method, seed
        self.V = self._vectors(space)
        fw = np.array([c["framework"] for c in eng.chunks])
        self.members_fw = {f: np.where(fw == f)[0] for f in self.FRAMEWORKS}
        self.labels = np.full(len(eng.chunks), -1)
        self.cluster_fw: list[str] = []
        cents, self.k_chosen, self.silhouette = [], {}, {}
        for f in self.FRAMEWORKS:
            idx = self.members_fw[f]
            X = self.V[idx]
            kk = (k.get(f) if isinstance(k, dict) else k)
            if kk is None:
                kk, sil = self._choose_k(X, k_range)
            else:
                lab, _ = self._cluster(X, kk)
                sil = float(silhouette_score(X, lab, metric="cosine")) if len(set(lab)) > 1 else 0.0
            lab, C = self._cluster(X, kk)
            off = len(self.cluster_fw)
            self.labels[idx] = lab + off
            self.cluster_fw += [f] * len(C)
            cents.append(C)
            self.k_chosen[f], self.silhouette[f] = kk, sil
        self.C = np.vstack(cents)
        self.cluster_fw = np.array(self.cluster_fw)
        self.members = {c: np.where(self.labels == c)[0] for c in range(len(self.C))}
        self._label_terms = None
        self._pr_cache: dict[int, np.ndarray] = {}

    # ---------------- building blocks ----------------
    def _vectors(self, space: str) -> np.ndarray:
        if space == "tfidf":
            return normalize(self.eng.lsa.X).toarray().astype(np.float32)
        if space == "lsa":
            return self.eng.lsa.vectors
        if space == "minilm":
            return self.eng.minilm.vectors
        raise ValueError(space)

    def _cluster(self, X, k):
        return spherical_kmeans(X, k, seed=self.seed) if self.method == "kmeans" else hac(X, k)

    def _choose_k(self, X, k_range):
        best = (None, -np.inf)
        for k in k_range:
            if k >= len(X):
                continue
            lab, _ = self._cluster(X, k)
            s = silhouette_score(X, lab, metric="cosine")
            if s > best[1]:
                best = (k, float(s))
        return best

    def cluster_terms(self, c: int, n: int = 5) -> list[str]:
        if self._label_terms is None:
            X = self.eng.lsa.X
            vocab = np.array(self.eng.lsa.tfidf.get_feature_names_out())
            self._label_terms = {}
            for cc, m in self.members.items():
                w = np.asarray(X[m].mean(0)).ravel()
                self._label_terms[cc] = list(vocab[np.argsort(-w)[:8]])
        return self._label_terms[c][:n]

    def similarity_matrix(self, src: str, tgt: str):
        s = np.where(self.cluster_fw == src)[0]
        t = np.where(self.cluster_fw == tgt)[0]
        return s, t, self.C[s] @ self.C[t].T

    def cluster_map(self, src: str = "flask", tgt: str = "fastapi", one_to_one: bool = False):
        s, t, S = self.similarity_matrix(src, tgt)
        if one_to_one:
            r, c = linear_sum_assignment(-S)
            m = {int(s[i]): (int(t[j]), float(S[i, j])) for i, j in zip(r, c)}
            for i in range(len(s)):          # unmatched rows (more src than tgt clusters)
                if int(s[i]) not in m:
                    j = int(np.argmax(S[i]))
                    m[int(s[i])] = (int(t[j]), float(S[i, j]))
            return m
        return {int(s[i]): (int(t[np.argmax(S[i])]), float(S[i].max())) for i in range(len(s))}

    # ---------------- query time ----------------
    def align(self, query: str, src: str = "flask", tgt: str = "fastapi", m: int = 1, w: float = 0.5,
              tau: float = 0.0, restrict: bool = True, one_to_one: bool = False, k: int = 10,
              evidence: int = 3, route: str = "centroid") -> AlignResult:
        """route='centroid': source cluster -> nearest target centroids (cluster map);
        route='evidence': target clusters nearest to the source evidence vector."""
        eng = self.eng
        q = eng.strip_filters(query)
        src_hits = eng.run("bm25f", f"framework:{src} {q}", k=evidence)
        if not src_hits:
            return AlignResult(None, None, [], 0.0, True, [], 0)
        top = src_hits[0][0]
        c_src = int(self.labels[top])
        e = normalize(self.V[[d for d, _ in src_hits]].mean(0, keepdims=True))[0]
        s, t, S = self.similarity_matrix(src, tgt)
        row = S[list(s).index(c_src)] if route == "centroid" else self.C[t] @ e
        if one_to_one and route == "centroid":
            first = self.cluster_map(src, tgt, one_to_one=True)[c_src][0]
            order = [first] + [int(t[j]) for j in np.argsort(-row) if int(t[j]) != first]
        else:
            order = [int(t[j]) for j in np.argsort(-row)]
        targets = order[:m]
        sim = float(self.C[c_src] @ self.C[targets[0]]) if route == "centroid" else float(self.C[targets[0]] @ e)
        if restrict:
            cand = np.concatenate([self.members[c] for c in targets])
        else:
            cand = self.members_fw[tgt]
        dense_scores = self.V[cand] @ e
        dense_run = sorted(zip(cand.tolist(), dense_scores.tolist()), key=lambda x: -x[1])[:100]
        lex_run = eng.index.bm25f(q, k=100, k1=eng.params["k1"], b=eng.params["b"],
                                  weights=eng.params["weights"], allowed=set(cand.tolist()))
        ranked = rrf([dense_run, lex_run], [w, 1 - w], k=k) if lex_run else dense_run[:k]
        return AlignResult(top, c_src, targets, sim, sim < tau, ranked, len(cand))

    def cluster_pagerank(self, eng, q: str) -> np.ndarray:
        """Personalised PageRank teleporting to the pages of the query's cluster."""
        hits = eng.run("bm25f", q, k=1)
        if not hits:
            return eng.pagerank
        c = int(self.labels[hits[0][0]])
        if c not in self._pr_cache:
            pages = {int(eng.page_of[d]) for d in self.members[c] if eng.page_of[d] >= 0}
            self._pr_cache[c] = eng.graph.pagerank(teleport=eng.graph.teleport_for(pages=pages, mix=0.1))[0]
        return self._pr_cache[c]

    # ---------------- integration ----------------
    def attach(self, eng):
        eng.clusters = self
        return self

    @classmethod
    def load_or_build(cls, eng, **kw):
        return cls(eng, **kw)

    def describe(self, c: int) -> str:
        pages = Counter(self.eng.chunks[d]["page_title"] for d in self.members[c]).most_common(2)
        return f"{self.cluster_fw[c]}#{c} [{', '.join(self.cluster_terms(c, 4))}] ({len(self.members[c])} chunks; {pages[0][0]})"
