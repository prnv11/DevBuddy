"""
Dense retrievers: all chunks embedded in one vector space, cosine similarity.

  * MiniLMRetriever - sentence-transformers all-MiniLM-L6-v2 (384-d), the model
    DevBuddy already used. Embeddings are cached in data/emb_minilm.npy.
    Needs `pip install sentence-transformers` and the model download from
    Hugging Face on first use.
  * LSARetriever    - tf-idf (code-aware tokenizer) reduced with truncated SVD.
    Fully local, no model download; also the "shared space" for clustering.

Both expose .vectors (N x d, L2-normalised), .encode_query(text) and .search().
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ir.index import doc_zones
from ir.text import terms


def passage(c: dict) -> str:
    z = doc_zones(c)
    head = " > ".join(c.get("heading_path") or [])
    return f"{head}. {z['body']} {z['code'][:600]}"


class _Base:
    vectors: np.ndarray

    def encode_query(self, q: str) -> np.ndarray:
        raise NotImplementedError

    def search(self, q: str | np.ndarray, k: int = 10, allowed: set[int] | None = None):
        qv = self.encode_query(q) if isinstance(q, str) else q
        s = self.vectors @ qv
        if allowed is not None:
            mask = np.full(len(s), -np.inf)
            idx = np.fromiter(allowed, int)
            mask[idx] = s[idx]
            s = mask
        k = min(k, int(np.isfinite(s).sum()))
        top = np.argpartition(-s, k - 1)[:k] if k else []
        top = sorted(top, key=lambda i: -s[i])
        return [(int(i), float(s[i])) for i in top]


class LSARetriever(_Base):
    name = "lsa"

    def __init__(self, chunks: list[dict], dims: int = 200, seed: int = 0):
        self.tfidf = TfidfVectorizer(analyzer=terms, sublinear_tf=True, min_df=2, max_df=0.5)
        X = self.tfidf.fit_transform([passage(c) for c in chunks])
        self.svd = TruncatedSVD(n_components=min(dims, X.shape[1] - 1), random_state=seed)
        self.vectors = normalize(self.svd.fit_transform(X)).astype(np.float32)
        self.X = X                                       # sparse tf-idf, reused by clustering

    def encode_query(self, q: str) -> np.ndarray:
        v = self.svd.transform(self.tfidf.transform([q]))
        return normalize(v)[0].astype(np.float32)


class MiniLMRetriever(_Base):
    name = "minilm"
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, chunks: list[dict], cache: Path | None = None):
        from sentence_transformers import SentenceTransformer   # optional dependency
        self.model = SentenceTransformer(self.MODEL)
        texts = [passage(c) for c in chunks]
        # key on ids AND text, so re-extracted chunks (same ids, new text) are re-embedded
        key = hashlib.sha1("\n".join(c["id"] + "\t" + t for c, t in zip(chunks, texts)).encode()).hexdigest()[:12]
        if cache and cache.exists() and cache.with_suffix(".key").exists() \
                and cache.with_suffix(".key").read_text() == key:
            self.vectors = np.load(cache)
        else:
            self.vectors = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                             show_progress_bar=True).astype(np.float32)
            if cache:
                np.save(cache, self.vectors)
                cache.with_suffix(".key").write_text(key)

    def encode_query(self, q: str) -> np.ndarray:
        return self.model.encode([q], normalize_embeddings=True)[0].astype(np.float32)


def minilm_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def rrf(runs: list[list[tuple[int, float]]], weights: list[float] | None = None, k: int = 10,
        c: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum_s w_s / (c + rank_s(d))."""
    weights = weights or [1.0] * len(runs)
    acc: dict[int, float] = {}
    for w, run in zip(weights, runs):
        for r, (d, _) in enumerate(run, start=1):
            acc[d] = acc.get(d, 0.0) + w / (c + r)
    return sorted(acc.items(), key=lambda x: -x[1])[:k]
