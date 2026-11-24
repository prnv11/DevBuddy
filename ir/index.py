"""
Zoned positional inverted index with BM25 and BM25F ranking.

index.postings[zone][term] -> list of (doc, tf, positions)   (doc ids ascending)
index.length[zone][doc]    -> number of tokens of doc in zone
index.df[term]             -> number of docs containing term in any zone

Zones: title (page title), heading (heading path below the title),
body (prose), code (<pre> blocks).

Scoring is term-at-a-time: one pass over each query term's postings adds
into a score accumulator per candidate doc; the top K are taken with a heap.
Boolean and parametric constraints are applied *before* scoring by
restricting the candidate set:

    framework:flask       only Flask chunks
    section:tutorial      URL path contains /tutorial/
    +term                 doc must contain term (any zone)
    -term                 doc must not contain term
    code:jsonify          term only matched in the code zone (also title:, heading:, body:)
"""
from __future__ import annotations

import heapq
import math
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ir.text import analyze, terms

ZONES = ("title", "heading", "body", "code")
DEFAULT_ZONE_WEIGHTS = {"title": 2.0, "heading": 2.5, "body": 1.0, "code": 1.2}


def doc_zones(c: dict) -> dict[str, str]:
    hp = c.get("heading_path") or []
    return {
        "title": c.get("page_title", ""),
        "heading": " ".join(hp[1:]) if len(hp) > 1 else "",
        "body": c.get("text", ""),
        "code": c.get("code", ""),
    }


@dataclass
class Query:
    terms: list[tuple[str, str | None]] = field(default_factory=list)   # (term, zone or None)
    must: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)
    text: str = ""


FILTER_RE = re.compile(r"(?<!\S)(framework|section|zone|title|heading|body|code):(\S+)")


def parse_query(q: str) -> Query:
    out = Query()
    rest = q
    for m in FILTER_RE.finditer(q):
        key, val = m.group(1), m.group(2)
        if key in ("framework", "section"):
            out.filters[key] = val.lower()
        else:
            for t in terms(val):
                out.terms.append((t, key))
        rest = rest.replace(m.group(0), " ")
    for tok in rest.split():
        if tok.startswith("+") and len(tok) > 1:
            out.must += terms(tok[1:])
            out.terms += [(t, None) for t in terms(tok[1:])]
        elif tok.startswith("-") and len(tok) > 1:
            out.must_not += terms(tok[1:])
        else:
            out.terms += [(t, None) for t in terms(tok)]
    out.text = " ".join(t for t, _ in out.terms)
    return out


class InvertedIndex:
    def __init__(self):
        self.docs: list[dict] = []           # chunk metadata (no big text duplicated)
        self.ids: list[str] = []
        self.postings = {z: defaultdict(list) for z in ZONES}
        self.length = {z: [] for z in ZONES}
        self.avglen = {z: 0.0 for z in ZONES}
        self.df: dict[str, int] = defaultdict(int)
        self.N = 0

    # ---------------- build ----------------
    @classmethod
    def build(cls, chunks: list[dict]) -> "InvertedIndex":
        ix = cls()
        for d, c in enumerate(chunks):
            ix.ids.append(c["id"])
            ix.docs.append({k: c[k] for k in ("id", "framework", "url", "page_url", "page_title", "heading_path")})
            seen_terms = set()
            for z, text in doc_zones(c).items():
                toks = analyze(text)
                ix.length[z].append(len({p for _, p in toks}))
                by_term: dict[str, list[int]] = defaultdict(list)
                for t, p in toks:
                    by_term[t].append(p)
                for t, ps in by_term.items():
                    ix.postings[z][t].append((d, len(ps), ps))
                    seen_terms.add(t)
            for t in seen_terms:
                ix.df[t] += 1
        ix.N = len(chunks)
        for z in ZONES:
            ix.avglen[z] = sum(ix.length[z]) / max(1, ix.N)
        ix.postings = {z: dict(p) for z, p in ix.postings.items()}
        ix.df = dict(ix.df)
        return ix

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.__getstate__(), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        ix = cls()
        with open(path, "rb") as f:
            ix.__dict__.update(pickle.load(f))
        return ix

    # ---------------- helpers ----------------
    def idf(self, t: str) -> float:
        df = self.df.get(t, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def docs_with(self, t: str) -> set[int]:
        s = set()
        for z in ZONES:
            s.update(d for d, _, _ in self.postings[z].get(t, ()))
        return s

    def candidates(self, q: Query) -> set[int] | None:
        """Boolean / parametric pre-filter. None means 'no restriction'."""
        cand: set[int] | None = None
        fw = q.filters.get("framework")
        if fw:
            cand = {i for i, d in enumerate(self.docs) if d["framework"] == fw}
        sec = q.filters.get("section")
        if sec:
            s = {i for i, d in enumerate(self.docs) if f"/{sec.strip('/')}/" in d["page_url"]}
            cand = s if cand is None else cand & s
        for t in q.must:
            s = self.docs_with(t)
            cand = s if cand is None else cand & s
        if q.must_not:
            excl = set().union(*(self.docs_with(t) for t in q.must_not))
            cand = (set(range(self.N)) if cand is None else cand) - excl
        return cand

    # ---------------- ranking ----------------
    def bm25(self, query: str | Query, k: int = 10, k1: float = 1.2, b: float = 0.75,
             allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """Plain BM25 over the union of all zones (one bag of words per chunk)."""
        q = parse_query(query) if isinstance(query, str) else query
        cand = self.candidates(q)
        if allowed is not None:
            cand = allowed if cand is None else cand & allowed
        if not hasattr(self, "_total_len"):
            self._total_len = [sum(self.length[z][d] for z in ZONES) for d in range(self.N)]
        total_len = self._total_len
        avg = sum(total_len) / max(1, self.N)
        acc: dict[int, float] = defaultdict(float)
        qtf: dict[tuple, int] = defaultdict(int)
        for t, z in q.terms:
            qtf[(t, z)] += 1
        for (t, zone), _ in qtf.items():
            tf_by_doc: dict[int, int] = defaultdict(int)
            for z in ([zone] if zone else ZONES):
                for d, tf, _ in self.postings[z].get(t, ()):
                    if cand is None or d in cand:
                        tf_by_doc[d] += tf
            idf = self.idf(t)
            for d, tf in tf_by_doc.items():
                norm = k1 * (1 - b + b * total_len[d] / avg)
                acc[d] += idf * tf * (k1 + 1) / (tf + norm)
        return heapq.nlargest(k, acc.items(), key=lambda x: x[1])

    def bm25f(self, query: str | Query, k: int = 10, k1: float = 1.2, b: float = 0.75,
              weights: dict[str, float] | None = None, allowed: set[int] | None = None
              ) -> list[tuple[int, float]]:
        """BM25F: per-zone length-normalised tf, weighted, then one saturation."""
        w = weights or DEFAULT_ZONE_WEIGHTS
        q = parse_query(query) if isinstance(query, str) else query
        cand = self.candidates(q)
        if allowed is not None:
            cand = allowed if cand is None else cand & allowed
        acc: dict[int, float] = defaultdict(float)
        for t, zone in dict.fromkeys(q.terms):
            wtf: dict[int, float] = defaultdict(float)
            for z in ([zone] if zone else ZONES):
                avg = self.avglen[z] or 1.0
                for d, tf, _ in self.postings[z].get(t, ()):
                    if cand is None or d in cand:
                        wtf[d] += w[z] * tf / (1 - b + b * self.length[z][d] / avg)
            idf = self.idf(t)
            for d, x in wtf.items():
                acc[d] += idf * x * (k1 + 1) / (x + k1)
        return heapq.nlargest(k, acc.items(), key=lambda x: x[1])

    def bm25f_weighted(self, qweights: dict[str, float], k: int = 10, k1: float = 1.2, b: float = 0.75,
                       weights: dict[str, float] | None = None, allowed: set[int] | None = None):
        """BM25F where each query term carries a weight (expanded / Rocchio queries)."""
        w = weights or DEFAULT_ZONE_WEIGHTS
        acc: dict[int, float] = defaultdict(float)
        for t, qw in qweights.items():
            if qw <= 0:
                continue
            wtf: dict[int, float] = defaultdict(float)
            for z in ZONES:
                avg = self.avglen[z] or 1.0
                for d, tf, _ in self.postings[z].get(t, ()):
                    if allowed is None or d in allowed:
                        wtf[d] += w[z] * tf / (1 - b + b * self.length[z][d] / avg)
            idf = self.idf(t)
            for d, x in wtf.items():
                acc[d] += qw * idf * x * (k1 + 1) / (x + k1)
        return heapq.nlargest(k, acc.items(), key=lambda x: x[1])

    def term_vector(self, d: int) -> dict[str, float]:
        """tf-idf vector of a doc (used by Rocchio feedback)."""
        if not hasattr(self, "_fwd"):
            fwd = [defaultdict(int) for _ in range(self.N)]
            for z in ZONES:
                for t, plist in self.postings[z].items():
                    for doc, tf, _ in plist:
                        fwd[doc][t] += tf
            self._fwd = fwd
        return {t: (1 + math.log(tf)) * self.idf(t) for t, tf in self._fwd[d].items()}

    def __getstate__(self):
        s = dict(self.__dict__)
        s.pop("_fwd", None)
        s.pop("_total_len", None)
        return s
