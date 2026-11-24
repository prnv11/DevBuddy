"""
Answer composition for the dashboard.

  * Single-framework question -> hybrid retrieval (framework filter if named).
  * Cross-framework question ("Flask X in FastAPI", "equivalent of", "vs") ->
    concept alignment (part 2): source chunks + aligned target chunks.
  * Answer: Groq LLM with numbered, cited sources if GROQ_API_KEY is set;
    otherwise an extractive answer built from the retrieved passages.
    Either way every claim points at a source URL.
  * Feedback: refine(question, relevant, non_relevant) re-ranks with Rocchio.
"""
from __future__ import annotations

import os
import re
import threading

from ir.engine import Engine

FW_NAMES = {"flask": "Flask", "fastapi": "FastAPI"}
CROSS_RE = re.compile(r"\b(equivalent|counterpart|instead of|compare|comparison|vs\.?|versus|mapping|map to|"
                      r"translate|port|migrat\w*|coming from|like flask|like fastapi)\b", re.I)

PROMPT = """You are DevBuddy, a precise assistant for developers learning a new web framework.
Answer ONLY from the numbered documentation excerpts. Cite every claim with [n].
If the excerpts do not contain the answer, say so. {extra}

{sources}

Question: {question}
Answer (markdown, concise, with a short code example if the excerpts contain one):"""


def mentioned_frameworks(q: str) -> list[str]:
    found = [(m.start(), m.group(0).lower()) for m in re.finditer(r"fastapi|flask", q, re.I)]
    out = []
    for _, f in sorted(found):
        if f not in out:
            out.append(f)
    return out


class Answerer:
    _lock = threading.Lock()
    _instance = None

    ALIGNER = "lsa/hac"          # fast to build; see eval/results/part2.md for the comparison

    def __init__(self, data_dir="data"):
        import json
        from pathlib import Path
        from ir.cluster import ConceptAligner
        self.eng = Engine(data_dir)
        space, method = self.ALIGNER.split("/")[:2]
        self.aligner = ConceptAligner(self.eng, space=space, method=method)
        self.align_kw = {"m": 5, "w": 0.7, "route": "evidence", "tau": 0.0}
        res = Path(__file__).resolve().parent.parent / "eval" / "results"
        for name in ("part1.json", "part2.json"):
            f = res / name
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            if name == "part1.json":                  # tuned BM25F / fusion weight (dev split)
                self.eng.params.update(d["tuned"]["bm25f"])
                if "hybrid_lsa" in d["tuned"]:
                    self.eng.params["rrf_w"] = d["tuned"]["hybrid_lsa"]["rrf_w"]
            elif self.ALIGNER in d.get("configs", {}):  # m, w, route, tau tuned on the seed pairs
                c = d["configs"][self.ALIGNER]
                self.align_kw = {k: c[k] for k in ("m", "w", "route", "one_to_one", "tau") if k in c}
        self._llm = None

    @classmethod
    def get(cls, data_dir="data") -> "Answerer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(data_dir)
            return cls._instance

    # ---------------- retrieval ----------------
    def route(self, q: str, primary: str | None, learning: str | None):
        fws = mentioned_frameworks(q)
        prof = [f.lower() for f in (primary, learning) if f and f.lower() in FW_NAMES]
        cross = bool(CROSS_RE.search(q)) or len(fws) == 2
        if cross:
            if len(fws) == 2:
                # "FastAPI equivalent of Flask X": the framework before equivalent/counterpart is the target;
                # "flask abort in fastapi": the framework after in/to/for is the target
                m1 = re.search(r"\b(fastapi|flask)(?:'s)?\s+(?:equivalent|counterpart|version|way)\b", q, re.I)
                m2 = re.search(r"\b(?:in|to|for|into|with|using)\s+(fastapi|flask)\b", q, re.I)
                tgt = (m1 or m2).group(1).lower() if (m1 or m2) else fws[1]
                src = fws[0] if fws[0] != tgt else fws[1]
            elif len(fws) == 1:
                src = fws[0]
                tgt = "fastapi" if src == "flask" else "flask"
            elif len(prof) == 2:
                src, tgt = prof[0], prof[1]
            else:
                src, tgt = "flask", "fastapi"
            return "mapping", src, tgt
        fw = fws[0] if fws else None
        return "search", fw, None

    def retrieve(self, q: str, primary=None, learning=None, k: int = 5):
        mode, a, b = self.route(q, primary, learning)
        eng = self.eng
        clean = re.sub(r"\b(fastapi|flask)\b", " ", q, flags=re.I)
        clean = CROSS_RE.sub(" ", clean)
        clean = re.sub(r"\b(what|is|the|of|in|to|for|how|do|i|equivalent)\b", " ", clean, flags=re.I).strip() or q
        if mode == "mapping":
            src_hits = eng.run("bm25f", f"framework:{a} {clean}", k=3)
            r = self.aligner.align(clean, src=a, tgt=b, k=k, **self.align_kw)
            info = {"source_cluster": self.aligner.describe(r.source_cluster) if r.source_cluster is not None else None,
                    "target_cluster": self.aligner.describe(r.target_clusters[0]) if r.target_clusters else None,
                    "similarity": round(r.similarity, 3), "no_match": r.no_match, "src": a, "tgt": b}
            ranked = src_hits + r.ranked
            return mode, info, ranked, clean
        query = f"framework:{a} {q}" if a else q
        return mode, {"framework": a}, eng.run("hybrid", query, k=k), q

    def sources(self, ranked):
        out, seen = [], set()
        for d, s in ranked:
            if d in seen:
                continue
            seen.add(d)
            c = self.eng.chunks[d]
            out.append({"n": len(out) + 1, "id": c["id"], "framework": c["framework"],
                        "url": c["url"], "heading": " > ".join(c["heading_path"]),
                        "snippet": c["text"][:400], "code": c["code"][:600], "score": round(float(s), 4)})
        return out

    # ---------------- generation ----------------
    def llm(self):
        if self._llm is None and os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
                self._llm = ChatGroq(model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0)
            except Exception:
                self._llm = False
        return self._llm or None

    def compose(self, question: str, mode: str, info: dict, srcs: list[dict]) -> str:
        if not srcs:
            return "I could not find anything about this in the FastAPI or Flask documentation index."
        llm = self.llm()
        if llm is not None:
            block = "\n\n".join(f"[{s['n']}] ({FW_NAMES[s['framework']]}) {s['heading']} - {s['url']}\n"
                                f"{s['snippet']}\n{s['code']}" for s in srcs)
            extra = ""
            if mode == "mapping":
                extra = (f"Compare the {FW_NAMES[info['src']]} concept with its {FW_NAMES[info['tgt']]} "
                         f"counterpart side by side. If there is no direct counterpart, say so.")
            try:
                return str(llm.invoke(PROMPT.format(extra=extra, sources=block, question=question)).content)
            except Exception as e:
                head = f"_LLM unavailable ({type(e).__name__}); showing the retrieved passages._\n\n"
                return head + self.extractive(mode, info, srcs)
        return self.extractive(mode, info, srcs)

    @staticmethod
    def extractive(mode: str, info: dict, srcs: list[dict]) -> str:
        def bullet(s):
            first = re.split(r"(?<=[.!?])\s", s["snippet"], maxsplit=2)
            text = " ".join(first[:2])
            return f"- **{s['heading'].split(' > ')[-1]}**: {text} [{s['n']}]"
        lines = []
        if mode == "mapping":
            a, b = FW_NAMES[info["src"]], FW_NAMES[info["tgt"]]
            src = [s for s in srcs if s["framework"] == info["src"]]
            tgt = [s for s in srcs if s["framework"] == info["tgt"]]
            lines.append(f"### {a} → {b}")
            if info.get("no_match"):
                lines.append(f"_Low-confidence match (cluster similarity {info['similarity']}): {b} may have no direct "
                             f"counterpart, so check the sources below._")
            lines.append(f"\n**In {a}:**")
            lines += [bullet(s) for s in src[:2]]
            lines.append(f"\n**Closest {b} documentation:**")
            lines += [bullet(s) for s in tgt[:4]]
            code = next((s["code"] for s in tgt if s["code"]), "")
        else:
            lines += [bullet(s) for s in srcs[:4]]
            code = next((s["code"] for s in srcs if s["code"]), "")
        if code:
            lines.append(f"\n```python\n{code.strip()}\n```")
        lines.append("\n_Extractive answer (no LLM configured): set GROQ_API_KEY for a written answer._")
        return "\n".join(lines)

    # ---------------- public API ----------------
    def answer(self, question: str, primary=None, learning=None, k: int = 5) -> dict:
        mode, info, ranked, _ = self.retrieve(question, primary, learning, k=k)
        srcs = self.sources(ranked)
        return {"mode": mode, "route": info, "sources": srcs, "response": self.compose(question, mode, info, srcs)}

    def refine(self, question: str, relevant: list[str], non_relevant: list[str], primary=None, learning=None,
               k: int = 5, regenerate: bool = True) -> dict:
        from ir.feedback import feedback_search
        eng = self.eng
        rel = [eng.id2idx[i] for i in relevant if i in eng.id2idx]
        non = [eng.id2idx[i] for i in non_relevant if i in eng.id2idx]
        mode, info, first, q = self.retrieve(question, primary, learning, k=k)
        non_set = set(non)
        if mode == "mapping":
            # keep the source-framework side, re-rank only the target-framework side
            keep = [(d, s) for d, s in first if eng.chunks[d]["framework"] == info["src"] and d not in non_set]
            tgt = feedback_search(eng, f"framework:{info['tgt']} {q}", rel, non, system="hybrid", k=k + len(non))
            ranked = keep[:3] + [(d, s) for d, s in tgt if d not in non_set][:k]
        else:
            ranked = feedback_search(eng, q, rel, non, system="hybrid", k=k + len(non))
            ranked = [(d, s) for d, s in ranked if d not in non_set][:k]
        srcs = self.sources(ranked)
        out = {"mode": mode, "route": info, "sources": srcs}
        if regenerate:
            out["response"] = self.compose(question, mode, info, srcs)
        return out
