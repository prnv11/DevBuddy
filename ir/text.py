"""
Code-aware tokenizer.

Identifiers are indexed in their full form *and* split into their parts, so
both exact API names and natural-language words match:

    request.args      -> request.args, request, args
    APIRouter         -> apirouter, api, router
    HTTPException     -> httpexception, http, exception
    add_url_rule      -> add_url_rule, add, url, rule
    response_model    -> response_model, response, model

All forms produced from one source token share that token's position, so
positions (used for phrase / proximity features) stay aligned with the text.
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|\d+(?:\.\d+)*")
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset("""
a an and are as at be but by for from has have how i if in into is it its of on or that the
their then there these this to was what when where which while who why will with you your
can do does should would we our my me so not no yes than too very just also any all each
""".split())


def _parts(word: str) -> list[str]:
    """Split one identifier (no dots) into snake / camel parts."""
    out = []
    for piece in word.split("_"):
        if not piece:
            continue
        out += [m.group(0) for m in CAMEL_RE.finditer(piece)]
    return out


def analyze(text: str, keep_stopwords: bool = False) -> list[tuple[str, int]]:
    """Return (term, position) pairs."""
    out: list[tuple[str, int]] = []
    pos = 0
    for m in TOKEN_RE.finditer(text):
        raw = m.group(0)
        forms: list[str] = []
        full = raw.lower().strip("_.")
        if not full:
            continue
        forms.append(full)
        segments = raw.split(".")
        if len(segments) > 1:
            forms += [s.lower() for s in segments if s]
        for seg in segments:
            p = _parts(seg)
            if len(p) > 1:
                forms += [x.lower() for x in p]
        seen = set()
        for f in forms:
            if f in seen or len(f) < 2 and not f.isdigit():
                continue
            seen.add(f)
            if not keep_stopwords and f in STOPWORDS and "." not in f and "_" not in f:
                continue
            out.append((f, pos))
        pos += 1
    return out


def terms(text: str) -> list[str]:
    return [t for t, _ in analyze(text)]
