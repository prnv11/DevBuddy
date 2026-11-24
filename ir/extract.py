"""
HTML -> page record + heading-bounded chunks.

Main content only: the sidebar, header, footer and version switcher are
removed before anything is extracted, so `outlinks` are *content* links
(the ones that carry editorial endorsement for PageRank/HITS). Navigation
links are kept separately in `nav_links` so the effect of including them
can be measured.

Chunk record (one per heading section, or per Sphinx API object):
    id, framework, url, page_title, heading_path, text, code, outlinks
"""
from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

MAIN_SELECTORS = [
    "article.md-content__inner",        # Material for MkDocs (FastAPI)
    'div.body[role="main"]',            # Sphinx / Pallets theme (Flask)
    "main article", "article", "main", '[role="main"]',
]
NAV_SELECTORS = ["nav", ".md-sidebar", ".sphinxsidebar", "header", "footer", ".md-footer",
                 ".related", ".footer"]
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
LEAF_BLOCKS = {"p", "li", "td", "th", "dt", "blockquote", "figcaption", "caption", "span", "em",
               "strong", "a", "code", "b", "i"}
MAX_WORDS = 350


def find_main(soup: BeautifulSoup) -> Tag | None:
    for sel in MAIN_SELECTORS:
        el = soup.select_one(sel)
        if el is not None:
            return el
    return soup.body


def links_in(el: Tag, base: str) -> list[str]:
    out = []
    for a in el.find_all("a", href=True) if isinstance(el, Tag) else []:
        if "headerlink" in (a.get("class") or []):
            continue
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        out.append(urljoin(base, href))
    if isinstance(el, Tag) and el.name == "a" and el.get("href"):
        out.append(urljoin(base, el["href"]))
    return out


def clean_text(s: str) -> str:
    s = s.replace("¶", " ")
    return re.sub(r"\s+", " ", s).strip()


class _Chunker:
    def __init__(self, base_url: str):
        self.base = base_url
        self.path: list[tuple[int, str, str]] = []   # (level, title, anchor)
        self.chunks: list[dict] = []
        self._new()

    def _new(self):
        self.cur = {"heading_path": [t for _, t, _ in self.path],
                    "anchor": self.path[-1][2] if self.path else "",
                    "text": [], "code": [], "links": []}
        self.chunks.append(self.cur)

    def heading(self, level: int, title: str, anchor: str):
        while self.path and self.path[-1][0] >= level:
            self.path.pop()
        self.path.append((level, title, anchor))
        self._new()

    def pop_to(self, level: int):
        while self.path and self.path[-1][0] >= level:
            self.path.pop()

    def text(self, s: str, el: Tag | None = None):
        s = clean_text(s)
        if s:
            self.cur["text"].append(s)
        if el is not None:
            self.cur["links"] += links_in(el, self.base)

    def code(self, s: str):
        s = s.strip("\n")
        if s.strip():
            self.cur["code"].append(s)


PYPATH_RE = re.compile(r"^[a-z_][a-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")


def qualified(anchor: str, fallback: str) -> str:
    """API anchors like 'flask.Request.args' -> 'Request.args' (keeps the class)."""
    if anchor and PYPATH_RE.match(anchor):
        parts = anchor.split(".")
        return ".".join(parts[1:]) if len(parts) > 1 else anchor
    return fallback


def _heading_anchor(el: Tag) -> str:
    if el.get("id"):
        return el["id"]
    parent = el.parent
    if isinstance(parent, Tag) and parent.name == "section" and parent.get("id"):
        return parent["id"]
    a = el.find("a", class_="headerlink")
    return a["href"].lstrip("#") if a and a.get("href") else ""


def _walk(el: Tag, ch: _Chunker, obj_level: int):
    for child in el.children:
        if isinstance(child, NavigableString):
            ch.text(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in ("script", "style", "nav", "form", "button", "svg", "img"):
            continue
        if name in HEADINGS:
            title = clean_text(child.get_text(" "))
            anchor = _heading_anchor(child)
            ch.heading(HEADINGS[name], qualified(anchor, title), anchor)
            continue
        if name == "pre":
            ch.code(child.get_text())
            ch.cur["links"] += links_in(child, ch.base)
            continue
        if name == "dl" and "py" in (child.get("class") or []):
            # Sphinx API object: <dl class="py method"><dt id=...>sig</dt><dd>doc</dd></dl>
            for part in child.find_all(["dt", "dd"], recursive=False):
                if part.name == "dt":
                    sig = clean_text(part.get_text(" "))
                    sig = re.sub(r"^(class|method|property|function|data|attribute|exception)\s+", "", sig)
                    short = sig.split("(")[0].strip() + ("()" if "(" in sig else "")
                    anchor = part.get("id", "")
                    q = qualified(anchor, short)
                    ch.heading(obj_level, q + ("()" if "(" in sig and q != short else ""), anchor)
                    ch.text(sig)
                else:
                    _walk(part, ch, obj_level + 1)
            ch.pop_to(obj_level)
            ch._new()
            continue
        has_struct = child.find(list(HEADINGS) + ["pre", "dl"]) is not None
        if not has_struct and name in LEAF_BLOCKS | {"div", "section", "ul", "ol", "table", "dl", "dd",
                                                     "details", "summary", "tr", "tbody", "thead"}:
            ch.text(child.get_text(" "), child)
        else:
            _walk(child, ch, obj_level)


def _split_long(text_parts: list[str], max_words=MAX_WORDS) -> list[list[str]]:
    groups, cur, n = [], [], 0
    for t in text_parts:
        w = len(t.split())
        if cur and n + w > max_words:
            groups.append(cur)
            cur, n = [], 0
        cur.append(t)
        n += w
    if cur:
        groups.append(cur)
    return groups or [[]]


def extract(html: str, url: str, framework: str, normalize) -> tuple[dict, list[dict]]:
    """Return (page_record, chunks). `normalize(url) -> url|None` maps links."""
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.find("title")
    nav_links = []
    for sel in NAV_SELECTORS:
        for el in soup.select(sel):
            nav_links += links_in(el, url)
    main = find_main(soup)
    for sel in NAV_SELECTORS:
        for el in main.select(sel):
            el.decompose()
    h1 = main.find("h1")
    page_title = clean_text(h1.get_text(" ")) if h1 else clean_text(title_el.get_text() if title_el else url)

    ch = _Chunker(url)
    _walk(main, ch, obj_level=5)

    def norm_all(links):
        seen, out = set(), []
        for l in links:
            n = normalize(l)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    chunks, used = [], {}
    slug = re.sub(r"^https?://[^/]+/", "", url).strip("/") or "index"
    slug = re.sub(r"^en/stable(/|$)", "", slug) or "index"
    for c in ch.chunks:
        if not c["text"] and not c["code"]:
            continue
        if not c["heading_path"]:
            c["heading_path"] = [page_title]
        outlinks = [l for l in norm_all(c["links"]) if l != url]
        groups = _split_long(c["text"])
        for gi, g in enumerate(groups):
            base_id = f"{framework}/{slug}#{c['anchor'] or 'top'}"
            k = used.get(base_id, 0)
            used[base_id] = k + 1
            cid = base_id if k == 0 else f"{base_id}~{k}"
            chunks.append({
                "id": cid,
                "framework": framework,
                "url": url + (f"#{c['anchor']}" if c["anchor"] else ""),
                "page_url": url,
                "page_title": page_title,
                "heading_path": c["heading_path"],
                "text": " ".join(g),
                "code": "\n\n".join(c["code"]) if gi == 0 else "",
                "outlinks": outlinks if gi == 0 else [],
            })
    content_links = [l for l in norm_all(links_in(main, url)) if l != url]
    page = {"url": url, "framework": framework, "title": page_title,
            "outlinks": content_links,
            "nav_links": [l for l in norm_all(nav_links) if l != url],
            "text": clean_text(main.get_text(" "))}
    return page, chunks
