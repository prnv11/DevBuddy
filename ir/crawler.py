"""
Polite focused crawler for the FastAPI and Flask documentation.

Politeness and integrity
  * robots.txt is fetched once per host and every URL is checked against it
    (urllib.robotparser); a robots Crawl-delay larger than ours wins.
  * At least `--delay` seconds (default 1.0) between requests to one host.
    Hosts are scheduled independently, so two sites crawl in parallel
    without ever hitting one host faster than the delay.
  * Back-off on 429 / 5xx.
  * URL normalisation: lower-case scheme/host, default port removed,
    fragment dropped, dot-segments resolved, "index.html" stripped,
    directory-style trailing slash.
  * Spider-trap filters: any query string, old/alternate version trees,
    translated copies, over-deep or repeating paths, out-of-scope sections.
  * Exact duplicates (SHA-256 of main text) and near duplicates (Jaccard over
    word 5-shingles >= 0.85) are not indexed.

Output (JSONL):  pages.jsonl, chunks.jsonl, crawl_report.json

Live run (on your own machine):
    python -m ir.crawler --out data/
Offline run against the local mirror (see tools/serve_mirror.py):
    python -m ir.crawler --out data/ \
        --mirror fastapi=http://127.0.0.1:8101 --mirror flask=http://127.0.0.1:8102
URLs are always stored in their canonical live form, so chunk ids and
relevance judgments are identical between the two runs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import posixpath
import re
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from ir.extract import extract

USER_AGENT = "DevBuddyBot/1.0 (IR course project; polite crawler)"


@dataclass
class Site:
    framework: str
    live: str                       # canonical origin, e.g. https://fastapi.tiangolo.com
    seeds: list[str]
    allow: list[str]                # path regexes; a URL must match one
    deny: list[str] = field(default_factory=list)


SITES = {
    "fastapi": Site(
        framework="fastapi",
        live="https://fastapi.tiangolo.com",
        seeds=["/", "/tutorial/", "/advanced/", "/reference/", "/how-to/", "/deployment/"],
        allow=[r"^/$", r"^/(tutorial|advanced|how-to|reference|deployment)/",
               r"^/(features|python-types|async|fastapi-cli|environment-variables|virtual-environments|"
               r"alternatives|history-design-future|benchmarks)/$"],
        deny=[r"^/[a-z]{2}(-[a-z]+)?/",          # translations: /es/, /zh/, /zh-hant/ ...
              r"^/(release-notes|fastapi-people|management|search|img|css|js)/"],
    ),
    "flask": Site(
        framework="flask",
        live="https://flask.palletsprojects.com",
        seeds=["/en/stable/"],
        allow=[r"^/en/stable/"],
        deny=[r"^/en/stable/(_sources|_static|_images|search|genindex|py-modindex|changes|license|contributing)/"],
    ),
}

VERSION_RE = re.compile(r"/(en/)?(latest|\d+\.\d+(\.x)?|v\d[\w.]*|dev|main|master)/")


class Crawler:
    def __init__(self, sites: list[Site], mirror: dict[str, str], delay: float = 1.0,
                 max_pages: int = 600, max_depth: int = 8, timeout: float = 15.0,
                 raw_dir: Path | None = None):
        self.raw_dir = raw_dir                   # if set, fetched HTML is kept (gzip) for --reextract
        self.sites = sites
        self.mirror = mirror                     # framework -> fetch origin
        self.delay = delay
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.timeout = timeout
        self.robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.host_delay: dict[str, float] = {}
        self.next_ok: dict[str, float] = {}
        self.queues: dict[str, deque] = {s.framework: deque() for s in sites}
        self.seen: set[str] = set()
        self.stats = Counter()
        self.skips = Counter()
        self.skip_examples: dict[str, list[str]] = {}
        self.pages, self.chunks = [], []
        self.hashes: set[str] = set()
        self.shingles: list[tuple[str, set]] = []
        self.fetch_log: list[tuple[str, float]] = []

    # ---------------- URL handling ----------------
    def site_for(self, url: str) -> Site | None:
        origin = "{0.scheme}://{0.netloc}".format(urlsplit(url)).lower()
        for s in self.sites:
            if origin in (s.live.lower(), self.mirror.get(s.framework, "").lower()):
                return s
        return None

    @staticmethod
    def normalize(url: str) -> str | None:
        p = urlsplit(url.strip())
        if p.scheme not in ("http", "https"):
            return None
        host = (p.hostname or "").lower()
        port = p.port
        netloc = host if port in (None, 80, 443) else f"{host}:{port}"
        path = posixpath.normpath(p.path or "/") if p.path else "/"
        if p.path.endswith("/") and not path.endswith("/"):
            path += "/"
        path = re.sub(r"/index\.html?$", "/", path)
        if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
            path += "/"
        path = re.sub(r"/{2,}", "/", path)
        query = p.query
        return urlunsplit((p.scheme.lower(), netloc, path, query, ""))

    def to_canonical(self, url: str) -> str:
        for fw, origin in self.mirror.items():
            if url.startswith(origin):
                return SITES[fw].live + url[len(origin):]
        return url

    def to_fetch(self, url: str, site: Site) -> str:
        if site.framework in self.mirror and url.startswith(site.live):
            return self.mirror[site.framework] + url[len(site.live):]
        return url

    def canonical_link(self, url: str) -> str | None:
        """Normalisation used for stored outlinks (no trap checks, canonical host)."""
        n = self.normalize(url)
        if not n:
            return None
        return self.to_canonical(n)   # query strings kept so the trap filter sees them

    def reject_reason(self, url: str, site: Site | None, depth: int) -> str | None:
        if site is None:
            return "off-site"
        p = urlsplit(url)
        if p.query:
            return "query-string"
        path = p.path
        if VERSION_RE.search(path) and not path.startswith("/en/stable/"):
            return "old-version"
        if any(re.search(d, path) for d in site.deny):
            return "denied-section"
        if not any(re.search(a, path) for a in site.allow):
            return "out-of-scope"
        segs = [s for s in path.split("/") if s]
        if len(segs) > 8 or len(url) > 200:
            return "too-deep"
        if len(segs) != len(set(segs)):
            return "repeating-path"
        if re.search(r"\.(png|jpe?g|gif|svg|css|js|zip|gz|pdf|txt|ico|woff2?)$", path, re.I):
            return "non-html"
        if depth > self.max_depth:
            return "max-depth"
        return None

    # ---------------- politeness ----------------
    def robot(self, fetch_url: str) -> urllib.robotparser.RobotFileParser:
        p = urlsplit(fetch_url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                self._wait(origin)
                req = urllib.request.Request(origin + "/robots.txt", headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    rp.parse(r.read().decode("utf-8", "replace").splitlines())
                self.stats["robots_fetched"] += 1
            except urllib.error.HTTPError as e:
                rp.parse([] if e.code in (401, 403, 404) else ["User-agent: *", "Disallow: /"])
            except Exception:
                rp.parse([])
            cd = rp.crawl_delay(USER_AGENT)
            self.host_delay[origin] = max(self.delay, float(cd) if cd else 0.0)
            self.robots[origin] = rp
        return self.robots[origin]

    def _wait(self, origin: str):
        now = time.monotonic()
        t = self.next_ok.get(origin, 0.0)
        if t > now:
            time.sleep(t - now)
        self.next_ok[origin] = time.monotonic() + self.host_delay.get(origin, self.delay)
        self.fetch_log.append((origin, time.monotonic()))

    def fetch(self, url: str) -> tuple[int, str, str]:
        origin = "{0.scheme}://{0.netloc}".format(urlsplit(url))
        self._wait(origin)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                ctype = r.headers.get("Content-Type", "")
                body = r.read().decode("utf-8", "replace")
                return r.status, ctype, body
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                retry = e.headers.get("Retry-After")
                back = float(retry) if retry and retry.isdigit() else 30.0
                self.next_ok[origin] = time.monotonic() + back
            return e.code, "", ""

    # ---------------- integrity ----------------
    @staticmethod
    def _shingles(text: str, k: int = 5) -> set:
        w = text.lower().split()
        return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}

    def duplicate(self, url: str, text: str) -> str | None:
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in self.hashes:
            return "exact-duplicate"
        self.hashes.add(h)
        sh = self._shingles(text)
        for other, osh in self.shingles:
            if not osh or not sh:
                continue
            small, big = (sh, osh) if len(sh) < len(osh) else (osh, sh)
            if len(small) / len(big) < 0.85:
                continue                       # Jaccard cannot reach 0.85
            j = len(sh & osh) / len(sh | osh)
            if j >= 0.85:
                return f"near-duplicate-of:{other}"
        self.shingles.append((url, sh))
        return None

    # ---------------- main loop ----------------
    def skip(self, reason: str, url: str):
        self.skips[reason] += 1
        ex = self.skip_examples.setdefault(reason, [])
        if len(ex) < 5:
            ex.append(url)

    def enqueue(self, url: str, depth: int):
        n = self.normalize(url)
        if not n:
            return
        site = self.site_for(n)
        canon = self.to_canonical(n)
        if canon in self.seen:
            return
        self.seen.add(canon)
        why = self.reject_reason(canon, site, depth)
        if why:
            self.skip(why, canon)
            return
        self.queues[site.framework].append((canon, depth))

    def run(self, log_every: int = 25):
        for s in self.sites:
            for seed in s.seeds:
                self.enqueue(s.live + seed, 0)
        t0 = time.time()
        while any(self.queues.values()) and len(self.pages) < self.max_pages:
            # choose the host that can be hit soonest (per-host scheduling)
            ready = []
            for fw, q in self.queues.items():
                if q:
                    site = SITES[fw]
                    origin = "{0.scheme}://{0.netloc}".format(urlsplit(self.to_fetch(q[0][0], site)))
                    heapq.heappush(ready, (self.next_ok.get(origin, 0.0), fw))
            _, fw = heapq.heappop(ready)
            canon, depth = self.queues[fw].popleft()
            site = SITES[fw]
            fetch_url = self.to_fetch(canon, site)
            if not self.robot(fetch_url).can_fetch(USER_AGENT, fetch_url):
                self.skip("robots-disallowed", canon)
                continue
            status, ctype, body = self.fetch(fetch_url)
            self.stats["fetched"] += 1
            if status != 200 or "html" not in ctype:
                self.skip(f"http-{status}" if status != 200 else "non-html", canon)
                continue
            page, chunks = extract(body, canon, fw, self.canonical_link)
            if self.raw_dir is not None:
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                name = hashlib.sha1(canon.encode()).hexdigest()[:16] + ".html.gz"
                with gzip.open(self.raw_dir / name, "wt", encoding="utf-8") as f:
                    f.write(f"<!-- {canon} | {fw} -->\n" + body)
            dup = self.duplicate(canon, page["text"])
            # links are still followed from duplicates (they may lead elsewhere)
            for link in page["outlinks"] + page["nav_links"]:
                self.enqueue(link, depth + 1)
            if dup:
                self.skip(dup.split(":")[0], canon)
                continue
            page["depth"] = depth
            page["sha256"] = hashlib.sha256(page["text"].encode()).hexdigest()
            page["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.pages.append(page)
            self.chunks += chunks
            if len(self.pages) % log_every == 0:
                print(f"  {len(self.pages)} pages, {len(self.chunks)} chunks, "
                      f"queue={sum(map(len, self.queues.values()))}, {time.time() - t0:.0f}s", flush=True)
        self.elapsed = time.time() - t0

    def min_gap_per_host(self) -> dict[str, float]:
        last, gaps = {}, {}
        for origin, t in self.fetch_log:
            if origin in last:
                g = t - last[origin]
                gaps[origin] = min(gaps.get(origin, 1e9), g)
            last[origin] = t
        return gaps

    def remove_templates(self, min_share: float = 0.3, min_pages: int = 10) -> dict:
        """Template (boilerplate) removal inside the main content.

        A sentence that appears on >= min_share of a site's pages is site template text
        (e.g. a banner the theme inserts at the top of every article), not content.
        A code block is removed only if it is that common AND only ever occurs in chunks
        that also contain template text (so shared real snippets like `app = FastAPI()`
        are kept). Chunks left empty are dropped.
        """
        split = re.compile(r"(?<=[.!?])\s+")
        pages_of_fw: dict[str, set] = {}
        sent_pages: dict[tuple, set] = {}
        for c in self.chunks:
            pages_of_fw.setdefault(c["framework"], set()).add(c["page_url"])
            for sent in split.split(c["text"]):
                sent = sent.strip()
                if len(sent.split()) >= 4:
                    sent_pages.setdefault((c["framework"], sent), set()).add(c["page_url"])
        template = {k for k, v in sent_pages.items()
                    if len(v) >= max(min_pages, min_share * len(pages_of_fw[k[0]]))}
        if not template:
            return {"sentences": [], "code_blocks": [], "chunks_dropped": 0}

        def has_template(c):
            return any((c["framework"], x.strip()) in template for x in split.split(c["text"]))

        code_pages: dict[tuple, set] = {}
        code_n: dict[tuple, int] = {}          # chunks containing the block
        code_in_template: dict[tuple, int] = {}  # ... of which also contain template text
        flags = [has_template(c) for c in self.chunks]
        for c, ht in zip(self.chunks, flags):
            for blk in set(filter(None, (b.strip() for b in c["code"].split("\n\n")))):
                key = (c["framework"], blk)
                code_pages.setdefault(key, set()).add(c["page_url"])
                code_n[key] = code_n.get(key, 0) + 1
                code_in_template[key] = code_in_template.get(key, 0) + int(ht)
        # common code that (almost) always sits in template chunks is part of the template;
        # it is removed only from those template chunks, so a genuine occurrence elsewhere stays
        template_code = {k for k, v in code_pages.items()
                         if len(v) >= max(min_pages, min_share * len(pages_of_fw[k[0]]))
                         and code_in_template[k] >= 0.9 * code_n[k]}
        kept = []
        for c, ht in zip(self.chunks, flags):
            fw = c["framework"]
            c["text"] = " ".join(x for x in split.split(c["text"]) if (fw, x.strip()) not in template).strip()
            if ht:
                c["code"] = "\n\n".join(b for b in c["code"].split("\n\n") if (fw, b.strip()) not in template_code).strip()
            if c["text"] or c["code"]:
                kept.append(c)
        dropped = len(self.chunks) - len(kept)
        self.chunks = kept
        return {"sentences": sorted(s for _, s in template)[:20], "code_blocks": sorted(b for _, b in template_code)[:20],
                "chunks_dropped": dropped}

    def save(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        self.template_report = self.remove_templates()
        # restrict outlinks to pages we actually indexed (closed link graph)
        indexed = {p["url"] for p in self.pages}   # (also drops ?query links)
        for p in self.pages:
            p["outlinks"] = [l for l in p["outlinks"] if l.split("#")[0] in indexed]
            p["nav_links"] = [l for l in p["nav_links"] if l.split("#")[0] in indexed]
        with open(out / "pages.jsonl", "w", encoding="utf-8") as f:
            for p in self.pages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        report = {
            "pages_indexed": len(self.pages),
            "pages_by_framework": Counter(p["framework"] for p in self.pages),
            "chunks": len(self.chunks),
            "chunks_by_framework": Counter(c["framework"] for c in self.chunks),
            "chunks_with_code": sum(1 for c in self.chunks if c["code"]),
            "content_edges": sum(len(p["outlinks"]) for p in self.pages),
            "nav_edges": sum(len(p["nav_links"]) for p in self.pages),
            "fetched": self.stats["fetched"],
            "skipped": dict(self.skips.most_common()),
            "skip_examples": self.skip_examples,
            "min_seconds_between_requests_per_host": {k: round(v, 3) for k, v in self.min_gap_per_host().items()},
            "elapsed_seconds": round(self.elapsed, 1),
            "template_removed": self.template_report,
            "delay": self.delay,
            "mirror": self.mirror,
        }
        (out / "crawl_report.json").write_text(json.dumps(report, indent=2))
        return report


def reextract(out: Path):
    """Re-run extraction over the stored HTML of the last crawl, in crawl order."""
    old = [json.loads(l) for l in open(out / "pages.jsonl", encoding="utf-8")]
    c = Crawler(list(SITES.values()), {}, raw_dir=None)
    for p in old:
        f = out / "raw" / (hashlib.sha1(p["url"].encode()).hexdigest()[:16] + ".html.gz")
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            body = fh.read().split("\n", 1)[1]
        page, chunks = extract(body, p["url"], p["framework"], c.canonical_link)
        for k in ("depth", "sha256", "fetched_at"):
            page[k] = p.get(k)
        c.pages.append(page)
        c.chunks += chunks
    c.elapsed = 0.0
    old_report = json.loads((out / "crawl_report.json").read_text())
    rep = c.save(out)
    for k in ("fetched", "skipped", "skip_examples", "min_seconds_between_requests_per_host",
              "elapsed_seconds", "mirror"):
        if k not in old_report:
            continue
        rep[k] = old_report.get(k)
    (out / "crawl_report.json").write_text(json.dumps(rep, indent=2))
    (out / "index.pkl").unlink(missing_ok=True)
    print(f"re-extracted {len(c.pages)} pages -> {len(c.chunks)} chunks")
    tr = rep["template_removed"]
    print(f"template removal: {len(tr['sentences'])} sentences, {len(tr['code_blocks'])} code blocks, "
          f"{tr['chunks_dropped']} chunks dropped")
    for x in tr["sentences"] + tr["code_blocks"]:
        print("   removed:", x[:100])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--sites", nargs="+", default=["fastapi", "flask"])
    ap.add_argument("--delay", type=float, default=1.0, help="min seconds between requests to one host (>=1)")
    ap.add_argument("--max-pages", type=int, default=600)
    ap.add_argument("--mirror", action="append", default=[], metavar="FRAMEWORK=ORIGIN",
                    help="fetch FRAMEWORK from a local mirror origin instead of the live host")
    ap.add_argument("--no-raw", action="store_true", help="do not keep raw HTML in OUT/raw/")
    ap.add_argument("--reextract", action="store_true",
                    help="no network: rebuild pages/chunks from OUT/raw/ (after changing ir/extract.py)")
    a = ap.parse_args()
    if a.reextract:
        return reextract(a.out)
    if a.delay < 1.0:
        ap.error("--delay must be at least 1.0 second (politeness)")
    mirror = dict(m.split("=", 1) for m in a.mirror)
    c = Crawler([SITES[s] for s in a.sites], mirror, delay=a.delay, max_pages=a.max_pages,
                raw_dir=None if a.no_raw else a.out / "raw")
    c.run()
    rep = c.save(a.out)
    print(json.dumps({k: rep[k] for k in ("pages_indexed", "pages_by_framework", "chunks",
                                         "chunks_with_code", "content_edges", "skipped",
                                         "min_seconds_between_requests_per_host", "elapsed_seconds")},
                     indent=2, default=dict))


if __name__ == "__main__":
    main()
