"""Unit tests for the IR components.  Run:  python -m pytest -q tests/"""
import math

import networkx as nx
import numpy as np
import pytest

from eval.common import cohen_kappa, paired_bootstrap, per_query_metrics
from ir.cluster import spherical_kmeans
from ir.crawler import SITES, Crawler
from ir.feedback import rocchio_terms
from ir.index import InvertedIndex, parse_query
from ir.links import LinkGraph
from ir.text import analyze, terms


# ---------------- tokenizer ----------------
@pytest.mark.parametrize("text,expected", [
    ("request.args", {"request.args", "request", "args"}),
    ("APIRouter", {"apirouter", "api", "router"}),
    ("HTTPException", {"httpexception", "http", "exception"}),
    ("add_url_rule", {"add_url_rule", "add", "url", "rule"}),
])
def test_tokenizer_keeps_full_and_parts(text, expected):
    assert set(terms(text)) == expected


def test_tokenizer_parts_share_position():
    toks = analyze("use request.args here")
    pos = {t: p for t, p in toks}
    assert pos["request.args"] == pos["request"] == pos["args"] == 1
    assert pos["use"] == 0 and pos["here"] == 2


# ---------------- index / BM25 ----------------
def _chunks():
    return [
        {"id": f"d{i}", "framework": fw, "url": f"u{i}", "page_url": f"https://x/{sec}/p{i}/", "page_title": "T",
         "heading_path": ["T"], "text": t, "code": ""}
        for i, (fw, sec, t) in enumerate([
            ("flask", "a", "blueprint routes blueprint"),
            ("flask", "b", "routes and views"),
            ("fastapi", "a", "apirouter routes"),
        ])
    ]


def test_bm25_matches_formula():
    ix = InvertedIndex.build(_chunks())
    k1, b = 1.2, 0.75
    N, df = 3, 1
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    lens = [sum(ix.length[z][d] for z in ix.length) for d in range(3)]
    avg = sum(lens) / 3
    tf = 2
    expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lens[0] / avg))
    (d, s), = ix.bm25("blueprint", k=1)
    assert d == 0 and s == pytest.approx(expected)


def test_parametric_and_boolean_filters():
    ix = InvertedIndex.build(_chunks())
    assert {d for d, _ in ix.bm25f("framework:fastapi routes")} == {2}
    assert {d for d, _ in ix.bm25f("routes -blueprint")} == {1, 2}
    assert {d for d, _ in ix.bm25f("routes +views")} == {1}
    assert {d for d, _ in ix.bm25f("section:b routes")} == {1}
    q = parse_query("framework:flask code:jsonify hello")
    assert q.filters == {"framework": "flask"} and ("jsonify", "code") in q.terms


# ---------------- link analysis ----------------
def _graph(seed=0, n=40, p=0.08):
    rng = np.random.default_rng(seed)
    pages = []
    for i in range(n):
        out = [f"p{j}" for j in range(n) if j != i and rng.random() < p]
        if i % 7 == 0:
            out = []                       # dangling pages
        pages.append({"url": f"p{i}", "framework": "flask" if i % 2 else "fastapi", "outlinks": out, "nav_links": []})
    return LinkGraph(pages)


def test_pagerank_matches_networkx():
    g = _graph()
    r, deltas = g.pagerank()
    ref = nx.pagerank(g.to_networkx(), alpha=0.85, tol=1e-12)
    assert np.abs(r - np.array([ref[i] for i in range(g.N)])).max() < 1e-7
    assert deltas[-1] < 1e-8 and r.sum() == pytest.approx(1.0)


def test_personalised_pagerank_matches_networkx():
    g = _graph(1)
    v = g.teleport_for({"flask"})
    r, _ = g.pagerank(teleport=v)
    ref = nx.pagerank(g.to_networkx(), alpha=0.85, personalization={i: v[i] for i in range(g.N)},
                      dangling={i: 1.0 for i in range(g.N)}, tol=1e-12)
    assert np.abs(r - np.array([ref[i] for i in range(g.N)])).max() < 1e-7


def test_hits_matches_networkx_on_base_set():
    g = _graph(2, n=25, p=0.15)
    root = list(range(g.N))
    auth, hub = g.hits(root, iters=500, tol=1e-12)
    h_ref, a_ref = nx.hits(g.to_networkx(), max_iter=1000, tol=1e-12)
    a = np.array([auth[i] for i in range(g.N)])
    a_ref = np.array([a_ref[i] for i in range(g.N)])
    assert np.allclose(a / a.sum(), a_ref / a_ref.sum(), atol=1e-6)


# ---------------- metrics / agreement ----------------
def test_metrics_known_values():
    rel = {"a": 2, "b": 1, "c": 1}
    m = per_query_metrics(["a", "x", "b", "y"], rel)
    assert m["P@3"] == pytest.approx(2 / 3)
    assert m["RR"] == 1.0
    assert m["AP"] == pytest.approx((1 / 1 + 2 / 3) / 3)
    dcg = 3 / math.log2(2) + 1 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert m["nDCG@10"] == pytest.approx(dcg / idcg)


def test_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, 200).tolist()
    y = [v if rng.random() < 0.7 else int(rng.integers(0, 3)) for v in x]
    assert cohen_kappa(x, y) == pytest.approx(cohen_kappa_score(x, y))
    assert cohen_kappa(x, y, weights="linear") == pytest.approx(cohen_kappa_score(x, y, weights="linear"))


def test_bootstrap_detects_shift_and_null():
    rng = np.random.default_rng(0)
    a = rng.random(60).tolist()
    assert paired_bootstrap(a, [v + 0.2 for v in a]) < 0.01
    assert paired_bootstrap(a, a) == 1.0


# ---------------- crawler ----------------
def test_crawler_normalisation_and_traps():
    c = Crawler(list(SITES.values()), {})
    n = c.normalize
    assert n("HTTPS://Flask.PalletsProjects.com:443/en/stable/quickstart/index.html#x") == \
        "https://flask.palletsprojects.com/en/stable/quickstart/"
    assert n("https://fastapi.tiangolo.com/tutorial/../advanced/./events") == "https://fastapi.tiangolo.com/advanced/events/"
    fl, fa = SITES["flask"], SITES["fastapi"]
    assert c.reject_reason("https://flask.palletsprojects.com/en/2.3.x/api/", fl, 1) == "old-version"
    assert c.reject_reason("https://flask.palletsprojects.com/en/stable/search/?q=a", fl, 1) == "query-string"
    assert c.reject_reason("https://fastapi.tiangolo.com/es/tutorial/", fa, 1) == "denied-section"
    assert c.reject_reason("https://fastapi.tiangolo.com/tutorial/a/tutorial/a/", fa, 1) == "repeating-path"
    assert c.reject_reason("https://fastapi.tiangolo.com/tutorial/first-steps/", fa, 1) is None


def test_near_duplicate_detection():
    c = Crawler(list(SITES.values()), {})
    text = " ".join(f"word{i}" for i in range(300))
    assert c.duplicate("u1", text) is None
    assert c.duplicate("u2", text) == "exact-duplicate"
    assert c.duplicate("u3", text + " extra").startswith("near-duplicate")


# ---------------- clustering / feedback ----------------
def test_spherical_kmeans_separates_clusters():
    rng = np.random.default_rng(0)
    A = rng.normal([5, 0, 0], 0.3, (30, 3))
    B = rng.normal([0, 5, 0], 0.3, (30, 3))
    X = np.vstack([A, B])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    lab, C = spherical_kmeans(X, 2, n_init=3)
    assert len(set(lab[:30])) == 1 and len(set(lab[30:])) == 1 and lab[0] != lab[30]
    assert np.allclose(np.linalg.norm(C, axis=1), 1.0)


class _Eng:
    def __init__(self):
        self.index = InvertedIndex.build(_chunks())


def test_rocchio_clips_negatives_and_limits_expansion():
    eng = _Eng()
    qw = rocchio_terms(eng, "routes", rel=[0], nonrel=[2], n_terms=1)
    assert all(w > 0 for w in qw.values())
    assert "routes" in qw and len(set(qw) - {"routes"}) <= 1
    assert "apirouter" not in qw            # only in the non-relevant doc -> negative -> clipped


def test_template_removal_keeps_real_content():
    c = Crawler(list(SITES.values()), {})
    banner = "Use the official skill for your installed version. Install it from your project directory now."
    c.chunks = []
    for i in range(20):
        c.chunks.append({"id": f"fastapi/p{i}#top", "framework": "fastapi", "page_url": f"u{i}",
                         "text": f"{banner} Page {i} explains topic number {i} in detail.",
                         "code": "uvx skills\n\napp = FastAPI()"})
        c.chunks.append({"id": f"fastapi/p{i}#more", "framework": "fastapi", "page_url": f"u{i}",
                         "text": f"More about topic {i} here today.", "code": "app = FastAPI()"})
    c.chunks.append({"id": "fastapi/skills#use", "framework": "fastapi", "page_url": "u99",
                     "text": "The skills page documents how agents work.", "code": "uvx skills"})
    rep = c.remove_templates()
    assert len(rep["sentences"]) == 2 and rep["code_blocks"] == ["uvx skills"]
    assert all("official skill" not in x["text"] for x in c.chunks)
    assert all("app = FastAPI()" in x["code"] for x in c.chunks if x["id"].startswith("fastapi/p"))
    assert next(x for x in c.chunks if x["id"] == "fastapi/skills#use")["code"] == "uvx skills"
