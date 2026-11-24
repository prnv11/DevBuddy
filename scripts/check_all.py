"""
End-to-end health check for DevBuddy. Run from the project folder:

    python scripts/check_all.py          # everything except the long experiments (~1-2 min)
    python scripts/check_all.py --full   # also re-runs all experiments (~10 min)

Checks: environment, fixed files, crawled data, unit tests, every retrieval system,
answer + refine, and the whole web-app flow (register -> onboarding -> dashboard ->
chat -> refine -> logout -> login) on a TEMPORARY database. Your devbuddy.db is not touched.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        t = time.time()
        try:
            msg = fn() or ""
            RESULTS.append((name, True, msg))
            print(f"[PASS] {name} {msg} ({time.time() - t:.1f}s)", flush=True)
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=3)
        return fn
    return deco


def need(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------- 1. environment
@check("1. Python version >= 3.10")
def _():
    need(sys.version_info >= (3, 10), sys.version)
    return sys.version.split()[0]


@check("2. Required packages installed")
def _():
    mods = ["fastapi", "uvicorn", "sqlalchemy", "aiosqlite", "jinja2", "multipart", "jwt", "pwdlib",
            "dotenv", "numpy", "scipy", "sklearn", "networkx", "bs4", "lxml", "pytest", "httpx"]
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    need(not missing, f"missing: {missing} -> run: python -m pip install -r requirements.txt")
    st = importlib.util.find_spec("sentence_transformers") is not None
    return "(MiniLM: " + ("yes" if st else "NOT installed, dense = LSA") + ")"


@check("3. .env file present with required keys")
def _():
    env = ROOT / ".env"
    need(env.exists(), ".env not found in the project folder")
    text = env.read_text(encoding="utf-8", errors="replace")
    keys = ["SQLALCHEMY_DATABASE_URL", "SECRET_KEY", "ENCODING_ALGORITHM"]
    miss = [k for k in keys if k not in text]
    need(not miss, f"missing keys: {miss}")
    return "(GROQ_API_KEY set: " + ("yes" if "GROQ_API_KEY=" in text and "your_key" not in text else "no, extractive answers") + ")"


# ---------------------------------------------------------------- 2. files
@check("4. Fixed files are in the right place")
def _():
    must = {
        "app/main.py": "seed_tech_stacks",
        "app/api/auth.py": "await authenticate_user",
        "app/models/user.py": "String(255)",
        "app/api/user.py": "def refine_answer",
        "app/rag/pipeline/router.py": "Answerer",
        "ir/crawler.py": "def remove_templates",
        "ir/answer.py": "Low-confidence match",
        "ir/dense.py": "re-extracted chunks",
        "app/templates/dashboard.html": "low-confidence match",
    }
    bad = [f for f, pat in must.items() if pat not in (ROOT / f).read_text(encoding="utf-8")]
    need(not bad, f"old/wrong version of: {bad}")


# ---------------------------------------------------------------- 3. data
@check("5. Crawled data present (both frameworks)")
def _():
    pages = [json.loads(l) for l in open("data/pages.jsonl", encoding="utf-8")]
    chunks = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]
    fws = {c["framework"] for c in chunks}
    need(fws == {"flask", "fastapi"}, f"frameworks found: {fws}")
    need(len(pages) > 150 and len(chunks) > 1000, f"too small: {len(pages)} pages, {len(chunks)} chunks")
    ids = [c["id"] for c in chunks]
    need(len(ids) == len(set(ids)), "duplicate chunk ids")
    rep = json.loads(open("data/crawl_report.json", encoding="utf-8").read())
    need("template_removed" in rep, "data not re-extracted yet: run  python -m ir.crawler --out data --reextract")
    # a site banner repeats on many pages; one or two genuine mentions (e.g. FastAPI's own
    # "AI agent skills" section) are real content and must stay
    banner_pages = {c["page_url"] for c in chunks if "Use the official FastAPI skill" in c["text"]}
    need(len(banner_pages) < 5, f"site banner still on {len(banner_pages)} pages - re-run --reextract")
    return f"({len(pages)} pages, {len(chunks)} chunks)"


# ---------------------------------------------------------------- 4. unit tests
@check("6. Unit tests (pytest)")
def _():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    last = (r.stdout.strip().splitlines() or ["no output"])[-1]
    need(r.returncode == 0, last + "\n" + r.stdout[-1500:])
    return f"({last})"


# ---------------------------------------------------------------- 5. retrieval
ENG = {}


@check("7. Every retrieval system returns results")
def _():
    from ir.cluster import ConceptAligner
    from ir.engine import Engine
    eng = Engine("data")
    ConceptAligner(eng, space="lsa", method="kmeans").attach(eng)
    ENG["e"] = eng
    systems = eng.systems() + ["bm25f+pr", "bm25f+ppr", "bm25f+pprc", "bm25f+hits", "bm25f+prf"]
    empty = []
    for s in systems:
        if not eng.run(s, "framework:flask blueprint register routes", k=5):
            empty.append(s)
    need(not empty, f"no results from {empty}")
    top = eng.chunks[eng.run("bm25f", "request.args", k=1)[0][0]]["id"]
    return f"({len(systems)} systems; 'request.args' -> {top})"


@check("8. Concept alignment Flask -> FastAPI")
def _():
    from ir.cluster import ConceptAligner
    eng = ENG["e"]
    al = ConceptAligner(eng, space="lsa", method="hac")
    r = al.align("test_client", m=5, w=0.7, route="evidence")
    ids = [eng.chunks[d]["id"] for d, _ in r.ranked]
    need(ids and all(i.startswith("fastapi/") for i in ids), f"unexpected: {ids[:3]}")
    return f"(test_client -> {ids[0]})"


@check("9. Rocchio feedback re-ranks")
def _():
    from ir.feedback import feedback_search
    eng = ENG["e"]
    first = eng.run("bm25f", "split app into modules", k=5)
    rel, non = [first[2][0]], [first[0][0]]
    new = feedback_search(eng, "split app into modules", rel, non, system="bm25f", k=5)
    need([d for d, _ in new] != [d for d, _ in first], "ranking did not change")


# ---------------------------------------------------------------- 6. web app (temporary DB)
@check("10. Web app: pages, register, onboarding, chat, refine, login")
def _():
    tmp = Path(tempfile.mkdtemp()) / "check.db"
    os.environ["SQLALCHEMY_DATABASE_URL"] = "sqlite+aiosqlite:///" + tmp.as_posix()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")              # SECRET_KEY etc.; does not override the temp DB
    from fastapi.testclient import TestClient
    from app.main import app
    user = f"check{int(time.time())}"
    q = "What is the FastAPI equivalent of Flask blueprints?"
    steps = []
    with TestClient(app) as c:
        for url in ["/", "/auth/login", "/auth/register/email"]:
            r = c.get(url)
            need(r.status_code == 200, f"GET {url} -> {r.status_code}")
        steps.append("pages")
        r = c.post("/auth/register/email", data={"email": f"{user}@example.com"})
        need(r.status_code == 200 and "/auth/register/details" in str(r.url), f"register email -> {r.status_code} {r.url}")
        r = c.post(f"/auth/register/details?email={user}@example.com", data={"username": user, "password": "pass12345"})
        need(r.status_code == 200 and "/auth/onboarding" in str(r.url), f"register details -> {r.status_code} {r.url}")
        steps.append("register")
        need('value="Flask"' in r.text and 'value="FastAPI"' in r.text,
             "onboarding dropdowns are empty (techstack table not seeded) - is app/main.py the new version?")
        r = c.post("/auth/onboarding", data={"primary_tech_stack": "Flask", "learning_tech_stack": "FastAPI"})
        need(r.status_code == 200 and "/user/dashboard" in str(r.url), f"onboarding -> {r.status_code} {r.url}")
        steps.append("onboarding")
        r = c.post("/user/chat", data={"question": q})
        need(r.status_code == 200, f"chat -> {r.status_code}: {r.text[:200]}")
        j = r.json()
        need(j["mode"] == "mapping" and j["route"]["src"] == "flask" and j["route"]["tgt"] == "fastapi",
             f"wrong routing: {j['mode']} {j['route']}")
        need(len(j["sources"]) >= 3 and j["response"], "no sources / empty answer")
        steps.append("chat")
        ids = [s["id"] for s in j["sources"]]
        r = c.post("/user/refine", json={"question": q, "relevant": [ids[-1]], "non_relevant": [ids[0]]})
        need(r.status_code == 200 and ids[0] not in [s["id"] for s in r.json()["sources"]], f"refine -> {r.status_code}")
        steps.append("refine")
        r = c.get("/user/dashboard")
        need(r.status_code == 200 and q[:20] in r.text, "dashboard / chat history")
        steps.append("dashboard+history")
        c.cookies.clear()
        r = c.post("/auth/login", data={"username": user, "password": "wrong"})
        need(r.status_code == 401, f"wrong password should give 401, got {r.status_code}")
        r = c.post("/auth/login", data={"username": user, "password": "pass12345"})
        need(r.status_code == 200 and "/user/dashboard" in str(r.url), f"login -> {r.status_code} {r.url}")
        steps.append("login")
    return "(" + " > ".join(steps) + ")"


# ---------------------------------------------------------------- 7. experiments (optional)
if "--full" in sys.argv:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for i, mod in enumerate(["eval.pool", "eval.run part1", "eval.alignment", "eval.run part3", "eval.feedback"], 11):
        @check(f"{i}. Experiment: python -m {mod}")
        def _(mod=mod):
            r = subprocess.run([sys.executable, "-m", *mod.split()], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=env)
            out = r.stdout + r.stderr
            if r.returncode != 0 and "refusing to overwrite judge sheets" in out:
                return "(skipped: judge sheets already contain grades - good)"
            need(r.returncode == 0, out[-1500:])
            need("gold patterns match nothing" not in out, "some queries have no gold answers in this crawl")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 70)
failed = [r for r in RESULTS if not r[1]]
for name, ok, msg in RESULTS:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print("=" * 70)
print("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED - paste this output to get help")
sys.exit(1 if failed else 0)
