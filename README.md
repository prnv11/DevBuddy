# DevBuddy AI

**Cross-framework documentation retrieval for Flask and FastAPI, with concept alignment, link analysis and relevance feedback**

[Repository](https://github.com/prnv11/DevBuddy) · [Project report (PDF)](DevBuddy_Report.pdf) · [Extension details and results](EXTENSIONS.md)

**Project members:** Aastha Malhotra and Pranav Talwar (CSD358 Information Retrieval, Shiv Nadar University)

---

## Project Overview

Developers who move from one web framework to another know the concept they need but not its name in the
new framework: a Flask *Blueprint* is a FastAPI *APIRouter*, `abort()` becomes `HTTPException`. Keyword
search inside one documentation site handles exact API names well, but fails on paraphrased and
cross-framework questions.

DevBuddy is a retrieval system over the official **Flask** and **FastAPI** documentation, built and
evaluated from scratch. It crawls both sites politely, indexes them with its own zoned inverted index,
compares lexical, dense and hybrid retrieval, aligns concepts across the two frameworks by clustering,
tests whether link analysis helps ranking, and learns from 👍/👎 feedback in the web dashboard. Answers
are grounded in retrieved chunks and always cite their source URLs.

---

## What the final submission contains

The first version of DevBuddy was a RAG prototype (crawler, header-aware chunking, MMR re-ranking, a
FastAPI web app). For the final submission the retrieval pipeline was rebuilt and four IR extensions
were added, each with its own experiments:

| Part | What was built | Main result (test split, nDCG@10 unless noted) |
|:---|:---|:---|
| **Step 0: Data** | Polite crawler: robots.txt, ≥ 1 s per host, URL normalisation and trap filters, SHA-256 + Jaccard 5-shingle dedup, boilerplate/template removal, heading-bounded chunks with a separate code field | 211 pages, 2,967 chunks (1,779 with code), 493 content-link edges |
| **Part 1: Retrieval + evaluation** | Code-aware tokenizer, zoned positional inverted index, BM25 / BM25F, MiniLM and LSA dense retrieval, reciprocal rank fusion; 90 queries of 3 types, pooling, kappa, P@k / MAP / MRR / nDCG, bootstrap and Wilcoxon tests | Hybrid 0.472 > BM25F 0.434 > MiniLM 0.341 (hybrid vs BM25F p = 0.005). BM25F best on exact API names, MiniLM best on cross-framework questions |
| **Part 2: Concept alignment** | LSA / MiniLM spaces, spherical k-means and HAC, centroid and Hungarian matching, cluster pruning, no-match threshold; 30 gold Flask↔FastAPI pairs | hit@3 0.75 vs 0.60 for the BM25F baseline (not significant at n = 20, p = 0.36); pruning to 5 of 60 clusters keeps hit@10 |
| **Part 3: Link analysis** | PageRank, personalised PageRank (global and per cluster), query-time HITS, as a prior mixed into the score with weight β | Tuning sets β = 0: no reliable gain on this small, sparse link graph (negative result) |
| **Part 4: Relevance feedback** | Rocchio on terms and embeddings, gated pseudo-relevance feedback, residual-collection evaluation; 👍/👎 and *Refine* in the dashboard | Explicit Rocchio +0.08 to +0.16 for every system (p ≤ 0.007); ungated PRF slightly hurts |

The web app was also fixed and connected to the real pipeline: `/user/chat` now returns answers built
from retrieved chunks with their sources, and `/user/refine` re-ranks them using the user's feedback.

> The numbers use provisional judgments (answer patterns written with each query). The pooled
> two-judge human-judging workflow is included in `eval/`; see [EXTENSIONS.md](EXTENSIONS.md).

---

## System Architecture

```text
        [ flask.palletsprojects.com    fastapi.tiangolo.com ]
                              |
                              v   robots.txt, 1 s per host, URL filters
                     [ Polite crawler ]  -->  raw HTML cache
                              |
                              v   SHA-256 + Jaccard dedup, template removal
                     [ Chunker: heading path + text + code + links ]
                              |
          +-------------------+--------------------+------------------+
          v                   v                    v                  v
 [ Zoned inverted index ] [ Dense vectors ]  [ Link graph ]     [ Concept clusters ]
   BM25 / BM25F            MiniLM / LSA        PageRank / HITS    k-means / HAC
          |                   |                    |                  |
          +------> [ Hybrid ranking (RRF) + optional link prior ] <---+  (Flask <-> FastAPI mapping)
                              |
                              v
 [ User query ] -> [ Ranked chunks ] -> [ Cited answer (Groq LLM or extractive) ]
                              ^                    |
                              +--- Rocchio refine <-- 👍 / 👎 in the dashboard
```

---

## Repository Structure

```text
DevBuddy/
├── app/                      # FastAPI web app: auth, onboarding, dashboard, /user/chat, /user/refine
├── ir/                       # IR pipeline
│   ├── crawler.py, extract.py   # polite crawler, dedup, template removal, chunking (step 0)
│   ├── text.py, index.py        # code-aware tokenizer, zoned inverted index, BM25 / BM25F (part 1)
│   ├── dense.py                 # MiniLM / LSA dense retrieval, reciprocal rank fusion (part 1)
│   ├── cluster.py               # k-means / HAC clustering, concept alignment (part 2)
│   ├── links.py                 # PageRank, personalised PageRank, HITS (part 3)
│   ├── feedback.py              # Rocchio + pseudo-relevance feedback (part 4)
│   └── engine.py, answer.py     # all systems behind one API; cited answer composition
├── eval/                     # queries, concept pairs, pooling, judging, kappa, metrics, experiments
│   └── results/              # generated tables (part1-4.md), JSON and figures
├── data/                     # crawled pages.jsonl and chunks.jsonl (the dataset)
├── ingestion/                # original prototype's ingestion scripts
├── scripts/                  # check_all.py (end-to-end check), reproduce.sh (rebuild every result)
├── tests/                    # unit tests (PageRank vs networkx, kappa vs scikit-learn, BM25 formula, ...)
├── tools/                    # offline docs mirror for machines that cannot reach the live sites
├── DevBuddy_Report.pdf       # final project report
├── EXTENSIONS.md             # full results, significance tests and caveats
└── requirements.txt
```

---

## Installation and Setup

### Prerequisites

- Python **3.10+** (tested on 3.13 and 3.14)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/prnv11/DevBuddy.git
cd DevBuddy
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` (`copy .env.example .env` on Windows) and edit it:

```dotenv
PORT=8000
SQLALCHEMY_DATABASE_URL=sqlite+aiosqlite:///./devbuddy.db
SECRET_KEY=change-me
ENCODING_ALGORITHM=HS256
GROQ_API_KEY=your_groq_api_key_here   # optional; without it answers are extractive
```

### 5. Data

The crawled dataset is already in `data/`. To crawl again from the live sites (about 4 minutes):

```bash
python -m ir.crawler --out data
```

### 6. Check everything and reproduce the results

```bash
python scripts/check_all.py          # tests, index, retrieval, alignment, web app flow
python scripts/check_all.py --full   # also re-runs every experiment into eval/results/
```

### 7. Start the web app

```bash
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000), register, choose Flask/FastAPI, and ask questions
such as *"FastAPI equivalent of Flask Blueprint"*. Use 👍/👎 on the sources and press **Refine**.

---

## Team Members and Contributions

| Name | Roll Number | Key Contribution Area |
| :--- | :--- | :--- |
| Aastha Malhotra | 2310110007 | Track 4 Architecture: Focused crawler, robots.txt politeness scheduler, DOM boilerplate removal, and SHA-256/Jaccard shingle deduplication. |
| Pranav Talwar | 2310110555 | Retrieval Optimization & Evaluation: Maximal Marginal Relevance (MMR) re-ranking module, benchmark evaluation (\(P@3\), MRR), and repository management. |

---

## AI Use Declaration

In compliance with academic integrity guidelines, Large Language Models (LLMs) were used as assistive tools for:

- Generating initial boilerplate routines for DOM parsing and SQLite database connections.
- Assisting with LaTeX report markup formatting and prose polish.
- Drafting video screen-recording voiceover scripts.
- Implementing the final-submission extensions (crawler, retrieval and evaluation, concept alignment,
  link analysis, relevance feedback), the app fixes, and drafting the project report, using Claude
  (Anthropic) as a coding assistant.

The authors specified the extensions, ran every experiment on their own machines, and checked the
code and results; the numbers reported come from running the code in this repository.
