# DevBuddy AI

**A Dual-Track Information Retrieval System for Cross-Stack Developer Upskilling and Trustworthy Answer Generation**

[Repository](https://github.com/aasthamlhtra/DevBuddy)

---

## Project Overview

Modern software engineering moves at a rapid pace, forcing developers to continuously pick up new programming languages and frameworks. When learning a new stack, developers rarely start from scratch—they already understand core architectural concepts like routing, dependency injection, and request lifecycles. However, traditional documentation forces them to re-read beginner tutorials, while ungrounded LLMs frequently invent non-existent parameters or sever code blocks mid-syntax.

**DevBuddy AI** bridges this gap as a **cross-language concept-mapping engine**. Instead of relying on ungrounded generative AI, DevBuddy retrieves official documentation chunks from a polite, deduplicated index and maps technical primitives side-by-side (e.g., translating Flask's thread-local `g` object directly to FastAPI's dependency injection system `Depends()`).

This project integrates principles from two distinct Information Retrieval (IR) research tracks:

- **Track 1 (T1): Retrieval-Augmented Generation (RAG) and Trustworthy Answers**
- **Track 4 (T4): Web Crawling, Freshness, and Web Integrity**

---

## Key Features and IR Alignment

### Track 4: Web Crawling and Integrity Architecture

- **Focused Crawler with Politeness:** Parses and respects host `robots.txt` specifications, enforces per-domain rate-limiting (\(t_{\text{delay}} \ge 1.0\text{s}\)), and normalizes URLs to avoid spider traps and repeated crawling.
- **Content-Seen Check (SHA-256):** Computes cryptographic fingerprints over stripped main DOM bodies (`<main>`, `<article>`) to eliminate exact duplicate pages before indexing.
- **Near-Duplicate Detection (Jaccard over k-Shingles):** Filters out cross-version documentation redundancy using Jaccard shingle similarity thresholds (\(\tau = 0.85\)).

### Track 1: Inspectable RAG and Grounded Retrieval

- **Header-Aware Structural Passaging:** Replaces arbitrary character/token slicing with heading-bound (`#`, `##`, `###`) document chunking, keeping code blocks (`<pre><code>`) permanently attached to their explanatory paragraphs.
- **Parametric Zone Indexing:** Annotates document chunks into **Title Zone**, **Code Zone**, and **Body Zone** to apply weighted relevance scoring based on query intent.
- **Maximal Marginal Relevance (MMR) Re-Ranking:** Re-ranks candidate passages (\(\lambda = 0.7\)) to balance query relevance with content diversity, preventing redundant chunks from wasting LLM prompt context.
- **100% Citation Lineage:** Stores relational metadata linking every database record to its exact source URL, guaranteeing verifiable output citations.

---

## System Architecture

```text
                           [ Official Docs Seeds ]
                                      |
                                      v  (Robots.txt & Rate Limits)
                           [ Focused Web Crawler ]
                                      |
                                      v  (SHA-256 & Jaccard Shingles)
                           [ Integrity & Deduplication ]
                                      |
                                      v  (Header-Aware Passaging)
                           [ Parametric Zone Store ]
                                      |
[ User Query ] ---> [ Hybrid Retriever ] ---> [ MMR Re-Ranker ] ---> [ Grounded LLM Output ]
```

---

## Benchmark and Evaluation Results

All results are produced by code in this repository (`bash scripts/reproduce.sh`) and written to
`eval/results/`. See **[EXTENSIONS.md](EXTENSIONS.md)** for the full tables, significance tests and caveats.
The current numbers use provisional judgments; the pooled human-judging workflow is in `eval/`.

| Query type (test, nDCG@10) | BM25F | Dense (LSA) | Hybrid RRF |
| :--- | :---: | :---: | :---: |
| Exact API names | 0.709 | 0.574 | 0.699 |
| How-to questions | 0.458 | 0.399 | 0.455 |
| Cross-framework "equivalent of" | 0.146 | 0.190 | 0.162 |

---

## Repository Structure

```text
DevBuddy/
├── app/                      # FastAPI web app (auth, onboarding, dashboard, /user/chat, /user/refine)
├── ir/                       # IR pipeline
│   ├── crawler.py, extract.py   # polite crawler, dedup, chunking (step 0)
│   ├── text.py, index.py        # code-aware tokenizer, zoned inverted index, BM25 / BM25F (part 1)
│   ├── dense.py                 # MiniLM / LSA dense retrieval, reciprocal rank fusion
│   ├── cluster.py               # k-means / HAC clustering, concept alignment (part 2)
│   ├── links.py                 # PageRank, personalised PageRank, HITS (part 3)
│   ├── feedback.py              # Rocchio + pseudo-relevance feedback (part 4)
│   └── engine.py, answer.py     # all systems behind one API; cited answer composition
├── eval/                     # queries, concept pairs, pooling, judging, kappa, metrics, experiments
├── data/                     # crawled pages.jsonl / chunks.jsonl (+ raw HTML)
├── tools/                    # offline docs mirror for environments without access to the live sites
├── tests/                    # unit tests (PageRank vs networkx, kappa vs sklearn, BM25 formula, ...)
├── scripts/reproduce.sh      # rebuild every result
└── requirements.txt
```

---

## Installation and Setup Guide

### Prerequisites

- Python **3.10+**
- `pip` package manager
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/aasthamlhtra/DevBuddy.git
cd DevBuddy
```

### 2. Create and Activate a Virtual Environment

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and configure your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
PORT=8000
SQLALCHEMY_DATABASE_URL=sqlite+aiosqlite:///./devbuddy.db
SECRET_KEY=change-me
ENCODING_ALGORITHM=HS256
GROQ_API_KEY=your_groq_api_key_here   # optional; without it answers are extractive
```

### 5. Run Data Ingestion and Crawling Pipeline

To crawl the official FastAPI and Flask documentation (robots.txt, 1 s per host, about 4 minutes):

```bash
python -m ir.crawler --out data
bash scripts/reproduce.sh        # optional: tests + all experiments
```

### 6. Start the Web Server

Launch the application using Uvicorn:

```bash
uvicorn app.main:app --reload --port 8000
```

### 7. Access the Application

Open your browser and navigate to:

[http://localhost:8000/user/dashboard](http://localhost:8000/user/dashboard)

---

## Team Members and Contributions

| Name | Roll Number | Key Contribution Area |
| :--- | :--- | :--- |
| Aastha Malhotra | 2310110007 | Track 4 Architecture: Focused crawler, robots.txt politeness scheduler, DOM boilerplate removal, and SHA-256/Jaccard shingle deduplication. |
| Mitaksh Goswami | 2410110875 | Track 1 Processing & Storage: SQLite relational metadata schema, header-aware structural passaging algorithm, and parametric zone indexing. |
| Pranav Talwar | 2310110555 | Retrieval Optimization & Evaluation: Maximal Marginal Relevance (MMR) re-ranking module, benchmark evaluation (\(P@3\), MRR), and repository management. |

---

## AI Use Declaration

In compliance with academic integrity guidelines, Large Language Models (LLMs) were utilized as assistive tools during development for:

- Generating initial boilerplate routines for DOM parsing and SQLite database connections.
- Assisting with LaTeX report markup formatting and prose polish.
- Drafting video screen-recording voiceover scripts.

All Information Retrieval algorithms, crawling rules, zone weighting strategies, MMR re-ranking code, and benchmark evaluations were independently designed, implemented, and verified by the authors.
