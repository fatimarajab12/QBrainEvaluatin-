# QBrain RAG Lab

`rag_lab` is an API-first Retrieval-Augmented Generation (RAG) system focused on SRS (Software Requirements Specification) documents.
It is designed for:

- document upload and project-scoped vector indexing
- retrieval quality evaluation
- feature extraction from indexed knowledge
- test case generation from extracted features
- notebook-based experimentation and reporting

---

## Table of Contents

- [1. Technology Stack](#1-technology-stack)
- [2. Architecture and Code Organization](#2-architecture-and-code-organization)
- [3. End-to-End Workflow](#3-end-to-end-workflow)
- [4. Environment and Setup](#4-environment-and-setup)
- [5. Supabase Setup](#5-supabase-setup)
- [6. Running the API](#6-running-the-api)
- [7. API Usage Guide](#7-api-usage-guide)
- [8. Evaluation Guide](#8-evaluation-guide)
- [9. Metrics Explained](#9-metrics-explained)
- [10. Data and File Conventions](#10-data-and-file-conventions)
- [11. Troubleshooting](#11-troubleshooting)

---

## 1. Technology Stack

- **Language**: Python 3.11+
- **API framework**: FastAPI
- **ASGI server**: Uvicorn
- **Data validation**: Pydantic
- **LLM and embeddings**: OpenAI APIs
- **Persistence and vector DB**: Supabase
- **Interactive analysis**: Jupyter Notebook

### Core persistence tables (Supabase)

- `projects`
- `project_vectors`
- `features`
- `test_cases`
- optional related metrics tables used by API services

---

## 2. Architecture and Code Organization

```text
rag_lab/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   ├── routes/schemes/
│   │   ├── controllers/
│   │   ├── services/
│   │   └── repositories/
│   ├── application/
│   ├── infrastructure/
│   ├── config/
│   └── services/
├── data/
│   ├── srs/
│   ├── srs/uploads/
│   └── ground_truth/
├── scripts/
├── notebooks/
├── supabase_setup.sql
├── requirements.txt
└── pyproject.toml
```

### Layer responsibilities

- `src/api/routes/`: HTTP endpoint definitions and request routing
- `src/api/routes/schemes/`: request/response models
- `src/api/controllers/`: thin endpoint orchestration
- `src/api/services/`: use-case/business logic
- `src/api/repositories/`: low-level Supabase operations
- `src/application/`: domain pipelines (chunking, extraction, evaluation logic)
- `src/infrastructure/`: OpenAI, embedding generation, vector retrieval integrations
- `src/config/`: runtime settings and defaults

---

## 3. End-to-End Workflow

### Step 1: Create project
Create a project entity in Supabase:

- `POST /api/v1/projects/`

### Step 2: Upload SRS and build knowledge source
Upload one document to a project:

- `POST /api/v1/projects/{project_id}/upload-srs`

Current behavior:
- accepts form-data key `srs` (or `file`)
- stores uploaded runtime copy under `data/srs/uploads/`
- updates project `doc_path`
- parses and chunks text
- computes embeddings
- inserts vectors into `project_vectors` (scoped by `project_id`)

### Step 3: Retrieval
Retrieve relevant chunks for a query:

- `POST /api/v1/projects/{project_id}/retrieval`

Request example:

```json
{
  "query": "What is the system purpose?",
  "k": 5
}
```

### Step 4: Feature generation
Generate features from indexed document knowledge:

- `POST /api/v1/features/projects/{project_id}/generate-features`

### Step 5: Test case generation
Generate test cases from selected features:

- `POST /api/v1/test-cases/features/{feature_id}/generate-test-cases`

---

## 4. Environment and Setup

From `D:\Qbrainpython\rag_lab`:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Create `.env` in `rag_lab`:

```text
OPENAI_API_KEY=your_api_key_here
USE_SUPABASE=true
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_SRS_BUCKET=srs-files
```

Optional settings:

- `OPENAI_MODEL`
- `RAG_TOP_K`
- `GENERATION_TEMPERATURE`
- any configuration available in `src/config/settings.py`

---

## 5. Supabase Setup

1. Open Supabase SQL Editor.
2. Run `supabase_setup.sql`.
3. Ensure storage bucket exists (`srs-files` by default).
4. Confirm `.env` points to the same project.

If Supabase credentials are invalid or missing, all project/vector operations fail.

---

## 6. Running the API

From `D:\Qbrainpython\rag_lab`:

```powershell
$env:PYTHONPATH="src"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

Docs:

- Swagger UI: `http://127.0.0.1:8001/docs`
- OpenAPI JSON: `http://127.0.0.1:8001/openapi.json`

Health check:

- `GET /api/v1/health`

---

## 7. API Usage Guide

### Core endpoint groups

- **Base**: root and health
- **Projects**: CRUD, upload, stats, project-scoped retrieval
- **Features**: generation + CRUD + project queries
- **Test Cases**: generation + CRUD + exports
- **RAG**: generic ingestion/retrieval/query/document pipeline endpoints
- **Chatbot**: context/query endpoints

### Recommended practical sequence

1. Create project
2. Upload SRS to project
3. Run project retrieval and verify relevance
4. Generate features
5. Generate test cases for selected features

### Why project-scoped retrieval matters

Using `POST /api/v1/projects/{project_id}/retrieval` ensures:
- retrieval only from that project’s vectors
- no accidental cross-project contamination
- consistent evaluation behavior

---

## 8. Evaluation Guide

### Retrieval evaluation script

Script:
- `scripts/evaluate_retrieval_api.py`

What it does:
- creates temporary project(s)
- uploads referenced SRS files
- runs retrieval for each question in ground truth
- computes `Hit@k`, `Precision@k`, `Recall@k`, `MRR`
- saves report to `results/retrieval_api_eval/report.json`

Run command:

```powershell
python scripts/evaluate_retrieval_api.py --base-url http://127.0.0.1:8001/api/v1 --k 5 --mode unified_project
```

Modes:

- `unified_project`: uploads all files into one project (best for file-level benchmark)
- `project_per_file`: one temporary project per file

### Notebook for evaluation

- `notebooks/06_retrieval_api_evaluation.executed.ipynb`

Use it to:
- run evaluation from notebook
- inspect summary metrics
- inspect per-query ranked outputs

---

## 9. Metrics Explained

- `Hit@k`: percentage of queries where relevant target appears in top-k
- `Precision@k`: relevant_in_top_k / k
- `Recall@k`: relevant_in_top_k / number_of_relevant_targets
- `MRR`: average reciprocal rank of the first relevant hit

### Important interpretation note

In file-level evaluation with one relevant file per query and `k=5`:
- max `Precision@5` is `1/5 = 0.2`

So `Precision@5 = 0.2` can still represent perfect file-level retrieval.

---

## 10. Data and File Conventions

- `data/srs/`: canonical source/reference SRS files
- `data/srs/uploads/`: runtime uploaded copies (generated by API upload)
- `data/ground_truth/`: benchmark and validation datasets
- `results/retrieval_api_eval/`: generated evaluation reports

This split keeps source data clean while preserving runtime artifacts.

---

## 11. Troubleshooting

- **All retrieval metrics are zero**
  - verify correct `base-url`/port
  - ensure vectors exist for the evaluated project(s)
  - ensure file-name matching in evaluation uses stored upload names

- **`Connection refused` during evaluation**
  - API is not running on configured host/port
  - start uvicorn and rerun

- **Supabase errors**
  - verify `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`
  - confirm schema was created using `supabase_setup.sql`

- **After clearing vectors**
  - rerun upload flow before evaluating retrieval

---

## Quick Start (Minimal)

```powershell
cd D:\Qbrainpython\rag_lab
.\.venv\Scripts\activate
$env:PYTHONPATH="src"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

Then run:

```powershell
python scripts/evaluate_retrieval_api.py --base-url http://127.0.0.1:8001/api/v1 --k 5 --mode unified_project
```
