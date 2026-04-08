# QBrain RAG Lab

`QBrain/rag_lab` is the research and evaluation workspace for QBrain's Retrieval-Augmented Generation (RAG) pipeline.
It is designed to answer one practical question: **how reliably can we turn SRS documents into grounded QA outputs?**

This repository includes:

- document ingestion and chunking
- vector indexing (FAISS + embeddings)
- retrieval + answer generation
- structured benchmark evaluation
- paper-ready tables and figures
- optional RAGAS-based secondary evaluation

## Demo

- Prototype walkthrough video: [Watch on Google Drive](https://drive.google.com/file/d/1irk-1AWHVviviPdNAYdpFCI1CAQXnGHE/view?usp=sharing)

## Repository layout

```text
QBrain/rag_lab/
├── pyproject.toml
├── requirements.txt
├── .env
├── src/qbrain_rag/
│   ├── config/
│   ├── infrastructure/
│   ├── application/
│   └── services/
├── scripts/
│   ├── run_ingestion.py
│   ├── run_retrieval.py
│   ├── run_document_pipeline.py
│   ├── run_ground_truth_eval.py
│   ├── run_rag_benchmark.py
│   └── verify_rag_index.py
├── notebooks/
│   ├── 01_rag_stage1_ingestion.ipynb
│   ├── 02_rag_stage2_indexing.ipynb
│   ├── 03_rag_stage3_retrieval.ipynb
│   ├── 04_rag_stage4_generation.ipynb
│   ├── 05_document_pipeline.ipynb
│   ├── 06_evaluation_metrics.ipynb
│   ├── 08_rag_benchmark_suite.ipynb
│   ├── 09_paper_retrieval_generation_eval.ipynb
│   └── 10_ragas_eval.ipynb
├── data/
│   ├── srs/
│   ├── benchmarks/rag_eval/questions/
│   ├── ground_truth/retrieval/
│   └── outputs/evaluation/rag_benchmark/
└── results/
    ├── tables/
    └── figures/
```

## Setup

```powershell
cd D:\Qbrainpython\QBrain\rag_lab
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Configure `.env` in `QBrain/rag_lab`:

- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `RAG_TOP_K` (optional, default `5`)
- `GENERATION_TEMPERATURE` (optional)

## Pipeline overview

For each SRS evaluation run:

1. Load document (`.pdf`/`.txt`)
2. Chunk text (`chunk_size=2000`, `chunk_overlap=300`)
3. Create embeddings (`text-embedding-3-small`)
4. Build FAISS index
5. Retrieve top-k chunks
6. Generate answer (`gpt-4o-mini`)
7. Compare generated output to ground truth

### RAG step-by-step details

For each question, the system does the following:

1. Reads the target SRS file from `QBrain/rag_lab/data/srs/`.
2. Splits the document into overlapping chunks (`2000` size, `300` overlap).
3. Converts each chunk into an embedding using `text-embedding-3-small`.
4. Stores embeddings in FAISS with metadata (`source_file`, `chunk_id`).
5. Embeds the question and retrieves top-k relevant chunks (`k=5` by default).
6. Injects retrieved chunks into the prompt context.
7. Generates an answer with `gpt-4o-mini`.
8. Compares generated answer with `expected_answer` using semantic similarity.
9. Records retrieval, generation, and end-to-end metrics.

This makes answers grounded in the actual SRS content rather than relying on the model's internal memory alone.

## Quick commands

```powershell
# ingestion preview
python scripts/run_ingestion.py data/srs/your_file.pdf

# retrieval check
python scripts/run_retrieval.py --doc data/srs/your_file.pdf --query "Your question" -k 5

# feature + test generation
python scripts/run_document_pipeline.py data/srs/your_file.pdf --max-features 5

# index sanity check
python scripts/verify_rag_index.py
```

## Benchmark evaluation

Run full benchmark across all question files:

```powershell
python scripts/run_rag_benchmark.py --k 5 --threshold 0.72 --max-questions-per-srs 10
```

Run threshold sweep (recommended for analysis):

```powershell
python scripts/run_rag_benchmark.py --k 5 --threshold 0.72 --max-questions-per-srs 10 --threshold-sweep 0.65,0.68,0.70,0.72,0.75
```

Run a single ground-truth file:

```powershell
python scripts/run_ground_truth_eval.py --gt data/ground_truth/retrieval/retrieval_ground_truth.json -k 5 --threshold 0.72
```

## Evaluation scope

`run_rag_benchmark.py` reads:

- `data/benchmarks/rag_eval/questions/retrieval_ground_truth*.json`

Current full benchmark includes:

- `retrieval_ground_truth_ertms.json` -> `2007 - ertms.pdf` (10 questions)
- `retrieval_ground_truth_keepass.json` -> `2008 - keepass.pdf` (10 questions)
- `retrieval_ground_truth_inventory.json` -> `2009 - inventory 2.0.pdf` (10 questions)
- `retrieval_ground_truth_gparted.json` -> `2010 - gparted.pdf` (5 questions)
- `retrieval_ground_truth.json` -> `JDECo_SRS.docx[1].pdf` (10 questions)

Total full run: **45 questions** on **5 SRS files**.

## Metrics reported

### Retrieval

- `hit_at_k`
- `precision_at_k`
- `recall_at_k`
- `mrr`

### Generation

- semantic `similarity` (expected vs generated)
- `gen_correct` based on threshold
- `generation_accuracy`
- `generation_avg_similarity`

### End-to-end

- `e2e_success` (retrieval + generation pass)
- `e2e_success_rate`
- `failure_type` (`pass`, `generation_fail`, `retrieval_fail`)

## Output files

Main output directory:

- `data/outputs/evaluation/rag_benchmark/`

Key files:

- `overall_summary.csv`
- `benchmark_full_results.csv`
- `threshold_sweep.csv` (if enabled)
- `retrieval/retrieval_by_question.csv`
- `retrieval/retrieval_summary_by_srs.csv`
- `generation/generation_by_question.csv`
- `generation/generation_summary_by_srs.csv`
- `e2e/e2e_by_question.csv`
- `e2e/e2e_summary_by_srs.csv`
- `diagnostics/failure_breakdown.csv`
- `diagnostics/metrics_by_category.csv`

## Paper workflow

1. Run benchmark (`run_rag_benchmark.py`)
2. Open `notebooks/09_paper_retrieval_generation_eval.ipynb`
   - exports benchmark tables and figures
3. Open `notebooks/10_ragas_eval.ipynb`
   - exports RAGAS tables

Generated paper assets:

- `results/tables/`
- `results/figures/`

## RAGAS (optional secondary evaluation)

Install dependencies if needed:

```powershell
python -m pip install ragas datasets langchain-openai -U
```

RAGAS notebook:

- `notebooks/10_ragas_eval.ipynb`

Input:

- `data/outputs/evaluation/rag_benchmark/benchmark_full_results.csv`

Outputs:

- `results/tables/ragas_by_question.csv`
- `results/tables/ragas_overall_summary.csv`
- `results/tables/ragas_summary_by_srs.csv`
- `results/tables/ragas_overall_summary.tex`
- `results/tables/ragas_summary_by_srs.tex`

Notes:

- Warnings such as deprecation notices or "LLM returned 1 generations..." may appear.
- These warnings are non-fatal as long as files are produced successfully.

## Ground truth references

- Multi-file benchmark set:
  - `data/benchmarks/rag_eval/questions/retrieval_ground_truth*.json`
- Single-file default:
  - `data/ground_truth/retrieval/retrieval_ground_truth.json`

Ensure each `srs_file` value exactly matches the filename present in `data/srs/`.

## Latest run results (benchmark + RAGAS)

The following results come from a full benchmark run on:

- 5 SRS files
- 45 total questions
- `k=5`
- baseline threshold `0.72`
- threshold sweep `0.65,0.68,0.70,0.72,0.75`

Command used:

```powershell
python scripts/run_rag_benchmark.py --k 5 --threshold 0.72 --max-questions-per-srs 10 --threshold-sweep 0.65,0.68,0.70,0.72,0.75
```

### Benchmark summary

- `retrieval_hit@5`: **1.000**
- `retrieval_precision@5`: **0.200**
- `retrieval_recall@5`: **1.000**
- `retrieval_mrr`: **1.000**
- `generation_accuracy_sim>=0.72`: **0.6889**
- `generation_avg_similarity`: **0.7850**
- `e2e_success_rate`: **0.6889**

Benchmark summary table:

a~~ ` | Metric | Value |
|---|---:|
| SRS files | 5 |
| Questions | 45 |
| Retrieval Hit@5 | 1.0000 |
| Retrieval Precision@5 | 0.2000 |
| Retrieval Recall@5 | 1.0000 |
| Retrieval MRR | 1.0000 |
| Generation Accuracy (sim >= 0.72) | 0.6889 |
| Generation Avg Similarity | 0.7850 |
| End-to-End Success Rate | 0.6889 |

### Failure breakdown

- `pass`: **31**
- `generation_fail`: **14**
- `retrieval_fail`: **0**

Failure percentages:

- pass rate: **68.89%** (31/45)
- generation failure rate: **31.11%** (14/45)
- retrieval failure rate: **0.00%** (0/45)

Per-SRS end-to-end results:

| SRS file | Questions | Retrieval Hit Rate | Generation Accuracy | E2E Success |
|---|---:|---:|---:|---:|
| `2007 - ertms.pdf` | 10 | 1.00 | 0.90 | 0.90 |
| `2008 - keepass.pdf` | 10 | 1.00 | 0.70 | 0.70 |
| `2009 - inventory 2.0.pdf` | 10 | 1.00 | 0.60 | 0.60 |
| `2010 - gparted.pdf` | 5 | 1.00 | 0.60 | 0.60 |
| `JDECo_SRS.docx[1].pdf` | 10 | 1.00 | 0.60 | 0.60 |

### Threshold sweep (generation/e2e success)

- `0.65` -> **0.8667**
- `0.68` -> **0.8222**
- `0.70` -> **0.7778**
- `0.72` -> **0.6889**
- `0.75` -> **0.6222**

Threshold sweep table:

| Threshold | Generation Accuracy | E2E Success | Generation Fail Count | Pass Count |
|---:|---:|---:|---:|---:|
| 0.65 | 0.8667 | 0.8667 | 6 | 39 |
| 0.68 | 0.8222 | 0.8222 | 8 | 37 |
| 0.70 | 0.7778 | 0.7778 | 10 | 35 |
| 0.72 | 0.6889 | 0.6889 | 14 | 31 |
| 0.75 | 0.6222 | 0.6222 | 17 | 28 |

### RAGAS summary (full run, 45 questions)

- `faithfulness`: **0.9410**
- `answer_relevancy`: **0.8240**
- `answer_correctness`: **0.6693**

RAGAS summary table:

| RAGAS metric | Value |
|---|---:|
| Faithfulness | 0.9410 |
| Answer Relevancy | 0.8240 |
| Answer Correctness | 0.6693 |

Result locations:

- Benchmark CSVs: `QBrain/rag_lab/data/outputs/evaluation/rag_benchmark/`
- RAGAS tables: `QBrain/rag_lab/results/tables/`
- Figures/tables for paper: `QBrain/rag_lab/results/figures/` and `QBrain/rag_lab/results/tables/`

## RAG vs No-RAG (artifact comparison)

This comparison evaluates the document pipeline outputs (features shared, test cases compared per feature):

- Target SRS: `data/srs/JDECo_SRS.docx[1].pdf`
- Notebook: `QBrain/rag_lab/notebooks/11_compare_rag_vs_no_rag.ipynb`
- Mode:
  - RAG: semantic top-k retrieval per feature
  - No-RAG baseline: `random_k_matched` (same `k` and context budget, but no semantic retrieval)
- Run config:
  - `top_k=5`
  - `context_budget=24000`
  - `temperature=0.0`
  - `features_compared=9`

### Comparison summary

- `testcases_total`: **50** (RAG) vs **43** (No-RAG)
- `schema_valid_rate`: **1.000** vs **1.000**
- `avg_completeness`: **0.9133** vs **0.9070**
- `avg_steps`: **4.24** vs **4.19**
- `steps_ge_3_rate`: **1.000** vs **0.9767**
- Feature-level wins by testcase count: **RAG 6**, **No-RAG 0**, **Ties 3**

### Output files

- Raw side-by-side output: `QBrain/rag_lab/results/tables/rag_vs_no_rag/rag_vs_no_rag_raw.json`
- Run metadata: `QBrain/rag_lab/results/tables/rag_vs_no_rag/rag_vs_no_rag_run_meta.json`
- Aggregate table: `QBrain/rag_lab/results/tables/rag_vs_no_rag/rag_vs_no_rag_summary.csv`
- Per-feature table: `QBrain/rag_lab/results/tables/rag_vs_no_rag/rag_vs_no_rag_per_feature.csv`
