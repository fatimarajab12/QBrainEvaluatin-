# QBrain rag_lab (Jupyter + CLI)

Research workspace for QBrain RAG experiments:

- Ingestion/chunking/indexing (FAISS + OpenAI embeddings)
- Retrieval + generation
- Structured benchmark (retrieval, generation, end-to-end)
- Paper-ready outputs (tables/figures)
- Optional RAGAS secondary evaluation

## 1) Project structure

```text
QBrain/rag_lab/
├── pyproject.toml
├── .env
├── src/qbrain_rag/
│   ├── config/            # settings from environment
│   ├── infrastructure/    # document loaders, embeddings, vector store, LLM
│   ├── application/       # chunking, feature extraction, evaluation utilities
│   └── services/          # RAGService facade
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
│   ├── srs/               # SRS files (.pdf/.txt)
│   ├── ground_truth/retrieval/
│   ├── benchmarks/rag_eval/questions/
│   └── outputs/evaluation/rag_benchmark/
└── results/
    ├── tables/
    └── figures/
```

## 2) Setup

```powershell
cd D:\Qbrainpython\QBrain\rag_lab
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
copy .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

Optional environment variables:

- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `RAG_TOP_K` (default: `5`)
- `GENERATION_TEMPERATURE` (default from config)

## 3) Core pipeline

The implemented pipeline is:

1. Load SRS file (`.pdf`, `.txt`)
2. Chunk text (`chunk_size=2000`, `chunk_overlap=300`)
3. Embed chunks (`text-embedding-3-small`)
4. Build FAISS vector store + metadata
5. Retrieve top-k chunks
6. Generate answer (`gpt-4o-mini`)
7. Evaluate against ground truth

## 4) Quick CLI usage

```powershell
# stage 1: ingestion
python scripts/run_ingestion.py data/srs/your_file.pdf

# stage 2-3: indexing + retrieval
python scripts/run_retrieval.py --doc data/srs/your_file.pdf --query "Your question" -k 5

# feature extraction + test cases
python scripts/run_document_pipeline.py data/srs/your_file.pdf --max-features 5

# optional index sanity check
python scripts/verify_rag_index.py
```

## 5) Benchmark and evaluation (all question files)

Run full benchmark on all `retrieval_ground_truth*.json` files:

```powershell
python scripts/run_rag_benchmark.py --k 5 --threshold 0.72 --max-questions-per-srs 10
```

Run with threshold sweep (recommended for paper analysis):

```powershell
python scripts/run_rag_benchmark.py --k 5 --threshold 0.72 --max-questions-per-srs 10 --threshold-sweep 0.65,0.68,0.70,0.72,0.75
```

Single ground-truth file evaluation:

```powershell
python scripts/run_ground_truth_eval.py --gt data/ground_truth/retrieval/retrieval_ground_truth.json -k 5 --threshold 0.72
```

## 6) Output files (what to use in paper)

### Benchmark outputs

`data/outputs/evaluation/rag_benchmark/`

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

### Paper-ready assets

`results/tables/` and `results/figures/`

- From notebook `09_*`: overall/per-SRS tables + figures
- From notebook `10_*`: RAGAS tables

## 7) Notebook workflow for paper

1. Run full benchmark:
   - `scripts/run_rag_benchmark.py` with your final settings
2. Open:
   - `notebooks/09_paper_retrieval_generation_eval.ipynb`
   - Exports core benchmark tables/plots
3. Open:
   - `notebooks/10_ragas_eval.ipynb`
   - Exports RAGAS-based secondary evaluation

## 8) RAGAS notes

- Install once if needed:

```powershell
python -m pip install ragas datasets langchain-openai -U
```

- `10_ragas_eval.ipynb` expects benchmark output with `retrieved_contexts_json` (already produced by current `run_rag_benchmark.py`).
- RAGAS may print warnings like:
  - deprecation notices
  - "LLM returned 1 generations instead of requested 3"
  These are not fatal; evaluation still completes.

## 9) Ground truth files

- Benchmark set folder:
  - `data/benchmarks/rag_eval/questions/retrieval_ground_truth*.json`
- Single-file eval default:
  - `data/ground_truth/retrieval/retrieval_ground_truth.json`

Keep `srs_file` values exactly matching filenames in `data/srs/`.

## 10) Suggested citation sentence (experimental settings)

> We evaluate QBrain on five SRS documents using manually curated retrieval QA ground truth. We report retrieval (Hit@k/Precision@k/Recall@k/MRR), generation accuracy via semantic similarity thresholds, end-to-end success, and complementary RAGAS metrics.

## 11) Prototype demo video

- Full prototype walkthrough (upload, analysis, and QA generation):
  [Watch on Google Drive](https://drive.google.com/file/d/1irk-1AWHVviviPdNAYdpFCI1CAQXnGHE/view?usp=sharing)

## 12) Detailed evaluation explanation (what exactly is evaluated)

This section explains exactly what is evaluated, on which files/questions, how metrics are computed, and where every result is saved.

### A) Evaluation scope (files and questions)

`run_rag_benchmark.py` reads all files matching:

- `data/benchmarks/rag_eval/questions/retrieval_ground_truth*.json`

Current benchmark set includes:

- `retrieval_ground_truth_ertms.json` (10 questions) -> `2007 - ertms.pdf`
- `retrieval_ground_truth_keepass.json` (10 questions) -> `2008 - keepass.pdf`
- `retrieval_ground_truth_inventory.json` (10 questions) -> `2009 - inventory 2.0.pdf`
- `retrieval_ground_truth_gparted.json` (5 questions) -> `2010 - gparted.pdf`
- `retrieval_ground_truth.json` (10 questions) -> `JDECo_SRS.docx[1].pdf`

Total in full run: **45 questions** over **5 SRS files**.

### B) What happens for each question

For every question item in ground truth:

1. Load target SRS from `data/srs/<srs_file>`
2. Chunk + embed + index in FAISS
3. Retrieve top-k chunks (default `k=5`)
4. Generate answer with LLM (`gpt-4o-mini`)
5. Compare generated answer with `expected_answer` using semantic similarity
6. Mark `gen_correct` using threshold (for example `0.72`)

### C) Metrics that are reported

#### Retrieval metrics

- `hit_at_k`: whether relevant file appears in top-k
- `precision_at_k`: retrieved relevant / k
- `recall_at_k`: retrieved relevant / number of relevant files
- `mrr`: reciprocal rank of first relevant retrieval

#### Generation metrics

- `similarity`: semantic similarity between expected vs generated answer
- `gen_correct`: `similarity >= threshold`
- `generation_accuracy`: average of `gen_correct`
- `generation_avg_similarity`: average similarity over all questions

#### End-to-end metrics

- `e2e_success`: retrieval success AND generation correctness
- `e2e_success_rate`: average success across questions
- `failure_type`: `pass`, `generation_fail`, or `retrieval_fail`

### D) Threshold sweep (calibration only)

When `--threshold-sweep` is enabled, the pipeline recomputes generation/e2e success for multiple thresholds **without changing model or retrieval pipeline**.

Use this for analysis:

```powershell
python scripts/run_rag_benchmark.py --k 5 --threshold 0.72 --max-questions-per-srs 10 --threshold-sweep 0.65,0.68,0.70,0.72,0.75
```

### E) Where each result is saved

Main folder:

- `data/outputs/evaluation/rag_benchmark/`

Important files:

- `overall_summary.csv` -> top-level final metrics
- `benchmark_full_results.csv` -> full per-question raw output
- `threshold_sweep.csv` -> threshold sensitivity analysis

Subfolders:

- `retrieval/retrieval_by_question.csv` -> retrieval details per question
- `retrieval/retrieval_summary_by_srs.csv` -> retrieval averages per SRS
- `generation/generation_by_question.csv` -> generated answers + similarity per question
- `generation/generation_summary_by_srs.csv` -> generation averages per SRS
- `e2e/e2e_by_question.csv` -> end-to-end status per question
- `e2e/e2e_summary_by_srs.csv` -> end-to-end averages per SRS
- `diagnostics/failure_breakdown.csv` -> counts of pass/failure types
- `diagnostics/metrics_by_category.csv` -> category-level summary

### F) RAGAS evaluation (secondary evaluation)

Notebook:

- `notebooks/10_ragas_eval.ipynb`

Input:

- `data/outputs/evaluation/rag_benchmark/benchmark_full_results.csv`

Output:

- `results/tables/ragas_by_question.csv`
- `results/tables/ragas_overall_summary.csv`
- `results/tables/ragas_summary_by_srs.csv`
- `results/tables/ragas_overall_summary.tex`
- `results/tables/ragas_summary_by_srs.tex`

RAGAS metrics used:

- `faithfulness`
- `answer_relevancy`
- `answer_correctness`

### G) Recommended reading order for paper writing

1. Start with `overall_summary.csv`
2. Use `e2e_summary_by_srs.csv` for per-document table
3. Use `failure_breakdown.csv` for diagnostics paragraph
4. Use `threshold_sweep.csv` for calibration figure/table
5. Add RAGAS summaries from `results/tables/` as complementary validation
