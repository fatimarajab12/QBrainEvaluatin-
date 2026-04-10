# QBrain RAG Lab

`QBrain/rag_lab` is the research and evaluation workspace for a Retrieval-Augmented Generation (RAG) pipeline built on System Requirements Specification (SRS) documents.
The repository evaluates how effectively SRS text can be converted into grounded QA outputs using retrieval, generation, and benchmark analysis.

## What’s included

- Document ingestion and chunking
- Embedding generation and FAISS vector indexing
- Retrieval of top-k relevant chunks
- Grounded answer generation with OpenAI
- Structured benchmark evaluation and diagnostics
- Notebook-driven analysis and reports

## Repository layout

```text
QBrain/rag_lab/
├── pyproject.toml
├── requirements.txt
├── .env
├── src/qbrain_rag/
│   ├── application/
│   ├── config/
│   ├── infrastructure/
│   └── services/
├── scripts/
│   ├── clear_vector_cache.py
│   ├── generate_rag_benchmark_report.py
│   ├── run_document_pipeline.py
│   ├── run_document_pipeline_batch.py
│   ├── run_ingestion.py
│   ├── run_rag_benchmark.py
│   ├── run_retrieval.py
│   └── verify_rag_index.py
├── notebooks/
│   ├── 01_rag_stage1_ingestion.executed.ipynb
│   ├── 02_rag_stage2_indexing.executed.ipynb
│   ├── 03_rag_stage3_retrieval.ipynb
│   ├── 04_rag_stage4_creation_features_tests.ipynb
│   ├── 10_rag_ragas_evaluation.ipynb
│   ├── 10_rag_ragas_evaluation.html
│   ├── 12_rag_benchmark_results_dashboard.executed.ipynb
│   └── 12_rag_benchmark_results_dashboard.executed.html
├── data/
│   ├── faiss_cache/
│   ├── ground_truth/
│   ├── outputs/
│   └── srs/
└── results/
    ├── figures/
    └── pipeline_runs/
```

## Setup

```powershell
cd D:\Qbrainpython\QBrain\rag_lab
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Create a `.env` file in `rag_lab` with at least:

```text
OPENAI_API_KEY=your_api_key_here
```

Optional values:

- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `RAG_TOP_K` (default: `5`)
- `GENERATION_TEMPERATURE`

## Core workflow

1. Ingest SRS documents from `data/srs/`
2. Chunk text with overlap
3. Embed chunks and store them in a FAISS index
4. Retrieve top-k chunks for a question
5. Generate a grounded answer using retrieved context
6. Evaluate retrieval, generation, and end-to-end metrics

## Pipeline details

- The pipeline starts by loading SRS documents from `data/srs/`.
- The document text is split into overlapping chunks to preserve context during retrieval.
- Each chunk is converted into an embedding using an OpenAI embedding model.
- Embeddings are stored in a FAISS index for fast similarity search.
- When a question is received, the question is embedded and the most relevant chunks are retrieved.
- The retrieved chunks are used as reference context to generate an answer with OpenAI.
- The final output is evaluated for retrieval quality, generation quality, and end-to-end success.

## Data and outputs

```powershell
# Preview ingestion for a document
python scripts/run_ingestion.py data/srs/your_file.pdf

# Run retrieval for a question
python scripts/run_retrieval.py --doc data/srs/your_file.pdf --query "Your question" -k 5

# Generate features/tests from a document
python scripts/run_document_pipeline.py data/srs/your_file.pdf --max-features 5

# Validate the FAISS index
python scripts/verify_rag_index.py
```

## Benchmark evaluation

Run a full benchmark across all available question files:

```powershell
python scripts/run_rag_benchmark.py --k 5 --threshold 0.72 --max-questions-per-srs 10
```

Run a threshold sweep for analysis:

```powershell
python scripts/run_rag_benchmark.py --k 5 --threshold 0.72 --max-questions-per-srs 10 --threshold-sweep 0.65,0.68,0.70,0.72,0.75
```

## Data and outputs

Input locations:

- `data/srs/` — source SRS documents
- `data/ground_truth/` — labeled retrieval and generation ground truth
- `data/faiss_cache/` — persisted FAISS indexes

Benchmark output directory:

- `data/outputs/evaluation/rag_benchmark/`

Common output files:

- `overall_summary.csv`
- `benchmark_full_results.csv`
- `retrieval/retrieval_by_question.csv`
- `generation/generation_by_question.csv`
- `e2e/e2e_by_question.csv`
- `diagnostics/failure_breakdown.csv`
- `diagnostics/metrics_by_category.csv`

## Results explanation

- `overall_summary.csv`: A high-level summary of benchmark metrics across all documents and questions.
- `benchmark_full_results.csv`: Detailed results per question, including retrieval, similarity scores, and final pass/fail status.
- `retrieval/retrieval_by_question.csv`: Retrieval accuracy and score details for each question.
- `generation/generation_by_question.csv`: Generation quality metrics for each question, including similarity and correctness.
- `e2e/e2e_by_question.csv`: End-to-end results for each question from retrieval through generation.
- `diagnostics/failure_breakdown.csv`: Breakdown of failure types such as `pass`, `generation_fail`, and `retrieval_fail`.
- `diagnostics/metrics_by_category.csv`: Performance analysis by question category or document type.

## Metrics tracked

Retrieval metrics:

- `hit_at_k`
- `precision_at_k`
- `recall_at_k`
- `mrr`

Generation metrics:

- semantic `similarity` between generated and expected answers
- `gen_correct` (threshold-based correctness)
- `generation_accuracy`
- `generation_avg_similarity`

End-to-end metrics:

- `e2e_success`
- `e2e_success_rate`
- `failure_type` (`pass`, `generation_fail`, `retrieval_fail`)

## Interpreting percentage scores

- Most benchmark scores are reported as decimals between `0` and `1`. A value like `0.78` means `78%`.
- `hit_at_k`: e.g. `0.80` means the correct chunk was found in the top-k retrieved items for 80% of the questions.
- `mrr`: mean reciprocal rank measures how early the correct chunk appears. A higher value closer to `1.0` means the correct chunk was retrieved near the top.
- `similarity`: a value of `0.78` means the generated answer is 78% similar to the expected answer according to the chosen semantic metric.
- `generation_accuracy`: a value of `0.76` means 76% of generated answers passed the generation-quality threshold.
- `e2e_success` and `e2e_success_rate`: these show the proportion of questions that succeeded across both retrieval and generation. For example, `0.78` means 78% of questions ended with a successful full pipeline result.
- `failure_type`: this explains why the pipeline failed on a question. `retrieval_fail` means the relevant context was not retrieved, while `generation_fail` means the retrieved context did not produce a correct answer.

## Notebooks and analysis

Explore the repo using the provided notebooks:

- `01_rag_stage1_ingestion.executed.ipynb`
- `02_rag_stage2_indexing.executed.ipynb`
- `03_rag_stage3_retrieval.ipynb`
- `04_rag_stage4_creation_features_tests.ipynb`
- `10_rag_ragas_evaluation.ipynb`
- `12_rag_benchmark_results_dashboard.executed.ipynb`

## Notes

- This workspace is built around OpenAI embedding and generation APIs.
- Adjust model settings in the `.env` file or `src/qbrain_rag/infrastructure/llm.py` if needed.
- Use `scripts/run_rag_benchmark.py` for full evaluation.
- Use `scripts/verify_rag_index.py` to check index integrity.
