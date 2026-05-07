# RAG Pipeline for AI Research Papers

A Retrieval-Augmented Generation (RAG) pipeline that answers questions over a corpus of 75 AI/ML research papers from arXiv. The system compares two chunking strategies, three retrieval methods, and four prompt variants, then evaluates them with the RAGAS framework.

> Course project — Team 5: Lam Tran, Mahima Goyal, Yining Mao

## Overview

The pipeline runs in five stages:

1. **PDF Parsing** — extract text page-by-page with PyMuPDF; extract tables separately with pdfplumber and convert them to Markdown.
2. **Chunking** — two strategies: a recursive splitter (`size=512, overlap=64`) and a section-aware splitter (`size=1200, overlap=100`) that first splits on section headers via regex.
3. **Embedding & Indexing** — `all-MiniLM-L6-v2` (384-dim) embeddings, indexed in two ChromaDB collections (one per chunking strategy).
4. **Retrieval** — three methods: cosine similarity (baseline), hybrid BM25 + dense via Reciprocal Rank Fusion, and HyDE (hypothetical-answer expansion).
5. **Generation** — top-K chunks are formatted into a labeled context block and passed to GPT-4o-mini with strict grounding/citation instructions.

All intermediate artifacts (parsed documents, chunks, ChromaDB snapshots, evaluation outputs) are pickled to disk so each stage can be resumed without recomputation.

## Architecture

```
                75 arXiv PDFs
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   PyMuPDF (fitz)             pdfplumber
   page-by-page text       table → Markdown
        └─────────────┬─────────────┘
                      ▼
            ParsedDocument objects
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  Recursive chunks          Section-aware chunks
   (512 / 64, 1848)          (1200 / 100, 843)
        └─────────────┬─────────────┘
                      ▼
        all-MiniLM-L6-v2  →  384-dim vectors
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   ChromaDB: rag_recursive   ChromaDB: rag_section_aware
        └─────────────┬─────────────┘
                      ▼
              User query
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
    Cosine      Hybrid (BM25+RRF)   HyDE
       └──────────────┼──────────────┘
                      ▼
        Top-K chunks → labeled context block
                      ▼
                GPT-4o-mini
                      ▼
          Grounded answer + [Source N] citations
```

## Repository Structure

```
.
├── rag pipeline.ipynb        # End-to-end pipeline notebook
├── requirements.txt          # Pinned dependencies
├── outputs/                  # Pickled intermediate artifacts + result CSVs
│   ├── chunks_rec.pkl
│   ├── chunks_sec.pkl
│   ├── collection_rec.pkl
│   ├── collection_sec.pkl
│   ├── documents.pkl
│   ├── results.csv
│   └── hit_rate_results.csv
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key (used for generation and the HyDE step only — embeddings run locally)

### Install

```bash
git clone https://github.com/<your-username>/rag-pipeline-ai-papers.git
cd rag-pipeline-ai-papers
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and add your key:

```bash
OPENAI_API_KEY=sk-...
```

## Usage

Open the notebook and run cells top-to-bottom:

```bash
jupyter notebook "rag pipeline.ipynb"
```

The notebook exposes the main building blocks as functions you can call directly:

| Stage | Function |
|-------|----------|
| Parse a PDF | `parse_pdf(pdf_path)` |
| Recursive chunking | `chunk_recursive(documents)` |
| Section-aware chunking | `chunk_section_aware(documents)` |
| Build a Chroma collection | `build_vector_store(chunks, collection_name, model)` |
| Cosine retrieval | `retrieve_cosine(query, collection, model, k=5)` |
| Hybrid BM25 + dense | `retrieve_hybrid(query, collection, bm25, chunks, model, k=5)` |
| HyDE retrieval | `retrieve_hyde(query, collection, model, client, k=5)` |
| End-to-end answer | `answer(query, collection, model, client)` |
| Full evaluation grid | `run_full_qa_grid(...)` |

Cached artifacts in `outputs/` let you skip parsing and embedding on subsequent runs.

## Configuration

| Hyperparameter | Value |
|---|---|
| Recursive chunk size / overlap | 512 / 64 chars |
| Section-aware chunk size / overlap | 1200 / 100 chars |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) |
| Vector store | ChromaDB (in-memory) |
| Generator | `gpt-4o-mini` |
| Top-K | 5 |
| RRF α | 0.5 |
| RAGAS eval sample | 50 queries |

## Results

Evaluation uses RAGAS (faithfulness, answer relevancy, context precision, context recall, answer correctness) plus a deterministic Hit Rate @ 3 against ground-truth qrels. Scores are broken down by chunking × retrieval × query modality.

Headline numbers from `outputs/results.csv`:

| Chunking | Retrieval | Modality | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness | Hit@3 |
|---|---|---|---|---|---|---|---|---|
| Recursive | Cosine | text | 0.95 | 0.88 | 0.84 | 0.65 | 0.60 | 0.16 |
| Recursive | Cosine | text-image | 0.84 | 0.84 | 0.76 | 0.68 | 0.53 | 0.21 |
| Recursive | Cosine | text-table | 0.83 | 0.89 | 0.83 | 1.00 | 0.55 | 0.25 |
| Recursive | Hybrid | text | 0.86 | 0.86 | 0.80 | 0.57 | 0.59 | 0.16 |
| Recursive | Hybrid | text-image | 0.81 | 0.66 | 0.69 | 0.33 | 0.43 | 0.22 |
| Section-aware | Cosine | text | 0.92 | 0.91 | 0.78 | 0.81 | 0.57 | 0.16 |
| Section-aware | Cosine | text-image | 0.95 | 0.87 | 0.68 | 0.65 | 0.47 | 0.21 |
| **Section-aware** | **Cosine** | **text-table** | **0.96** | **0.88** | **0.96** | **0.83** | **0.70** | **0.25** |
| Section-aware | Hybrid | text | 0.81 | 0.81 | 0.81 | 0.66 | 0.61 | 0.17 |

**Best overall configuration:** Section-aware chunking + Cosine retrieval. It achieves the strongest balance across faithfulness, answer relevancy, context precision, and answer correctness, especially on text-table queries.

**Critical weakness:** Hit Rate @ 3 sits between 0.13 and 0.25 across every configuration — the retriever surfaces the correct document in the top-3 only 13–25% of the time, which caps how high answer correctness can go regardless of generation quality.

## Key Findings

- **Faithfulness is consistently high** (0.69–1.00) thanks to the grounded prompt. The system rarely hallucinates *given* its retrieved context — but it often retrieves the wrong context.
- **Cosine > Hybrid** on this corpus. BM25 introduces noise on semantically driven academic queries; the hybrid leg drops answer relevancy from ~0.90 (cosine) to 0.64–0.66 on text-table and text-image queries.
- **Section-aware > Recursive** for table-heavy queries. Preserving section boundaries keeps tables anchored to their explanatory text, lifting answer correctness on text-table queries from 0.55 → 0.70.
- **The minimal prompt never refuses** to answer, even when context is insufficient. Explicit grounding + refusal instructions are required to keep the model honest.

## Error Modes

Four representative failures are documented in the writeup:

1. **Table parsing failure** — correct chunk retrieved, but the LLM misreads a flat-Markdown table and picks the wrong row ($5 vs. $2.5).
2. **Under-reading retrieved context** — the answer is present in the chunk, but the model hedges with "not detailed in the provided context."
3. **Wrong-section retrieval** — Introduction-section chunk retrieved instead of the Methods section, producing a plausible but wrong answer.
4. **Hallucination from off-topic chunk** — thematically adjacent chunk (governance vs. adversarial robustness) gets retrieved and the model confabulates.

Proposed mitigations (also in the writeup): query rewriting + decomposition before retrieval, a retrieved-evidence sufficiency check before generation, and a post-generation faithfulness verifier that grounds each claim back to a specific span.

## Limitations & Future Work

- **Embeddings.** `all-MiniLM-L6-v2` is fast and local but compresses semantics; larger models (`text-embedding-3-small`, BGE-large) would likely close the semantic-gap failures.
- **Static fusion.** RRF weights BM25 and dense equally; query-dependent weighting or a learned fusion would help.
- **Tables.** Flat Markdown linearization breaks row/column structure. Treating tables as first-class units (with caption + repeated headers) or using TAPAS/TAPEX would address this.
- **Modality.** Images and equations are out of scope. Future work: caption-and-index figures, swap PyMuPDF for an equation-aware parser like Mathpix or GROBID.

## Tech Stack

PyMuPDF · pdfplumber · LangChain text splitters · sentence-transformers · ChromaDB · rank_bm25 · OpenAI Python SDK · RAGAS · NumPy · pandas

## Team

- Lam Tran
- Mahima Goyal
- Yining Mao

## License

MIT — see [LICENSE](LICENSE).
