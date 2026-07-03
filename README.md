# mlx_rag

A local, multimodal Retrieval-Augmented Generation (RAG) system for a personal
scientific-literature library. It parses PDFs and Office files — **including figures,
three-line tables, and scanned pages** — into a searchable index, then answers
questions over them with citations.

Embeddings and reranking run **locally** — MLX on Apple Silicon, sentence-transformers
(torch) on Windows/Linux (GPU-accelerated when available); the chat/vision LLM runs
through any **OpenAI-compatible** provider (Qwen by default), switchable in one line.

## Features

- **Multimodal parsing** — text, tables, and figures. Figures are described by a
  vision model; borderless "three-line" tables are read by a table-OCR model;
  fully **scanned PDFs** (no text layer) are transcribed page-by-page.
- **Robust figure/table handling** — figure captions on any side (above / below /
  beside) and across pages; table captions above *or* below the table.
- **Hybrid retrieval** — dense vectors (FAISS) + lexical BM25, fused with
  Reciprocal Rank Fusion, then a cross-encoder reranker and a Kneedle cutoff.
- **Gene-alias expansion** — queries mentioning a gene symbol also match its aliases
  (optional; degrades gracefully if unavailable).
- **Map-reduce QA with multi-turn follow-ups** — per-chunk extraction then synthesis;
  follow-up questions ("what about the second one?") are rewritten into standalone queries.
- **One-line provider switching** — edit `models.yaml` (`qwen` / `openai` / `claude` / `gemini`).
- **VL result cache** — every vision call is cached on disk, so re-parsing/re-chunking is nearly free.

## Architecture

```
   data/*.pdf,docx,pptx,xlsx,csv
              │
   ingest.py  ▼
     parse  ──►  text  +  VL(figures) + OCR(tables / scanned pages)
              │
     chunk  ──►  embed (bge-m3, local)  ──►  FAISS index  →  index_chunk{SIZE}/
              (vision results cached in vl_cache/)

   question
              │
   chat.py    ▼
     rewrite follow-up (multi-turn)  ──►  hybrid retrieve (FAISS + BM25 → RRF)
              │
     rerank (bge-reranker, local) → Kneedle cutoff
              │
     map-reduce over chunks (chat LLM)  ──►  grounded answer + sources
```

## Prerequisites

- **Python 3.11**.
- **Platform** — runs on macOS, Windows, or Linux. The embedding backend is chosen
  automatically (`EMBED_BACKEND=auto`):
  - **Apple Silicon** → MLX (native acceleration);
  - **Windows / Linux / GPU server** → sentence-transformers (torch; uses CUDA automatically
    if a GPU is present). This path needs no extra install — `sentence-transformers` is already
    a dependency (the reranker uses it), and `pip` skips the Apple-only `mlx-embeddings` via a
    platform marker.
- **Local models** — `bge-m3` (embeddings) and `bge-reranker-v2-m3` (reranker). By default they
  are auto-downloaded from HuggingFace on first run; set `EMBED_MODEL` / `RERANKER_MODEL` (env)
  to point at local copies instead.
- **An API key** for your chosen chat/vision provider (default: DashScope International for Qwen).

## Installation

```bash
# 1. clone, then create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. download the local models and set their paths in config.py
#    (EMBED_MODEL / RERANKER_MODEL)

# 4. configure your API key
cp .env.example .env        # then edit .env and fill in your key(s)
```

## Usage

```bash
# 1. put your documents in data/  (PDF / DOCX / PPTX / XLSX / CSV)

# 2. build the index  (first run calls the vision model per figure/table; cached afterwards)
python ingest.py

# 3. ask questions
python chat.py
```

Example:

```
Question: What chromosome is LGMD linked to?
[answer, grounded in the retrieved papers, with a Sources list]
```

## Configuration

- **`models.yaml`** — pick the provider/models. Set `active` to `qwen` / `openai` /
  `claude` / `gemini`, or use `llm_override` / `vl_override` per role. The Qwen block
  also defines `vl_model` (figure understanding) and `ocr_model` (scanned pages / tables).
- **`config.py`** — local model paths, `CHUNK_SIZE` (also decides which
  `index_chunk{SIZE}/` folder is used), `CHUNK_OVERLAP`, retrieval `CANDIDATE_K`, etc.
- **`.env`** — API keys (never committed; see `.env.example`).

## Project structure

```
config.py            # paths, chunking params, provider resolution
models.yaml          # provider/model registry (one-line switching)
model_client.py      # unified chat() / describe_image() + VL disk cache
ingest.py            # parse → chunk → embed → FAISS index
query.py             # hybrid retrieval (FAISS + BM25 + RRF) + rerank + cutoff
chat.py              # map-reduce QA + multi-turn query rewriting (REPL)
hgnc.py              # gene-alias query expansion (optional)
loaders/             # per-format parsers (pdf, docx, pptx, excel, txt, office images)
```

Regenerable/large artifacts (not committed): `.venv/`, `index_chunk*/`, `vl_cache/`,
`db/`, `data/`.

## How the vision routing works

Different jobs go to different models (edit in `models.yaml`):

| Job | Model (default) | Why |
| --- | --- | --- |
| Figure understanding | `qwen-vl-max` | reads/interprets charts, blots, microscopy |
| Scanned pages & tables | `qwen-vl-ocr` | faithful text/table transcription, cheaper |
| Chat / synthesis | `qwen-plus` | text-only reasoning |

Vision results are cached in `vl_cache/` keyed by image + prompt + model, so the
expensive vision passes are paid once.

## Known limitations

- **Silent-wrong figure descriptions** aren't caught (the correctness self-check was
  dropped to save cost); a confidently wrong description can enter the index.
- **Narrow tables in two-column layouts** may over-grab an adjacent column's width.
- **Undecodable/broken embedded images** cannot be rendered; such figures fall back to
  caption-only.
- Multi-turn history is used only to rewrite the query, not passed to the final answer step.

## Notes

- This is a personal research tool; source PDFs are not included (copyright + size).
- Costs are dominated by the one-time vision pass at ingest; the disk cache makes
  subsequent re-indexing (e.g. changing `CHUNK_SIZE`) nearly free.

## License

MIT — see [LICENSE](LICENSE).
