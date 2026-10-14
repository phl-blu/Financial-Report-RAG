# Financial Report RAG

An agentic Retrieval-Augmented Generation (RAG) system over Vietcombank Securities (VCBS) financial reports. The pipeline ingests reports from the VCBS public API, builds a hybrid lexical + dense index, reranks with a cross-encoder, and lets Gemini drive multi-turn, tool-calling search. A Gradio UI exposes the full flow with a collapsible "thinking" panel.

Bilingual: supports both English and Vietnamese queries and reports.

## Pipeline

```
VCBS API ──► paginate (en + vi) ──► filter requiredLogin=false ──► dedup ──► clean HTML ──► normalize
                                                                                      │
                                                                                      ▼
                                                                            vcbs_reports.json
                                                                                      │
      ┌───────────────────────────────────────────────────────────────────────────────┘
      ▼
  chunk (500 chars, 50 overlap)
      │
      ├──► BGE-M3 dense embeddings (1024-dim, fp16) ──► cached .npy
      └──► BM25Okapi lexical index
                     │
                     ▼
           hybrid fusion (α=0.4, min-max normalized)
                     │
                     ▼
         BGE reranker-v2-m3 (top 20 → top 10)
                     │
                     ▼
   Gemini agent (function calling, dual-query merge, time-filter metadata)
                     │
                     ▼
               Gradio UI
```

## Repository layout

| File | What it does |
|---|---|
| `data_report_ingestion.ipynb` / `data_report_ingestion.py` | Pulls reports from the VCBS API, filters public ones, deduplicates, cleans HTML, writes `vcbs_reports.json`. |
| `LiveDataRAG.ipynb` / `livedatarag.py` | Loads the normalized reports, builds dense + BM25 indices, runs the hybrid + reranker pipeline, serves Gradio, and runs the eval suite. |

Both scripts are exports of Colab notebooks and expect `/content/drive/MyDrive/RAG_Project` as the working directory.

## Components

- **Data source**: `https://www.vcbs.com.vn/api/v1/ttpt-reports` (paginated, locale = `en` / `vi`, filtered to `requiredLogin == false`).
- **Chunking**: fixed-size, 500 characters with 50-character overlap, over `name | symbol | category | description`.
- **Embeddings**: [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) (multilingual, 1024-dim, fp16 on T4). Cached per chunk-set hash.
- **Lexical search**: `rank_bm25` BM25Okapi over the same chunks.
- **Fusion**: min-max normalize both score vectors, then `alpha * bm25 + (1 - alpha) * dense` with `alpha = 0.4`. Chunk scores aggregate to the report level by max.
- **Reranker**: [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) cross-encoder on the top 20 → top 10.
- **PDF enrichment**: for top hits whose description is short, download the PDF, extract with `pdf-inspector`, cache under `pdf_cache/`, and splice up to 3000 characters per doc into the LLM context.
- **Agent**: Gemini with a single `search_reports` function tool. Decides whether to search, rephrases the query, and the pipeline runs a *dual-query merge* (original + rephrased) before reranking.
- **Time filter**: lightweight parser for "today / this week / this month / latest", "March 2026", "08/2026", and their Vietnamese equivalents, applied to `publishedAt`.
- **UI**: Gradio Blocks app with a question box, a collapsible "Thinking — Retrieved Documents" panel, and the final Markdown answer.

## Running it

The notebooks are built for Google Colab (free T4 is enough). To run:

1. Open `data_report_ingestion.ipynb` in Colab and execute end-to-end to produce `vcbs_reports.json` in your Drive at `/MyDrive/RAG_Project`.
2. Open `LiveDataRAG.ipynb`. Add a Colab Secret named `gemini-key` holding your [Gemini API key](https://aistudio.google.com/apikey), then run all cells. The last Gradio cell launches the UI (`demo.launch(debug=True)`).

Dependencies are installed inline in the notebooks:

```
pdf-inspector  gradio  rank_bm25  FlagEmbedding  google-genai
```

Secrets are read via Colab's `userdata.get(...)`; no keys are checked in.

### Optional: OpenRouter / Nemotron

An OpenRouter client is present but commented out in `livedatarag.py`. To use it instead of Gemini, add a Colab Secret `openrouter-key` and uncomment that block.

## Evaluation

The notebook ships with a mixed hand-written + auto-generated eval set (~65 queries across `easy / medium / hard / ambiguous / auto` difficulties), and runs four retrieval modes side by side:

- `bm25_only`
- `dense_only`
- `hybrid`
- `hybrid_reranker`

Metrics implemented: **Recall@K, Precision@K, MRR, NDCG@K, Hit Rate@K** (K = 10). An error-analysis pass distinguishes *Bad Ranking* (relevant doc exists in top 50 but outside top 10) from *Missing Document* (not retrieved at all), and the project report cell renders a summary with difficulty breakdowns.

Headline finding: hybrid fusion is the single biggest quality jump over any single-method baseline (MRR +28% over dense-only). Reranking adds modest recall / hit-rate at ~4x the latency.

## Known limitations

- LLM-as-judge groundedness / answer-correctness eval not yet wired up.
- No multi-run experiment tracking.
- Metadata filtering covers time only; category / locale filtering is TODO.
- Hand-written eval ground truth is keyword-sourced, not manually verified — some automated-label noise exists (e.g. substring collisions like `VCB` ↔ `VCBS`).
- Everything is Colab-first; there is no pip package or standalone CLI yet.

## License

Not specified.
