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

## Why these models

Every model in the stack is picked for a specific role, and each choice has a direct alternative we considered.

### `BAAI/bge-m3` — dense embeddings

- **Multilingual by design.** The corpus is a mix of Vietnamese and English VCBS reports, and users ask questions in either language (sometimes code-switched). BGE-M3 is trained on 100+ languages and does well on cross-lingual retrieval, so a Vietnamese query against an English-only report (or vice versa) still lands in the right neighborhood. English-first models like `text-embedding-3-small` or `all-MiniLM-L6-v2` would silently degrade on the Vietnamese half.
- **Long context (8192 tokens).** Report descriptions and PDF-extracted passages are longer than the 512-token ceiling of most sentence-transformer models. BGE-M3 ingests the whole chunk without truncation.
- **1024 dims is a sweet spot.** Dense enough to be expressive, small enough that ~1k reports × a few chunks each fit in a NumPy array and run cosine similarity in milliseconds on a free T4. No vector DB needed.
- **Open weights, fp16-friendly.** Runs locally on Colab's free T4 in fp16 with no API cost and no rate limits. Cached as a single `.npy` file keyed by a chunk hash, so re-runs are instant.
- **Alternatives rejected:** Gemini / OpenAI embedding APIs (per-call cost, rate limits, no offline eval); monolingual Vietnamese models like `vinai/phobert-base` (strong on Vietnamese alone but weak on English and on retrieval tasks specifically).

### `rank_bm25` BM25Okapi — lexical half of hybrid search

- **Tickers and codes are tokens, not semantics.** Users search for `HPG`, `VCB`, `MWG`, `TPDN` — short exact strings where dense similarity is unreliable. BM25 nails exact-match retrieval for these without any training.
- **Free, deterministic, zero-dependency.** `rank_bm25` is pure Python; no index server, no tuning.
- **Why fuse with dense instead of replacing?** The ablation shows hybrid fusion gives the single biggest MRR lift (+28% over dense-only). BM25 alone misses paraphrases ("steel outlook" ↔ "Hoa Phat industry analysis"); dense alone misses ticker-exact queries. Min-max normalization + `α = 0.4` leans slightly toward dense while keeping BM25's precision on short-code queries.

### `BAAI/bge-reranker-v2-m3` — cross-encoder reranker

- **Same family as the embedder.** Trained on the same multilingual data distribution as BGE-M3, so the two models' notions of relevance don't conflict.
- **Cross-encoder quality on a shortlist, bi-encoder speed in retrieval.** The reranker scores each (query, doc) pair jointly — much more accurate than cosine similarity — but is too expensive to run over the whole corpus. We only call it on the top 20 fused candidates and keep the top 10.
- **Honest about the tradeoff.** The eval shows the reranker adds modest recall / hit-rate gains at ~4x the latency, with a slight MRR regression on this corpus. We keep it because the recall gain matters more than MRR for an agentic flow that reads several docs, but the ablation table makes the cost visible rather than hiding it.

### Gemini (via `google-genai`) — agent / answer model

- **Native function calling with a clean schema.** The agent exposes one tool, `search_reports(query)`, and Gemini decides turn-by-turn whether to call it, how to rephrase, and when to stop. The `types.FunctionDeclaration` API matches what we need without an extra orchestration framework (LangChain / LlamaIndex would be overkill for one tool).
- **Strong Vietnamese generation.** Final answers need to come back in whichever language the user asked. Gemini handles Vietnamese output more naturally than most open models at comparable cost.
- **Generous free tier and large context window.** Enough context to hold a system prompt + 10 reranked docs + up to ~3k chars of PDF excerpt each, without truncation.
- **Alternatives rejected:**
  - GPT-4 / Claude: better quality but a hard spend floor for a prototype that re-runs the eval set repeatedly.
  - A fully local LLM (Llama, Qwen) on Colab: VRAM budget already spent on BGE-M3 + reranker; nothing left for a 7B+ model at useful quality.
  - OpenRouter / Nemotron: client is kept in the file (commented out) as a drop-in alternative — swap `userdata.get('gemini-key')` for `userdata.get('openrouter-key')` to A/B.

### `pdf-inspector` — PDF text extraction

- **Markdown output, not raw text.** Preserves table structure and headings, which matters for financial reports full of numeric tables.
- **Byte-stream API.** We download the PDF ourselves, cache the extracted text to disk keyed by filename, and never re-parse. Pairs cleanly with the `is_clean_text` guard that rejects extractor junk (OCR garbage, binary residue) before it reaches the LLM.

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
