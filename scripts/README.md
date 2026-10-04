# Generative AI Platform & Text Data Analysis

A local generative AI platform that processes **50,000 unstructured news articles**, enables **semantic search** by meaning, and uses **LLM pipelines (LangChain + LangGraph)** to answer questions and extract structured information, exposed through a **REST API**.

## Problem
- Keyword search misses documents that use different words ("layoffs" vs "job cuts")
- Reading thousands of files manually is impossible
- Key information (company, event, sentiment) is buried in raw text

## Solution
- **Semantic search** over embedded text chunks
- **RAG answers** grounded only in retrieved articles
- **Structured extraction** into validated JSON
- **REST API** so any app can use the platform

## Architecture
```
AG News (Hugging Face)
  → build_dataset.py   50,000 .txt files
  → ingest.py          load, clean, chunk → chunks.jsonl
  → index.py           embeddings → Chroma vector DB
  → graph.py           LangGraph: route → retrieve → answer | extract → validate → retry
  → api.py             FastAPI REST endpoints
```

## Tech stack
| Layer | Tools |
|---|---|
| Language | Python 3.13 |
| Data | Hugging Face `datasets` (AG News) |
| Ingestion | LangChain loaders + RecursiveCharacterTextSplitter |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector DB | Chroma |
| LLM | Llama 3.2 via Ollama (local, free) |
| Workflow | LangGraph |
| API | FastAPI + Uvicorn |
| Testing | PyTest + FastAPI TestClient |
| Version control | Git / GitHub |

## LangGraph workflow
| Node | Role |
|---|---|
| `route` | Detects request type: question or extraction |
| `retrieve` | Fetches the most relevant chunks from Chroma |
| `generate_answer` | Answers using only retrieved context (RAG) |
| `extract_info` | Returns structured JSON via a Pydantic schema |
| `check_extraction` | Validates output, retries up to 2 times |

## API endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service status |
| POST | `/search` | Top-k semantic search results with distance scores |
| POST | `/ask` | Grounded answer + source document IDs |
| POST | `/extract` | Company, event, sentiment, category as JSON |

Each response includes `latency_ms` for performance monitoring.

## Results (baseline)
| Metric | Value |
|---|---|
| Retrieval label precision@5 (100 queries) | 78.4% (random baseline: 25%) |
| Avg search latency | 2464 ms |
| Extraction category accuracy (20 docs) | 45.0% (random baseline: 25%) |
| Extraction failures (invalid JSON) | 0 |
| Avg extraction latency | 5492 ms |
| Tests | 5 passed |

Evaluation uses a fixed random seed so results are comparable across changes.

## Engineering decisions
- **Text cleaning** of broken HTML codes and whitespace before embedding
- **Metadata preserved** through chunking (doc ID, label, source) for traceability and evaluation
- **Batch indexing** (1,000 chunks per batch) with stable IDs
- **Structured output** with `Literal` fields to block invented values
- **Validation + retry loop** to avoid crashes on bad LLM output
- **Grounded prompting**: the LLM must answer from retrieved context only
- **Local LLM** (Ollama) for zero API cost and data privacy

## Known limitations & next steps
- Current vector index uses a **2,000-article sample** for faster iteration; full 50k indexing is supported by the pipeline
- **Search latency**: cache the embedding model once across the app (`lru_cache`)
- **Extraction accuracy**: add category definitions to the schema; test a larger LLM
- **Deduplication**: the news wire contains near-duplicate stories
- Add a **reranker** to improve retrieval precision

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2

python scripts/build_dataset.py
python src/ingest.py
python src/index.py
uvicorn api:app --reload --app-dir src
```
Then open http://127.0.0.1:8000/docs

## Tests & evaluation
```bash
pytest -v
python scripts/evaluate.py
```

## Project structure
```
genai-text-platform/
├── scripts/
│   ├── build_dataset.py
│   └── evaluate.py
├── src/
│   ├── ingest.py
│   ├── index.py
│   ├── search.py
│   ├── graph.py
│   └── api.py
├── tests/
│   ├── test_ingest.py
│   └── test_api.py
├── requirements.txt
├── pytest.ini
└── README.md
```