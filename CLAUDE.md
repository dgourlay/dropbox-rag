# CLAUDE.md

## Project

Local RAG system that indexes documents from configured filesystem folders, builds a hybrid search index (dense vectors + keyword), and exposes retrieval via MCP to local LLM tools (Claude Desktop, Claude Code). Single Python process + Qdrant Docker container. No cloud infrastructure required.

## Plans

- Spec: `plan/local/local-rag-spec.md`
- Agent team plan: `plan/local/agent-team-plan.md`
- Spec review: `plan/local/spec-review.md`
- Cloud variant (future): `plan/cloud/dropbox-rag-final-spec.md`

## Architecture

Single Python process handles: filesystem watching (watchdog) → Docling parsing (in subprocess for memory isolation) → normalization → dedup → chunking → embedding (local BGE-M3, 1024-dim) → summarization (LLM CLI) → Qdrant indexing. Summarization shells out to a user-configured LLM CLI tool (claude, kiro-cli, etc.) and stores document_summary and section_summary vectors in Qdrant. MCP server runs via stdio for Claude Desktop or Streamable HTTP. MCP handlers are async; CPU-bound ops dispatched via asyncio.to_thread(). Retrieval pipeline: 3-lane prefetch (document_summary top 20, section_summary top 20, chunks top 30) → RRF fusion with layer weighting (broad/specific query classification) → recency boost (90-day half-life, max 30% influence) → ONNX cross-encoder reranker → cited evidence returned to calling LLM.

## Tech Stack

- Python 3.11+, Pydantic v2, SQLite (WAL mode), Qdrant v1.17 (Docker)
- Docling (parsing, OcrAutoOptions), sentence-transformers/BGE-M3 (embeddings), onnxruntime (reranker)
- MCP SDK `mcp>=1.25,<2` (stdio + Streamable HTTP transport)

## Typing & Code Quality (Mandatory)

- **mypy strict mode** — `strict = true`, `plugins = ["pydantic.mypy"]`. Full annotations. No `Any` except behind typed wrappers.
- **Pydantic v2 models** at every module boundary. No raw dicts crossing boundaries (except within typed wrapper functions around external clients).
- **dataclasses** (`frozen=True, slots=True`) for stage-internal value objects only. Cross-stage data uses Pydantic.
- **Literal** for small field-annotation sets (ProcessStatus, SummaryLevel). **StrEnum** for sets used in iteration/runtime logic (RecordType, FileType).
- **Protocol classes** for all pluggable backends (Embedder, Summarizer, MetadataDB, VectorStore, Reranker, Parser).
- **Discriminated union Result types** for fallible operations — no `success: bool` + optional fields pattern.
- **ruff** for linting and formatting (ANN, TC rule sets enabled). Note: `TCH` was renamed to `TC` in ruff v0.8.0.
- **Async/sync**: indexing pipeline is sync. MCP handlers are async. CPU-bound ops via `asyncio.to_thread()`. Qdrant queries via `AsyncQdrantClient`.

- **`from __future__ import annotations`** in every module. Use `model_rebuild()` for cross-module Pydantic forward refs.

See `plan/local/local-rag-spec.md` §2.1 for full typing rules and examples.

## Project Structure

```
src/rag/
  types.py             # All Pydantic models, StrEnums, Literals, type aliases
  protocols.py         # All Protocol classes (Embedder, Summarizer, MetadataDB, VectorStore, Reranker, Parser)
  results.py           # Discriminated union Result types (ParseResult, SummaryResult, etc.)
  cli.py               # Entry points: rag init, rag index, rag serve, rag watch, rag status, rag search
  config.py            # TOML config loader (AppConfig Pydantic model)
  init.py              # Interactive setup wizard (rag init)
  sync/                # Filesystem watcher + scanner
  pipeline/            # classify → parse → normalize → dedup → chunk → embed → summarize → index
    parser/            # Docling wrapper (PDF/DOCX, subprocess) + text fallback (TXT/MD)
    summarizer.py      # CliSummarizer: LLM CLI for doc/section summaries (step 12)
  retrieval/           # 3-lane prefetch + RRF fusion + layer weighting + recency boost + reranker + citations
  mcp/                 # MCP server (async) + tool definitions (5 tools)
  db/                  # SQLite connection + Qdrant client, models, migrations
migrations/            # SQL schema files
tests/                 # Unit tests per module
tests/e2e/             # End-to-end tests (real Qdrant, real models, real MCP server)
tests/fixtures/        # Real sample documents with known searchable content
```

## Key Commands

```bash
rag init                          # Interactive setup wizard (folders, LLM CLI, Qdrant, MCP)
rag index                         # Full scan + process all documents
rag index --reindex               # Purge all index data, re-process everything (prompts confirmation)
rag index --reindex /path/to/file # Clear + re-process a single file
rag serve                         # Start MCP server (stdio)
rag watch                         # Filesystem watcher (auto-index on changes)
rag status                        # Dashboard: docs/chunks/errors, MCP health, liveness checks
rag doctor                        # Health check: Qdrant, OCR, models, folders
rag search "query"                # CLI search for testing
rag search "query" --debug        # Search with per-lane counts, weights, timing
rag search "query" --top-k 5      # Limit number of results
rag mcp-config --print            # Print MCP config JSON snippet
```

## Conventions

- Target chunk size: 512 tokens (tiktoken cl100k_base), 64-token overlap
- Embedding dimensions: 1024 (BGE-M3)
- Qdrant: single collection "documents", cosine distance, record_type payload field, all search via `query_points()` API (not removed `search()`)
- Qdrant indexing: deterministic UUID5 point IDs for overwrite semantics (no delete+upsert)
- UUID5 namespace: `NAMESPACE_RAG` constant, format `f"{doc_id}:{section_order}:{chunk_order}"`
- UUIDs stored as TEXT in SQLite
- All timestamps ISO 8601
- SQLite: WAL mode, busy_timeout=30000
- Config file: TOML at `~/.config/dropbox-rag/config.toml`
- Docling parsing runs in child subprocess (multiprocessing) for memory isolation
- MCP stdio servers: never write to stdout (corrupts JSON-RPC); use stderr for logging
- SQLite: `check_same_thread=False` for async MCP handler access
- Retrieval: 3-lane prefetch by record_type, layer weighting by query classification, recency boost (90-day half-life)
- MCP tools: 5 tools (search_documents, quick_search, get_document_context, list_recent_documents, get_sync_status)
- search_documents `format` parameter: "text" (default, LLM-friendly) or "json" (raw structured data)

## Testing

- `make lint` — ruff check + mypy strict
- `make test` — unit tests (fast, no Docker)
- `make test-e2e` — end-to-end with real Qdrant Docker, real BGE-M3 model, real `claude` CLI for summarization, real MCP server subprocess
- E2e tests use fixture documents with known query-answer pairs — asserts on specific content, not just "something returned"
- `make test-e2e` passing = system works. No ambiguity.
