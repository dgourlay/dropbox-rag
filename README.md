# dropbox-rag

A local RAG (Retrieval-Augmented Generation) system that indexes documents from
filesystem folders, builds a hybrid search index combining dense vectors and
keyword search, and exposes retrieval via MCP to local LLM tools like Claude
Desktop and Claude Code. Everything runs locally -- a single Python process plus
a Qdrant Docker container. No cloud infrastructure required.

This project is for developers and power users who want their local LLM tools to
have searchable, cited access to their personal document collections (PDFs, Word
docs, Markdown, plain text) without sending documents to external services.


## Features

- **Hybrid search** -- dense vector (BGE-M3) + keyword search with RRF fusion
- **ONNX cross-encoder reranking** for high-precision results
- **MCP integration** -- works as a tool for Claude Desktop and Claude Code
- **Multi-format parsing** -- PDF, DOCX, TXT, MD via Docling with OCR support
- **Filesystem watching** -- automatic re-indexing when documents change
- **Document summarization** -- shells out to any LLM CLI tool (claude, etc.)
- **Deduplication** -- content-hash based dedup across folders
- **Deterministic indexing** -- UUID5 point IDs for idempotent upserts
- **Fully local** -- all models run on-device, no API keys needed for core search


## Prerequisites

- **Python 3.11+**
- **Docker** (for Qdrant vector database)
- **4GB+ RAM** available for embedding and reranker models
- **macOS or Linux** (Windows not tested)
- An LLM CLI tool for summarization (optional) -- `claude`, `kiro-cli`, or similar


## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/dropbox-rag.git
   cd dropbox-rag
   ```

2. **Create a virtual environment and install**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Start Qdrant**

   ```bash
   docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant:v1.17.1
   ```

   Or use the included Docker Compose file:

   ```bash
   docker compose up -d
   ```

4. **Run the setup wizard**

   ```bash
   rag init
   ```

   This walks you through folder selection, LLM CLI detection, Qdrant
   connectivity check, and MCP configuration. It creates your config file at
   `~/.config/dropbox-rag/config.toml`.

5. **Download the reranker model**

   The ONNX reranker model needs to be exported manually (one-time step). The
   easiest way:

   ```bash
   make download-models
   ```

   Or run it directly:

   ```bash
   scripts/download-models.sh
   ```

   This exports the BGE reranker to ONNX format at
   `~/.cache/dropbox-rag/models/bge-reranker-v2-m3/`. The script will install
   `optimum[onnxruntime]` temporarily if needed.

   Note: The BGE-M3 embedding model (~1.5GB) downloads automatically on first
   use -- no manual step required.

6. **Index your documents**

   ```bash
   rag index
   ```

   This scans configured folders, parses documents, chunks text, generates
   embeddings, and stores everything in Qdrant. First run will download the
   BGE-M3 embedding model.

7. **Search**

   ```bash
   rag search "your query here"
   ```


## MCP Integration

The primary way to use dropbox-rag is as an MCP tool server for Claude Desktop
or Claude Code.

**View the MCP configuration:**

```bash
rag mcp-config --print
```

This prints the JSON snippet you need to add to your MCP client config.

**Auto-install into Claude Desktop or Claude Code:**

```bash
rag mcp-config --install claude-desktop
rag mcp-config --install claude-code
```

**Start the MCP server manually:**

```bash
rag serve
```

This starts the MCP stdio server. When configured as an MCP tool, Claude Desktop
or Claude Code will launch this process automatically -- you do not normally need
to run it by hand.

The MCP server exposes search, document listing, status, and summarization tools
that the calling LLM can invoke to retrieve cited evidence from your documents.


## CLI Reference

| Command | Description |
|---|---|
| `rag init` | Interactive setup wizard -- configures folders, LLM CLI, Qdrant, MCP |
| `rag index` | Full scan and process all documents in configured folders |
| `rag serve` | Start the MCP server (stdio transport) |
| `rag watch` | Filesystem watcher -- auto-indexes on document changes |
| `rag status` | Dashboard showing document/chunk/error counts per folder |
| `rag doctor` | Health check -- verifies Qdrant, OCR, models, folders |
| `rag search "query"` | CLI search for testing and debugging |
| `rag mcp-config` | Print or install MCP configuration (`--print`, `--install`) |


## Configuration

### Config file location

The config file is TOML. It is resolved in this order (first match wins):

1. Explicit path via `--config /path/to/config.toml` CLI flag
2. `RAG_CONFIG_PATH` environment variable
3. `./config.toml` in the current directory
4. `~/.config/dropbox-rag/config.toml` (default, created by `rag init`)

### Complete example

```toml
[folders]
paths = ["~/Documents", "~/Dropbox"]
extensions = ["pdf", "docx", "txt", "md"]
ignore = ["**/node_modules", "**/.git", "**/venv", "**/__pycache__"]

[database]
path = "~/.local/share/dropbox-rag/metadata.db"

[qdrant]
url = "http://localhost:6333"
collection = "documents"

[embedding]
model = "BAAI/bge-m3"
dimensions = 1024
batch_size = 32
cache_dir = "~/.cache/dropbox-rag/models"

[reranker]
model_path = "~/.cache/dropbox-rag/models/bge-reranker-v2-m3"
top_k_candidates = 30
top_k_final = 10

[summarization]
enabled = true
provider = "cli"
command = "claude"
args = ["--print", "--max-tokens", "2048"]
timeout_seconds = 60

[mcp]
transport = "stdio"
host = "127.0.0.1"
port = 8080

[watcher]
poll_interval_seconds = 5
debounce_seconds = 2
use_polling = false
batch_window_seconds = 10
```


## Architecture

A single Python process handles the full pipeline: filesystem watching
(watchdog) to Docling parsing (in a subprocess for memory isolation) to
normalization, deduplication, chunking, embedding (BGE-M3, 1024 dimensions), and
Qdrant indexing. Summarization shells out to a user-configured LLM CLI tool. The
MCP server runs via stdio for Claude Desktop or Streamable HTTP. MCP handlers
are async; CPU-bound operations are dispatched via `asyncio.to_thread()`. The
retrieval pipeline performs hybrid dense + keyword search via `query_points()`,
RRF fusion, ONNX cross-encoder reranking, and returns cited evidence to the
calling LLM.

### Project structure

```
src/rag/
  types.py             # Pydantic models, StrEnums, Literals, type aliases
  protocols.py         # Protocol classes (Embedder, Summarizer, MetadataDB, etc.)
  results.py           # Discriminated union Result types
  cli.py               # CLI entry points
  config.py            # TOML config loader (AppConfig Pydantic model)
  init.py              # Interactive setup wizard
  sync/                # Filesystem watcher + scanner
  pipeline/            # classify -> parse -> normalize -> dedup -> chunk -> embed -> summarize -> index
    parser/            # Docling wrapper (PDF/DOCX) + text fallback (TXT/MD)
  retrieval/           # Hybrid search: dense + keyword + RRF + reranker + citations
  mcp/                 # MCP server (async) + tool definitions
  db/                  # SQLite connection + Qdrant client
migrations/            # SQL schema files
tests/                 # Unit tests per module
tests/e2e/             # End-to-end tests (real Qdrant, real models, real MCP)
tests/fixtures/        # Sample documents with known searchable content
scripts/               # Helper scripts (model download, etc.)
```


## Development

```bash
# Lint and type check
make lint

# Run unit tests (fast, no Docker required)
make test

# Run end-to-end tests (requires Qdrant Docker, BGE-M3 model, claude CLI)
make test-e2e

# Run everything
make test-all

# Format code
make format
```

End-to-end tests use fixture documents with known query-answer pairs. They
assert on specific content in search results, not just "something returned."
If `make test-e2e` passes, the system works.


## Troubleshooting

**"Config file not found"**
Run `rag init` to create the config file interactively, or copy
`config.example.toml` to `~/.config/dropbox-rag/config.toml` and edit it.

**Qdrant connection refused**
Ensure the Qdrant Docker container is running:
```bash
docker ps | grep qdrant
# If not running:
docker compose up -d
```

**Model download hangs or fails**
Check your internet connection. If a partial download is corrupted, clear the
cache and retry:
```bash
rm -rf ~/.cache/dropbox-rag/models/
rag index  # re-downloads embedding model
make download-models  # re-exports reranker
```

**Reranker model not found**
The ONNX reranker model must be exported before first use. See Quick Start
step 5 or run:
```bash
make download-models
```

**Search returns no results**
Check that documents have been indexed:
```bash
rag status
```
If the document count is zero, run `rag index`. If indexing completed but search
still returns nothing, verify your query matches document content and check that
Qdrant is healthy with `rag doctor`.

**MCP server not responding in Claude Desktop**
Verify the MCP config is installed correctly:
```bash
rag mcp-config --print
```
Ensure the Python path in the config points to your virtualenv. Restart Claude
Desktop after config changes.


## License

See [LICENSE](LICENSE) for details.
