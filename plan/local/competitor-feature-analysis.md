# Competitor Feature Analysis: What to Steal

**Date: March 15, 2026**

Deep dive into shinpr/mcp-local-rag, Khoj, AnythingLLM, and RAGFlow -- extracting features worth implementing in local-rag.

---

## Features Worth Implementing (Ranked by Impact)

### Tier 1: High Impact, Reasonable Effort

#### 1. Semantic Chunking (from shinpr/mcp-local-rag)
**What they do:** Max-Min algorithm adapted from Kiss et al. (2025). Instead of fixed 512-token chunks with overlap, they split by sentences, embed each sentence, and decide whether to start a new chunk based on a dynamic similarity threshold. A sigmoid function creates natural pressure to split as chunks grow larger. Code blocks are extracted before splitting and reinjected after, preventing mid-block splits.

**Why it matters:** Fixed-size chunking regularly fragments meaning at arbitrary boundaries. The author reports this was their single biggest quality improvement -- it eliminated the "compensatory tool calls" problem where LLMs would make extra retrieval requests because chunks were incoherent.

**What to implement:** Add semantic chunking as an alternative strategy alongside the existing 512-token chunker. Use the existing BGE-M3 embeddings to compute sentence similarities. Preserve the current chunker as a fallback for speed-sensitive scenarios. Key parameters: window_size=5, max_sentences=15, dynamic threshold with sigmoid growth.

**Effort:** Medium. Core algorithm is ~100 lines. Need sentence segmentation (use `Intl.Segmenter` equivalent -- Python's `pysbd` or spacy sentence splitting), per-sentence embedding (can batch), and the similarity threshold logic.

---

#### 2. Parent-Child Chunking (from RAGFlow)
**What they do:** Documents are first segmented into large "parent chunks" (complete semantic units like full sections). Each parent is subdivided into fine-grained "child chunks" used for retrieval. At query time, the system matches against child chunks but returns the parent chunk for complete context.

**Why it matters:** This is essentially what local-rag's citation expansion does for summary hits (expand to source chunks), but generalized to all chunks. It solves the core RAG problem: small chunks match precisely but lack context; large chunks have context but match poorly. Parent-child gives you both.

**What to implement:** Already partially built -- the section/document summary system with citation expansion is a form of this. Could extend by storing parent chunk IDs on child chunks and offering a "expand to parent" option in retrieval. The section structure from Docling parsing maps naturally to parent chunks.

**Effort:** Low-Medium. The data model mostly supports this already via section relationships.

---

#### 3. Auto-Metadata Generation During Indexing (from RAGFlow)
**What they do:** During file parsing, an LLM automatically generates metadata for each chunk: summaries, keywords, and potential questions the chunk could answer. This enriches chunks with semantic information beyond raw text.

**Why it matters:** local-rag already generates document and section summaries, but doesn't generate per-chunk metadata. Adding auto-generated questions ("what questions does this chunk answer?") would dramatically improve retrieval -- queries often don't use the same vocabulary as the source text, but generated questions bridge that gap.

**What to implement:** During the summarization step, also generate 2-3 questions per chunk via the LLM CLI. Store these as payload fields in Qdrant. Use them as additional search targets in the keyword index. This is a form of "query anticipation" that complements HyDE (which generates hypothetical answers).

**Effort:** Medium. Adds LLM calls proportional to chunk count (could batch). Storage is trivial -- just another Qdrant payload field.

---

#### 4. Agent Skills / Query Formulation Guidance (from shinpr/mcp-local-rag)
**What they do:** Pre-built prompt instructions ("skills") that teach AI assistants how to use the RAG tools effectively. Addresses the "dual invisibility problem" in MCP: both the query the LLM formulates and the results it receives are hidden from the user. Skills include query formulation patterns for different query types, score interpretation thresholds, and best practices.

**Why it matters:** The quality of RAG results depends heavily on how the calling LLM formulates its query. A generic "find information about X" query will get worse results than a well-structured one. Skills close this gap without requiring changes to the retrieval engine itself.

**What to implement:** Create a `.claude/skills/` directory with markdown instructions that can be installed into CLAUDE.md or .claude files. Include: query formulation templates (definition, how-to, comparison, troubleshooting), guidance on when to use `search_documents` vs `quick_search` vs `get_document_context`, score interpretation for the `detail` levels, and progressive disclosure patterns.

**Effort:** Low. Pure documentation/prompt engineering. No code changes needed -- just well-crafted markdown files and an install command.

---

#### 5. Relevance Gap Grouping (from shinpr/mcp-local-rag)
**What they do:** Instead of returning a fixed top-K, they detect statistical gaps in relevance scores. Using `threshold = mean + 1.5 * std`, they identify natural clusters of results. "similar" mode returns only the first (most relevant) group; "related" returns the top 2 groups. This means sometimes you get 3 results, sometimes 12 -- based on actual relevance, not an arbitrary cutoff.

**Why it matters:** Fixed top-K returns noise at the bottom of the list for specific queries and truncates good results for broad queries. Adaptive result count based on score distribution is more principled and reduces the noise-to-signal ratio for the calling LLM.

**What to implement:** After RRF fusion and reranking, apply a score gap detector before returning results. Offer this as a parameter on `search_documents` -- e.g., `adaptive_k: true` alongside the existing `top_k`. The calling LLM gets cleaner results without having to judge relevance itself.

**Effort:** Low. ~30 lines of statistics (mean, std, gap detection) applied to the final ranked list.

---

### Tier 2: Medium Impact, Worth Considering

#### 6. Cross-Language Search (from RAGFlow)
**What they do:** Supported in knowledge and chat modules for multilingual datasets. Their hybrid search (full-text + vector + tensor) handles cross-language queries natively.

**Why it matters:** BGE-M3 already supports 100+ languages and cross-language retrieval at the embedding level. But the keyword search lane won't match across languages. For developers with multilingual docs (e.g., Japanese API docs queried in English), this matters.

**What to implement:** BGE-M3 handles cross-language dense retrieval natively -- nothing to change there. For the keyword lane, consider language detection on ingestion and storing translated key_topics. Or simply document that cross-language works on the dense lanes but not keyword.

**Effort:** Low for documentation, Medium for keyword translation.

---

#### 7. Visual Chunking Inspection (from RAGFlow)
**What they do:** Web UI showing exactly how documents were parsed and chunked, with the ability to manually adjust before indexing. Users can see chunk boundaries, table extraction results, and heading hierarchy.

**Why it matters:** Debugging RAG quality issues almost always comes back to "what got chunked where?" A CLI-friendly version of this (even just `rag inspect <file>` showing chunk boundaries) would be invaluable for power users tuning their setup.

**What to implement:** A `rag inspect <file>` command that shows: parsed sections with headings, chunk boundaries with token counts, summary previews, and any OCR regions detected. Output as formatted text or JSON. Not a web UI -- keep it CLI-native.

**Effort:** Low-Medium. Data is already in SQLite and Qdrant -- just needs a presentation layer.

---

#### 8. HTML/Web Content Ingestion (from shinpr/mcp-local-rag, AnythingLLM)
**What they do:** shinpr uses Readability.js to strip nav/ads from HTML and convert to markdown before indexing. AnythingLLM uses Puppeteer for full page scraping. Both allow indexing web content alongside local files.

**Why it matters:** Developers often reference web documentation alongside local files. Being able to `rag add-url https://docs.example.com/api` and have it indexed alongside local docs would be useful.

**What to implement:** A `rag add-url <url>` command that fetches the page, extracts main content (using a Python equivalent of Readability -- `readability-lxml` or `trafilatura`), converts to markdown, and feeds into the existing pipeline. Store the URL as the source path. Could also support a `urls` section in config.toml for recurring syncs.

**Effort:** Medium. Needs a new ingestion path, but the downstream pipeline (chunk, embed, summarize, index) is already built.

---

#### 9. Incremental Embedding Updates (from Khoj)
**What they do:** Change detection via MD5 hashing. Only new/modified entries get re-embedded; unchanged entries keep existing embeddings. Batch processing in groups of 200.

**Why it matters:** local-rag already does content-hash-based dedup at the document level, but if a document changes, all its chunks get re-embedded. For large documents with small edits, this wastes time.

**What to implement:** Store per-chunk content hashes. On re-index, compute chunk hashes first, diff against stored hashes, and only re-embed changed/new chunks. Delete removed chunks. This would significantly speed up re-indexing of frequently-edited documents.

**Effort:** Medium. Requires chunk-level hash tracking in SQLite and diffing logic in the pipeline.

---

#### 10. Query Mode / Strict Grounding (from AnythingLLM)
**What they do:** "Query mode" restricts responses to embedded documents only. The LLM explicitly states when information is not found rather than hallucinating. This is distinct from "Chat mode" which allows the LLM to supplement with general knowledge.

**Why it matters:** For local-rag's MCP tools, the calling LLM might supplement retrieved context with its own training data. A `strict: true` parameter on search tools could include a system instruction telling the LLM to only use the returned evidence.

**What to implement:** Add a `grounded: true` parameter to `search_documents` that wraps results in explicit instructions: "Answer ONLY from the following evidence. If the answer is not found, say so." This is a prompt engineering feature, not a retrieval change.

**Effort:** Very low. Just a parameter that adds wrapper text to the MCP tool response.

---

### Tier 3: Interesting but Lower Priority

#### 11. ColBERT Late-Interaction Reranking (from RAGFlow)
**What they do:** Instead of cross-encoder reranking (which is what local-rag uses), RAGFlow supports ColBERT-style late-interaction scoring via their Infinity engine. MaxSim scoring computes cosine similarity between every query token embedding and all document token embeddings. Over 100x more efficient than cross-encoders while maintaining quality. Can scale to top 100-1,000 results without significant latency.

**Why it matters:** Cross-encoder reranking is local-rag's current bottleneck at ~200-350ms for 30 candidates. ColBERT could rerank 10x more candidates in similar time, improving recall without sacrificing latency. However, it requires pre-computed token-level embeddings stored in the vector DB.

**What to implement:** This is a significant architectural change. Would require storing per-token embeddings alongside chunk embeddings in Qdrant, and implementing MaxSim scoring. Worth investigating but not a quick win.

**Effort:** High. New embedding storage format, new scoring logic, potentially significant Qdrant storage increase.

---

#### 12. RAPTOR Hierarchical Summarization (from RAGFlow)
**What they do:** Chunks are clustered by semantic similarity, then LLM generates summaries of each cluster. These summaries are recursively clustered and summarized again, building a tree. Both original chunks and generated summaries are indexed.

**Why it matters:** local-rag already has a pyramid summarization system, which is arguably better-designed (geometric word counts at document and section levels). RAPTOR's clustering approach could complement it by finding cross-section themes, but the existing system already covers this use case well.

**What to implement:** Not recommended -- the existing pyramid summarization is more principled and deterministic. RAPTOR's clustering-based approach is less predictable and consumes significantly more LLM tokens.

**Effort:** High with marginal benefit over existing system.

---

#### 13. Code Execution Sandbox (from Khoj)
**What they do:** Terrarium (Pyodide in WASM) or E2B sandbox for running Python code. The LLM can write and execute code for quantitative analysis, chart generation, and data processing. Code is always shown to users for transparency.

**Why it matters:** This is outside local-rag's scope (retrieval, not execution), but interesting for the future. If local-rag ever evolves beyond pure retrieval, code execution could enable computed answers from structured data in documents (e.g., "what's the average price across all invoices?").

**What to implement:** Not recommended for now. local-rag's design philosophy is "RAG returns evidence, the calling LLM synthesizes." Code execution belongs in the calling LLM's tool set, not the retrieval layer.

**Effort:** N/A -- out of scope.

---

#### 14. Scheduled Automations (from Khoj)
**What they do:** Cron-based tasks that run queries on a schedule and deliver results via email. "Get me a summary of new documents every Monday at 9am."

**Why it matters:** Interesting for a "second brain" use case but outside local-rag's core scope. The MCP integration means the calling LLM can schedule its own tasks.

**What to implement:** Not recommended for core product. Could be a separate tool that calls local-rag's MCP interface on a schedule.

**Effort:** N/A -- out of scope.

---

#### 15. Deep Research / Iterative Retrieval (from Khoj)
**What they do:** Multi-step retrieval where the LLM performs multiple rounds of searching, refining queries based on what it finds. On Google's FRAMES benchmark, this boosted gemini-1.5-flash from 26.3% to 63.5% accuracy -- a 141.4% improvement.

**Why it matters:** This is powerful but belongs in the calling LLM, not the retrieval engine. Claude Code already does iterative tool calls when needed. However, local-rag could make this easier by returning "related queries" suggestions alongside results.

**What to implement:** Add a `related_queries` field to search results that suggests follow-up queries based on the retrieved content. This nudges the calling LLM toward iterative refinement without building a research loop into the retrieval engine.

**Effort:** Low-Medium. Use the existing query classification + retrieved content to generate 2-3 follow-up query suggestions.

---

## Features NOT Worth Implementing

| Feature | Source | Why Skip |
|---------|--------|----------|
| Web UI / dashboard | RAGFlow, AnythingLLM, Khoj | local-rag is CLI+MCP-first by design. A web UI adds maintenance burden without serving the core use case. `rag status` is sufficient. |
| Multi-user / RBAC | AnythingLLM | local-rag is a single-user, local tool. Multi-user adds complexity for zero benefit. |
| Image generation | Khoj | Completely out of scope for a document retrieval system. |
| Voice I/O | RAGFlow, Khoj | Out of scope. The calling LLM handles voice. |
| WhatsApp/Emacs/Obsidian plugins | Khoj | MCP is the universal integration layer. Building per-platform plugins is anti-MCP. |
| Embeddable chat widget | AnythingLLM | local-rag serves LLM tools, not end users. |
| 50+ file format support | AnythingLLM | Diminishing returns. PDF, DOCX, TXT, MD cover >95% of developer docs. Add formats only on user demand. |
| GraphRAG | RAGFlow, Khoj | High token cost, frequently unsatisfactory results (RAGFlow's own assessment). The pyramid summarization system handles document structure better for local-rag's use case. |
| LLM orchestration | AnythingLLM | local-rag retrieves evidence; it doesn't orchestrate LLM conversations. That's Claude's job. |
| Video parsing | RAGFlow | Out of scope for document retrieval. |

---

## Implementation Priority Roadmap

**Phase 1 (Quick Wins):**
1. Agent Skills / Query Formulation Guidance (#4) -- pure docs, huge impact
2. Relevance Gap Grouping (#5) -- ~30 lines, better result quality
3. Query Mode / Strict Grounding (#10) -- trivial, useful parameter
4. `rag inspect` command (#7) -- debugging aid for power users

**Phase 2 (Retrieval Quality):**
5. Semantic Chunking (#1) -- biggest quality improvement per competitor experience
6. Auto-Generated Questions per Chunk (#3) -- bridges vocabulary gap
7. Incremental Embedding Updates (#9) -- faster re-indexing

**Phase 3 (Ecosystem):**
8. HTML/Web Content Ingestion (#8) -- expands what can be indexed
9. Parent-Child Chunk Expansion (#2) -- better context in results
10. Related Query Suggestions (#15) -- enables iterative refinement

---

## Sources

- [shinpr/mcp-local-rag GitHub](https://github.com/shinpr/mcp-local-rag)
- [shinpr Dev.to Article: Semantic Chunking](https://dev.to/shinpr/building-a-local-rag-for-agentic-coding-from-fixed-chunks-to-semantic-search-with-keyword-boost-15m8)
- [Khoj AI GitHub](https://github.com/khoj-ai/khoj)
- [Khoj Documentation](https://docs.khoj.dev/)
- [Khoj Blog: Evaluating Quality](https://blog.khoj.dev/posts/evaluate-khoj-quality/)
- [AnythingLLM GitHub](https://github.com/Mintplex-Labs/anything-llm)
- [AnythingLLM Documentation](https://docs.anythingllm.com)
- [AnythingLLM v1.8.5 Changelog](https://docs.anythingllm.com/changelog/v1.8.5)
- [AnythingLLM DeepWiki Architecture](https://deepwiki.com/Mintplex-Labs/anything-llm)
- [RAGFlow GitHub](https://github.com/infiniflow/ragflow)
- [RAGFlow: From RAG to Context (2025 Review)](https://ragflow.io/blog/rag-review-2025-from-rag-to-context)
- [RAGFlow v0.23.0 Release](https://ragflow.io/blog/ragflow-0.23.0-advanding-memory-rag-and-agent-performance)
- [RAGFlow Parent-Child Chunking Docs](https://ragflow.io/docs/configure_child_chunking_strategy)
- [RAGFlow RAPTOR Implementation](https://ragflow.io/blog/long-context-rag-raptor)
- [Infinity Hybrid Search (RAGFlow)](https://infiniflow.org/blog/best-hybrid-search-solution)
