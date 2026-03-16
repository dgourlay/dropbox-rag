# Auto-Generated Questions per Chunk — Build Specification

## 1. Document Purpose

This spec adds index-time question generation for chunks. At indexing time, an LLM generates 3-5 questions each chunk could answer. These questions are prepended to chunk text before embedding and stored as a keyword-indexed Qdrant payload field. This transforms chunk retrieval from question-to-passage matching into question-to-question matching, improving both dense and keyword search accuracy.

This is complementary to HyDE (which works at query time for unpredictable queries). Auto-generated questions work at index time and improve retrieval for the common case where users ask questions that align with content the chunks actually contain.

Academic lineage: doc2query (Nogueira 2019), docTTTTTquery (T5), Doc2Query-- (filtered), HyPE (2025), QuOTE (2025). HyPE reports up to 42 percentage point improvement in context precision. QuOTE reports 90.03% Top-1 Context Accuracy vs HyDE's 76.60%.

---

## 2. Design Decisions

| Decision | Resolution | Rationale |
|---|---|---|
| When to generate | **Index time**, after chunking, before embedding | Questions must exist before embedding so they are included in the vector. Zero query-time cost. |
| Questions per chunk | **3-5** (configurable, default 3) | Optimal for 512-token chunks per research. Diminishing returns beyond 5. |
| Storage strategy | **Prepend to chunk text before embedding** + **separate payload field** | Enriches the dense vector (QuOTE approach) AND adds query-like terms to keyword index. No index size multiplication. |
| LLM call granularity | **Batch multiple chunks per LLM call** | Same pattern as `BATCH_SECTION_PROMPT_TEMPLATE`. Group chunks under 80K char limit per call. |
| Pipeline position | **New step between chunking and embedding** | Must run before `embed_batch()` so augmented text is embedded. Separate from summarization (different granularity: chunks vs documents/sections). |
| Prompt design | **Separate prompt, not combined with summarization** | Summaries operate on document/section text. Questions operate on chunk text. Different inputs, different LLM calls. Combining would add complexity for no benefit. |
| Failure handling | **Graceful degradation** | If LLM fails for a batch, those chunks are indexed without questions. Same pattern as summarization. |
| Relationship to HyDE | **Complementary, independent** | HyDE generates hypothetical answers at query time. Questions are generated at index time. Both can be enabled simultaneously. |

---

## 3. Pipeline Integration

### Current Pipeline (steps from `runner.py`)

```
classify -> parse -> normalize -> dedup -> chunk -> save chunks -> embed -> index chunks -> summarize -> index summaries
```

### New Pipeline

```
classify -> parse -> normalize -> dedup -> chunk -> GENERATE QUESTIONS -> save chunks -> embed (augmented text) -> index chunks -> summarize -> index summaries
```

Question generation inserts between chunking and chunk saving. The generated questions are stored on the `Chunk` model, prepended to chunk text for embedding, and stored in the Qdrant payload for keyword search.

### Why Before Embedding (Not After Summarization)

Questions must be part of the embedded text. If questions were generated after embedding, the vectors would not reflect the question content, defeating the purpose. Summaries are generated after embedding because summary vectors are separate points in Qdrant. Chunk questions augment the chunk's own vector.

---

## 4. Batch Chunk Question Prompt

### `CHUNK_QUESTIONS_BATCH_PROMPT_TEMPLATE`

```python
CHUNK_QUESTIONS_BATCH_PROMPT_TEMPLATE = """\
Generate questions that each text chunk would be a good answer to. Return a JSON object with:
- "chunks": An array of question objects (one per chunk, in the same order)

Each chunk object must have:
- "chunk_order": The chunk number (as given below)
- "questions": A list of {questions_per_chunk} questions this chunk answers

Requirements for questions:
- Questions should resemble authentic user search queries
- Be explicit about entities and concepts — never use pronouns like "it", "this", "they"
- Vary question length from short (3-5 words) to longer (10-15 words)
- Cover different aspects of the chunk content
- For code content, include "how to" questions with actual function/class/API names

Return ONLY the JSON object, no other text.

Document title: {title}

Chunks:
{chunks_text}
"""
```

### Chunk Text Formatting

Reuse the same pattern as `_format_sections_text()`:

```python
def _format_chunks_text(chunks: list[tuple[int, str]]) -> str:
    """Format (chunk_order, text) pairs into numbered text for prompts."""
    parts: list[str] = []
    for chunk_order, text in chunks:
        excerpt = text[:MAX_EXCERPT_CHARS]
        parts.append(f"--- Chunk {chunk_order} ---\n{excerpt}")
    return "\n\n".join(parts)
```

### Example LLM Response

```json
{
  "chunks": [
    {
      "chunk_order": 0,
      "questions": [
        "How does BGE-M3 embedding model compare to bge-base?",
        "BGE-M3 dimensions and context window",
        "What embedding model should I use for multilingual RAG with long documents?"
      ]
    },
    {
      "chunk_order": 1,
      "questions": [
        "How to configure Qdrant collection for hybrid search?",
        "Qdrant text index setup",
        "What distance metric does Qdrant use for cosine similarity search?"
      ]
    }
  ]
}
```

---

## 5. Augmented Chunk Text

When questions are generated, the text sent to `embed_batch()` is augmented:

```python
def build_augmented_text(original_text: str, questions: list[str]) -> str:
    """Prepend generated questions to chunk text for embedding."""
    q_block = "\n".join(f"- {q}" for q in questions)
    return f"Questions this content answers:\n{q_block}\n\n{original_text}"
```

The augmented text is what gets embedded as the vector. The original `chunk.text` is preserved unchanged in the Qdrant `text` payload field (for display and keyword search on the original content). The questions are stored separately in a `generated_questions` payload field (for keyword search on question terms).

This means:
- **Dense vector** encodes both questions and content (enriched embedding)
- **Keyword index on `text`** matches original content terms
- **Keyword index on `generated_questions`** matches query-like terms (bridges vocabulary gap)
- **Display text** in MCP tool results shows original content, not augmented

---

## 6. Storage

### 6.1 Qdrant Payload

Add `generated_questions` field to `QdrantPayloadModel`:

```python
class QdrantPayloadModel(BaseModel):
    # ... existing fields ...
    generated_questions: list[str] | None = None
```

Configure Qdrant text index on `generated_questions` for keyword search:

```python
# In collection setup, alongside existing text index
client.create_payload_index(
    collection_name="documents",
    field_name="generated_questions",
    field_schema=models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        min_token_len=2,
        max_token_len=20,
        lowercase=True,
    ),
)
```

### 6.2 Chunk Model

Add `generated_questions` to the `Chunk` model:

```python
class Chunk(BaseModel):
    # ... existing fields ...
    generated_questions: list[str] | None = None
```

### 6.3 ChunkRow Model

Add `generated_questions` to `ChunkRow` for SQLite persistence:

```python
class ChunkRow(BaseModel):
    # ... existing fields ...
    generated_questions: str | None = None  # JSON-serialized list[str]
```

Stored as a JSON string in SQLite (same pattern as `key_topics` in the documents table).

### 6.4 QdrantPayloadReadBack

Add to the TypedDict:

```python
class QdrantPayloadReadBack(TypedDict):
    # ... existing fields ...
    generated_questions: list[str] | None
```

### 6.5 SQLite Migration

Add column to chunks table:

```sql
-- migrations/005_chunk_questions.sql
ALTER TABLE chunks ADD COLUMN generated_questions TEXT;
```

---

## 7. Type Changes Summary

| Type | Field | Type | Notes |
|---|---|---|---|
| `Chunk` | `generated_questions` | `list[str] \| None = None` | Set after question generation step |
| `ChunkRow` | `generated_questions` | `str \| None = None` | JSON-serialized for SQLite |
| `QdrantPayloadModel` | `generated_questions` | `list[str] \| None = None` | Keyword-indexed in Qdrant |
| `QdrantPayloadReadBack` | `generated_questions` | `list[str] \| None` | Read-back typing |

### Result Types

Add to `results.py`:

```python
class ChunkQuestionsSuccess(BaseModel):
    status: Literal["success"] = "success"
    chunks: list[ChunkQuestionEntry]


class ChunkQuestionEntry(BaseModel):
    chunk_order: int
    questions: list[str]


class ChunkQuestionsError(BaseModel):
    status: Literal["error"] = "error"
    error: str


ChunkQuestionsResult = Annotated[
    Annotated[ChunkQuestionsSuccess, Tag("success")]
    | Annotated[ChunkQuestionsError, Tag("error")],
    Discriminator("status"),
]
```

---

## 8. Batch Strategy

Reuse the 80K character limit pattern from `summarize_combined()`.

### Batching Logic

```python
def generate_chunk_questions_batch(
    self,
    chunks: list[Chunk],
    title: str | None,
    questions_per_chunk: int,
) -> list[Chunk]:
    """Generate questions for chunks in batched LLM calls.

    Groups chunks into batches that fit under _COMBINED_PROMPT_CHAR_LIMIT.
    Returns chunks with generated_questions populated. Chunks whose batch
    fails are returned with generated_questions=None (graceful degradation).
    """
```

### Batch Grouping

1. Walk the chunk list, accumulating chars (using `len(chunk.text[:MAX_EXCERPT_CHARS])`)
2. When accumulated chars exceed `_COMBINED_PROMPT_CHAR_LIMIT`, start a new batch
3. Each batch becomes one LLM CLI call
4. Parse JSON response, match by `chunk_order`, populate `chunk.generated_questions`

### Typical Batch Sizes

With 512-token chunks (~2000 chars each) and 80K char limit, each batch holds ~35-40 chunks. A 100-chunk document requires 2-3 LLM calls for questions.

---

## 9. Implementation in `summarizer.py`

Add a new method to `CliSummarizer` (not a new class). Question generation reuses the same CLI infrastructure: `_run_cli()`, JSON extraction, retry logic.

```python
class CliSummarizer:
    # ... existing methods ...

    def generate_chunk_questions(
        self,
        chunks: list[Chunk],
        title: str | None,
        questions_per_chunk: int = 3,
    ) -> list[Chunk]:
        """Generate questions for chunks via batched LLM calls."""
```

### Why Not a Separate Protocol

Question generation is tightly coupled to the summarizer infrastructure (same CLI tool, same JSON parsing, same retry logic). Adding it as a method on `CliSummarizer` is simpler than creating a new `QuestionGenerator` protocol. If a different backend is needed in the future, it can be extracted then.

However, add the method signature to the `Summarizer` protocol so mypy verifies it:

```python
class Summarizer(Protocol):
    # ... existing methods ...
    def generate_chunk_questions(
        self, chunks: list[Chunk], title: str | None, questions_per_chunk: int = 3,
    ) -> list[Chunk]: ...
```

---

## 10. Implementation in `runner.py`

### Single-File Path (`process_file`)

Insert question generation between chunking and chunk saving:

```python
# 8. Chunk
chunks = chunk_document(normalized)

# 8.5 Generate questions (if enabled)
if (self._summarizer and self._summarizer.available
        and self._config.chunk_questions.enabled):
    chunks = self._summarizer.generate_chunk_questions(
        chunks, title, self._config.chunk_questions.questions_per_chunk,
    )

# 9. Save chunks to DB
# ...

# 10. Embed (using augmented text)
texts = [build_augmented_text(c.text, c.generated_questions)
         if c.generated_questions else c.text
         for c in chunks]
vectors = self._embedder.embed_batch(texts) if texts else []
```

### Batch Path (`process_batch`)

In `_parser_thread()`, chunking happens in the parser thread. Question generation requires LLM calls (I/O bound, not CPU bound), so it runs in the consumer (main thread) alongside embedding:

```python
# In _index_parsed_file():
# Generate questions before embedding
if (self._summarizer and self._summarizer.available
        and self._config.chunk_questions.enabled):
    pr.chunks = self._summarizer.generate_chunk_questions(
        pr.chunks, pr.title, self._config.chunk_questions.questions_per_chunk,
    )
```

Questions must be generated before the cross-document `embed_batch()` call in `_flush_pending()`, which means they must be generated per-document in `_index_parsed_file()` before chunks are accumulated for batch embedding. This is the natural place: per-document question generation, then cross-document batch embedding.

---

## 11. Configuration

### TOML Config

```toml
[chunk_questions]
# Generate questions per chunk at index time (requires LLM CLI)
enabled = true
# Number of questions to generate per chunk
questions_per_chunk = 3
```

### Pydantic Config Model

```python
class ChunkQuestionsConfig(BaseModel):
    enabled: bool = True
    questions_per_chunk: int = Field(default=3, ge=1, le=10)
```

Add to `AppConfig`:

```python
class AppConfig(BaseModel):
    # ... existing fields ...
    chunk_questions: ChunkQuestionsConfig = ChunkQuestionsConfig()
```

### Dependency on Summarization

Question generation requires the same LLM CLI tool as summarization. If `summarization.enabled = false` and the CLI tool is not available, question generation is silently skipped (same graceful degradation pattern). The `chunk_questions.enabled` flag is independent of `summarization.enabled` -- you could disable summaries but still generate questions, or vice versa.

---

## 12. Impact on Retrieval

### Dense Lane (Chunk Vectors)

The chunk embedding now encodes both the content and the questions it answers. When a user query like "How do I configure Qdrant?" arrives, the query embedding is closer to a chunk whose vector includes "How to configure Qdrant collection for hybrid search?" than a chunk with only the raw configuration text. This is the core QuOTE/HyPE improvement.

No retrieval code changes needed. The vectors are simply better.

### Keyword Lane

The `generated_questions` payload field is keyword-indexed. This adds query-like vocabulary to the keyword search space. A user searching for "configure" will now match chunks that contain configuration instructions even if the original text uses "set up" or "establish" -- because the generated question used "configure." This bridges the vocabulary gap (the original doc2query insight).

Keyword search currently queries only the `text` field. To also search `generated_questions`, update the keyword query in the retrieval engine to search both fields:

```python
# In retrieval engine, keyword search construction:
# Add generated_questions as a second text match condition (OR)
models.FieldCondition(
    key="generated_questions",
    match=models.MatchText(text=query_text),
)
```

This is the only retrieval code change required.

### Summary Lanes

No change. Document and section summary vectors are separate points. They are not affected by chunk question generation.

### HyDE Interaction

HyDE and auto-generated questions are complementary. HyDE generates a hypothetical answer at query time (good for unpredictable queries). Questions are generated at index time (good for predictable queries that match what the content is about). Both can be enabled simultaneously. No interaction or conflict.

---

## 13. Graceful Degradation

Same pattern as summarization throughout:

1. **LLM CLI unavailable** -- `self._summarizer.available` returns `False`. Skip question generation entirely. Chunks indexed without questions.

2. **Batch LLM call fails** -- Log warning, return chunks in that batch with `generated_questions=None`. Other batches may succeed.

3. **JSON parse failure** -- Same JSON extraction pipeline as summarization (`json.loads` -> markdown fences -> outermost braces -> repair truncated JSON). If all extraction fails, treat as batch failure.

4. **Partial batch response** -- If the LLM returns questions for some chunks but not all (truncated JSON), assign questions to matched chunks, leave unmatched chunks with `generated_questions=None`.

5. **Feature disabled** -- `chunk_questions.enabled = false`. No LLM calls, no augmented text. Chunks embedded with original text only. Retrieval works identically to current behavior.

6. **Re-indexing existing documents** -- Documents indexed before this feature have no questions. They work fine. Running `rag index --reindex` regenerates everything including questions.

---

## 14. Testing

### Unit Tests

1. **`test_chunk_questions_prompt_formatting`** -- Verify `_format_chunks_text()` produces expected output for various chunk counts.

2. **`test_chunk_questions_json_parsing`** -- Feed mock LLM JSON responses through the parser. Verify questions are correctly assigned to chunks by `chunk_order`.

3. **`test_augmented_text_construction`** -- Verify `build_augmented_text()` prepends questions correctly. Verify original text is preserved when `generated_questions=None`.

4. **`test_batch_grouping`** -- Verify chunks are grouped into batches respecting the 80K char limit. Edge cases: single huge chunk, many tiny chunks, exactly at limit.

5. **`test_graceful_degradation`** -- Mock LLM failure. Verify chunks are returned with `generated_questions=None`. Verify no exception propagates.

6. **`test_partial_json_response`** -- Truncated JSON with some chunks' questions. Verify matched chunks get questions, unmatched get `None`.

7. **`test_config_disabled`** -- Verify no LLM calls when `chunk_questions.enabled = false`.

### E2E Tests

1. **`test_questions_in_qdrant`** -- Index a fixture document with questions enabled. Query Qdrant, verify `generated_questions` payload field is populated on chunk points.

2. **`test_question_keyword_search`** -- Index a fixture document. Search using a question phrase that appears in generated questions but not in original chunk text. Verify the chunk is returned via keyword lane.

3. **`test_questions_improve_dense_retrieval`** -- Index a fixture document. Compare search relevance with and without questions for a question-style query. (This is a qualitative test -- verify the target chunk ranks higher with questions enabled.)

---

## 15. Migration

Existing indexes work without changes. Documents indexed before this feature have `generated_questions=NULL` in SQLite and no `generated_questions` field in Qdrant payloads. Retrieval handles `None`/missing gracefully.

To add questions to existing documents, run `rag index --reindex`. This re-processes all documents through the full pipeline including question generation.

The SQLite migration (`ALTER TABLE chunks ADD COLUMN generated_questions TEXT`) is backward-compatible -- existing rows get `NULL`.

The Qdrant payload index on `generated_questions` is created on startup if it does not exist. Existing points without the field are simply not matched by keyword queries on that field.

---

## 16. Cost Analysis

### LLM Calls

For a corpus of 500 documents with ~10,000 chunks total:
- ~35-40 chunks per batch (80K char limit, ~2000 chars per chunk)
- ~250-285 LLM calls for question generation
- At ~5-10s per call: ~20-45 minutes additional indexing time
- Compare to summarization: ~500-1000 calls (1 combined call per doc + batch section calls for large docs)

Question generation roughly doubles the LLM call count during indexing. This is acceptable because it is a one-time cost (questions are cached with chunks) and the retrieval improvement is substantial.

### Embedding Cost

Each chunk's embedded text grows by ~45 words (3 questions x ~15 words). For a 512-token chunk (~384 words), this is a ~12% increase in text length. Embedding time increases proportionally -- negligible compared to LLM call time.

### Storage Cost

3 questions per chunk at ~15 words each = ~90 bytes per chunk. For 10,000 chunks = ~900KB. Negligible.

---

## 17. Future Considerations (Not in Scope)

- **Doc2Query-- filtering**: Compute similarity between generated questions and source chunk, discard low-similarity questions. 16% improvement over unfiltered. Could be added later without schema changes.
- **Question caching by content hash**: Skip question generation for unchanged chunks on re-index. Currently, `--reindex` regenerates everything. Content-hash-based caching could skip unchanged chunks.
- **Question quality scoring**: Track which generated questions actually match user queries (via search logs). Use to improve prompts over time.
