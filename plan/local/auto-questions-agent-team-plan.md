# Agent Team Plan: Auto-Generated Questions Feature

## Overview

This document defines the agent team strategy for implementing the auto-generated questions feature per `plan/local/auto-questions-spec.md`. This is a feature addition to an existing codebase, not a greenfield build. Agents are organized by dependency layers with maximum parallelism.

Each agent runs in an isolated worktree and produces a PR-ready branch. Agents that share no file dependencies can run in parallel within the same window.

---

## Dependency Graph

```
AQ1 (Types + Config)
 |
 ├──> AQ2 (Summarizer: question generation)
 |
 ├──> AQ3 (DB: migration + models + Qdrant index)
 |
 └──> AQ4 (Retrieval: keyword search on generated_questions)
       |
       (all of AQ2, AQ3, AQ4 complete)
       |
       v
     AQ5 (Runner: pipeline integration)
       |
       v
     AQ6 (MCP + CLI + Tests)
```

---

## Parallelism Windows

| Window | Agents | Description | Est. Time |
|--------|--------|-------------|-----------|
| **W1** | AQ1 | Types + Config (foundation — everything imports from these) | 10 min |
| **W2** | AQ2, AQ3, AQ4 | Summarizer, DB/Qdrant, Retrieval (parallel — no file overlaps) | 25 min |
| **W3** | AQ5 | Pipeline runner integration (depends on AQ1 types + AQ2 summarizer methods) | 20 min |
| **W4** | AQ6 | MCP tools, CLI displays, unit tests (depends on everything above) | 20 min |

**Total estimated wall-clock time:** ~75 minutes (vs ~95 minutes sequential).

---

## Agent Definitions

### AQ1: Types + Config (Foundation)

**Responsibility:** Add all new type fields and configuration model. This is the foundation layer — every other agent imports from `types.py` and `config.py`.

**Why combined:** These are small, tightly coupled changes (5 field additions across 4 models + 1 new config class). Splitting into two agents would create unnecessary coordination overhead for ~15 lines of code each.

**Files owned:**
- `src/rag/types.py` (modify)
- `src/rag/config.py` (modify)

**Changes:**

`src/rag/types.py`:
1. Add `generated_questions: list[str] | None = None` to `Chunk` model
2. Add `generated_questions: str | None = None` to `ChunkRow` model (JSON-serialized)
3. Add `generated_questions: list[str] | None = None` to `QdrantPayloadModel`
4. Add `generated_questions: list[str] | None` to `QdrantPayloadReadBack` TypedDict
5. Add `questions_enabled: bool = False` and `chunks_with_questions: int = 0` to `SyncStatusOutput`

`src/rag/config.py`:
1. Add `QuestionsConfig(BaseModel)` with `enabled: bool = True`
2. Add `questions: QuestionsConfig = QuestionsConfig()` to `AppConfig`

**Spec sections:** SS6.1-6.4, SS7, SS11

**Acceptance criteria:**
- `mypy --strict` passes on both files
- `ruff check` introduces no new errors
- All existing tests still pass (backward compatible — new fields have defaults)

---

### AQ2: Summarizer — Question Generation

**Responsibility:** Add the question generation prompt, batch logic, JSON parsing, augmented text builder, and formatting helpers to `CliSummarizer`. This is the core business logic of the feature.

**Why separate:** This is the largest single agent (~150 lines of new code). It has complex logic (batch grouping, JSON parsing, graceful degradation) that warrants focused attention and its own unit tests.

**Files owned:**
- `src/rag/pipeline/summarizer.py` (modify)

**Depends on:** AQ1 (imports `Chunk` model with `generated_questions` field)

**Changes:**

1. Add `CHUNK_QUESTIONS_BATCH_PROMPT_TEMPLATE` constant
2. Add `_format_chunks_text(chunks: list[Chunk]) -> str` helper
3. Add `build_augmented_text(original_text: str, questions: list[str]) -> str` module-level function
4. Add `generate_chunk_questions(self, chunks: list[Chunk], title: str | None) -> list[Chunk]` method to `CliSummarizer`
   - Batch grouping: walk chunks, accumulate chars, split at `_COMBINED_PROMPT_CHAR_LIMIT`
   - For each batch: format prompt, call `_run_cli()`, extract JSON, parse response
   - Match questions to chunks by `chunk_order`
   - Graceful degradation: batch failure -> chunks get `generated_questions=None`
   - Partial JSON: matched chunks get questions, unmatched get `None`

**Implementation notes:**
- Reuse `_run_cli()`, `_extract_json()`, `_repair_truncated_json()` — all existing infrastructure
- Follow the exact pattern of `summarize_combined()` for batch grouping logic
- Follow `_parse_combined_result()` pattern for JSON response parsing
- `build_augmented_text()` must be a module-level function (not a method) — `runner.py` imports it directly

**Spec sections:** SS4, SS5, SS8, SS9

**Acceptance criteria:**
- `build_augmented_text("original", ["Q1", "Q2"])` returns `"Questions this content answers:\n- Q1\n- Q2\n\noriginal"`
- `_format_chunks_text()` produces `--- Chunk N ---\n{text}` blocks
- Batch grouping respects 80K char limit
- LLM failure returns chunks with `generated_questions=None` (no exception)
- Partial JSON response assigns questions to matched chunks only

---

### AQ3: DB — Migration + Models + Qdrant Index

**Responsibility:** SQLite migration, update chunk save/load in models.py, add Qdrant payload index for `generated_questions`.

**Why combined:** These three changes are small (~10 lines each), touch the same conceptual layer (persistence), and have no internal dependencies that would benefit from splitting.

**Files owned:**
- `migrations/003_chunk_questions.sql` (create)
- `src/rag/db/models.py` (modify)
- `src/rag/db/qdrant.py` (modify)

**Depends on:** AQ1 (imports `ChunkRow` with `generated_questions` field)

**Changes:**

`migrations/003_chunk_questions.sql`:
```sql
ALTER TABLE chunks ADD COLUMN generated_questions TEXT;
```

`src/rag/db/models.py`:
1. Update `insert_chunks()` SQL to include `generated_questions` column
2. Update the tuple construction to include `c.generated_questions`
3. Update `get_chunks()` / `_row_to_chunk()` to read back `generated_questions`

`src/rag/db/qdrant.py`:
1. Add `"generated_questions"` to text index creation in both `QdrantVectorStore.ensure_collection()` and `AsyncQdrantVectorStore.ensure_collection()`:
   ```python
   client.create_payload_index(
       collection_name=self._collection,
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
   Note: `min_token_len=2` (not 3 like `text` field) per spec to catch short query terms.

**Spec sections:** SS6.1, SS6.3, SS6.5, SS17

**Acceptance criteria:**
- Migration SQL is valid
- `insert_chunks()` persists `generated_questions` as JSON string
- `get_chunks()` reads back `generated_questions` from stored JSON
- Qdrant text index created for `generated_questions` field
- Both sync and async Qdrant clients updated

---

### AQ4: Retrieval — Keyword Search on Questions

**Responsibility:** Update keyword search in the retrieval engine to also match against the `generated_questions` field.

**Why separate:** This is a surgical change to the retrieval query construction, but it touches sensitive search logic that benefits from isolated testing.

**Files owned:**
- `src/rag/db/qdrant.py` (modify — keyword query construction only, not index creation which is AQ3)

**Wait — file conflict with AQ3.** Both AQ3 and AQ4 modify `src/rag/db/qdrant.py`. This creates a merge conflict risk.

**Resolution:** Merge AQ3 and AQ4 into one agent. The combined scope is still small (~30 lines of changes across the file). See revised AQ3 below.

---

### AQ3 (Revised): DB + Qdrant + Retrieval

**Responsibility:** SQLite migration, update chunk save/load, add Qdrant payload index, and update keyword search to include `generated_questions`.

**Files owned:**
- `migrations/003_chunk_questions.sql` (create)
- `src/rag/db/models.py` (modify)
- `src/rag/db/qdrant.py` (modify)

**Depends on:** AQ1

**Additional changes beyond original AQ3:**

In both `QdrantVectorStore.query_keyword()` and `AsyncQdrantVectorStore.query_keyword()`:
- Change the text condition from a single `must` condition to an OR (`should`) over two fields:
  ```python
  text_condition = models.FieldCondition(
      key="text",
      match=models.MatchText(text=query),
  )
  questions_condition = models.FieldCondition(
      key="generated_questions",
      match=models.MatchText(text=query),
  )
  # Use should (OR) for text matching, must for other filters
  ```
- The existing filter conditions (folder_path, file_type, date, record_type) remain as `must` conditions
- The two text conditions become `should` conditions with `min_should` match of 1

**Spec sections:** SS6, SS12, SS17

---

### AQ5: Pipeline Runner Integration

**Responsibility:** Insert question generation step into both `process_file` (single path) and `process_batch` (batch path) in `runner.py`.

**Why separate:** Runner integration is the most coordination-sensitive change. It modifies the pipeline ordering and must correctly handle augmented text for embedding. Mistakes here break the entire indexing pipeline.

**Files owned:**
- `src/rag/pipeline/runner.py` (modify)

**Depends on:** AQ1 (types), AQ2 (summarizer methods + `build_augmented_text`)

**Changes:**

1. Add import: `from rag.pipeline.summarizer import build_augmented_text`
2. Add import: `from rag.config import QuestionsConfig` (TYPE_CHECKING)

3. **Single path (`process_file`):**
   - After chunking (step 8), before saving chunks (step 9):
     ```python
     if (self._summarizer and self._summarizer.available
             and self._config.questions.enabled):
         chunks = self._summarizer.generate_chunk_questions(chunks, title)
     ```
   - In the embedding step, use augmented text:
     ```python
     texts = [build_augmented_text(c.text, c.generated_questions)
              if c.generated_questions else c.text
              for c in chunks]
     ```

4. **Batch path (`process_batch`):**
   - In consumer loop, after receiving `_ParsedFileResult`, before appending to `pending`:
     ```python
     if (self._summarizer and self._summarizer.available
             and self._config.questions.enabled):
         pr.chunks = self._summarizer.generate_chunk_questions(pr.chunks, pr.title)
     ```
   - In `_flush_pending()`, use augmented text when collecting all_texts:
     ```python
     all_texts.extend(
         build_augmented_text(c.text, c.generated_questions)
         if c.generated_questions else c.text
         for c in pr.chunks
     )
     ```

**Critical constraint:** The `_ParsedFileResult` dataclass has a `chunks: list[Chunk]` field that must be mutable (it is — dataclass, not frozen). The runner reassigns `pr.chunks` after question generation.

**Spec sections:** SS3, SS10

**Acceptance criteria:**
- Question generation runs between chunking and embedding
- Augmented text is used for embedding, original text preserved in chunk
- Feature is skipped when `questions.enabled = false` or summarizer unavailable
- Both single and batch paths work correctly
- No regression in existing pipeline behavior

---

### AQ6: MCP + CLI + Unit Tests

**Responsibility:** Update MCP sync status tool, CLI status/doctor displays, and write all unit tests.

**Why combined:** The MCP and CLI changes are tiny (display-only). The unit tests cover all agents' work and are best written by an agent that can see the full picture. Combining avoids a 5th sequential window for tests.

**Files owned:**
- `src/rag/mcp/tools.py` (modify)
- `src/rag/cli.py` (modify)
- `tests/test_auto_questions.py` (create)

**Depends on:** AQ1-AQ5 (all prior agents)

**Changes:**

`src/rag/mcp/tools.py`:
1. In `_handle_sync_status()`: add `questions_enabled` and `chunks_with_questions` to output
2. In `_handle_search()` with `format=json`: include `generated_questions` in chunk result data

`src/rag/cli.py`:
1. In `status` command: display questions status line:
   ```
   Questions:    enabled (8,432 / 10,000 chunks have questions)
   ```
2. In `doctor` command: when `questions.enabled`, verify LLM CLI available:
   ```
   [checkmark] LLM CLI available for question generation
   ```

`tests/test_auto_questions.py` — Unit tests:
1. `test_format_chunks_text` — verify `_format_chunks_text()` output format
2. `test_build_augmented_text` — verify prepend format, verify passthrough when no questions
3. `test_chunk_questions_json_parsing` — mock `_run_cli()`, verify questions assigned to chunks
4. `test_batch_grouping` — many chunks, verify split at char limit
5. `test_graceful_degradation_llm_failure` — mock `_run_cli()` returning None, verify `generated_questions=None`
6. `test_partial_json_response` — truncated JSON, verify matched chunks get questions
7. `test_config_disabled` — `questions.enabled=False`, verify no LLM calls
8. `test_questions_config_defaults` — verify `QuestionsConfig` defaults
9. `test_chunk_model_with_questions` — verify `Chunk` model accepts questions field
10. `test_augmented_text_not_stored_as_chunk_text` — verify `chunk.text` unchanged after augmentation

**Spec sections:** SS14, SS15, SS16

**Acceptance criteria:**
- All new unit tests pass
- `make test` passes (no regression)
- `make lint` introduces no new errors (pre-existing errors are acceptable)
- MCP sync status includes question stats
- CLI status shows question count
- CLI doctor checks LLM availability for questions

---

## Revised Dependency Graph (Final)

```
AQ1 (Types + Config)
 |
 ├──> AQ2 (Summarizer)      ─┐
 |                             |
 └──> AQ3 (DB + Qdrant +     |
       Retrieval)             |
                               |
       (AQ2, AQ3 complete) ──┘
       |
       v
     AQ5 (Runner)
       |
       v
     AQ6 (MCP + CLI + Tests)
```

## Revised Parallelism Windows (Final)

| Window | Agents | Parallelism | Est. Time |
|--------|--------|-------------|-----------|
| **W1** | AQ1 | 1 agent | 10 min |
| **W2** | AQ2, AQ3 | 2 agents parallel | 25 min |
| **W3** | AQ5 | 1 agent | 20 min |
| **W4** | AQ6 | 1 agent | 20 min |

**Total: 4 agents, 4 windows, ~75 min wall-clock.**

---

## File Ownership Matrix

| File | AQ1 | AQ2 | AQ3 | AQ5 | AQ6 |
|------|-----|-----|-----|-----|-----|
| `src/rag/types.py` | **W** | R | R | R | R |
| `src/rag/config.py` | **W** | R | - | R | R |
| `src/rag/pipeline/summarizer.py` | - | **W** | - | R | R |
| `migrations/003_chunk_questions.sql` | - | - | **W** | - | - |
| `src/rag/db/models.py` | - | - | **W** | - | - |
| `src/rag/db/qdrant.py` | - | - | **W** | - | - |
| `src/rag/pipeline/runner.py` | - | - | - | **W** | R |
| `src/rag/mcp/tools.py` | - | - | - | - | **W** |
| `src/rag/cli.py` | - | - | - | - | **W** |
| `tests/test_auto_questions.py` | - | - | - | - | **W** |

**W** = writes (owns), **R** = reads (imports from), **-** = no interaction.

No file has two writers. This eliminates merge conflicts entirely.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Qdrant keyword OR query construction wrong | Medium | High (broken search) | AQ3 agent has focused scope; unit test covers OR logic |
| Runner batch path augmented text ordering | Medium | High (wrong vectors) | Spec SS10 is explicit; AQ5 acceptance criteria cover both paths |
| mypy strict violations from new fields | Low | Medium (lint failure) | All fields have defaults; `from __future__ import annotations` in every file |
| Pre-existing lint errors confuse agents | Medium | Low (wasted time) | Each agent instruction notes pre-existing errors to ignore |
| `_run_cli` reuse edge cases | Low | Medium | AQ2 follows exact pattern of existing `summarize_combined()` |

---

## Merge Order

1. Merge AQ1 first (foundation — all others depend on it)
2. Merge AQ2 and AQ3 (no conflicts — different files)
3. Merge AQ5 (depends on AQ2 methods being present)
4. Merge AQ6 (depends on everything)

If using a single branch with sequential commits, apply in order: AQ1 -> AQ2 -> AQ3 -> AQ5 -> AQ6.

---

## E2E Testing (Post-Merge, Manual)

After all agents merge, run `make test-e2e` to validate:
1. `rag index` with `questions.enabled=true` generates questions and stores in Qdrant
2. `rag search "question-style query"` returns results influenced by generated questions
3. `rag status` shows question count
4. `rag doctor` reports question generation health
5. MCP `get_sync_status` includes question stats

E2E tests require Docker Qdrant + real BGE-M3 model + real `claude` CLI. Not automatable in agent worktrees.
