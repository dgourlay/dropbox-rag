# Batch Summarization Plan

## Problem
Each file currently requires N+1 LLM CLI calls (1 document + N sections).
With ~5-10s per call, a 5-section document takes 30-60s in summarization alone.
178 files = hours of indexing time.

## Solution: Combined Single-Call Summarization

### New Prompt Design
Create a single combined prompt that asks for both document-level AND section-level
summaries in one LLM call. The response is a single JSON object containing:

```json
{
  "summary_8w": "...",
  "summary_16w": "...",
  "summary_32w": "...",
  "summary_64w": "...",
  "summary_128w": "...",
  "key_topics": ["..."],
  "doc_type_guess": "...",
  "sections": [
    {
      "heading": "Section Title",
      "section_summary_8w": "...",
      "section_summary_32w": "...",
      "section_summary_128w": "..."
    }
  ]
}
```

### Adaptive Splitting for Large Documents
- Estimate prompt size: len(doc_excerpt) + sum(len(section_text) for all sections)
- If total < 80,000 chars (~20K tokens): **1 call** (combined doc + all sections)
- If total >= 80,000 chars: **2-3 calls**:
  - Call 1: Document summary only (existing DOCUMENT_PROMPT_TEMPLATE)
  - Call 2+: Batch sections into groups that fit under the threshold

### Implementation

#### File: `src/rag/pipeline/summarizer.py`
1. Add `COMBINED_PROMPT_TEMPLATE` — single prompt requesting doc + section summaries
2. Add `BATCH_SECTION_PROMPT_TEMPLATE` — batched sections prompt for the split case
3. Add `summarize_combined()` method to `CliSummarizer`:
   - Takes full text, title, file_type, list of (heading, section_text) pairs
   - Estimates prompt size
   - If under threshold: uses COMBINED_PROMPT_TEMPLATE, returns combined result
   - If over threshold: calls `summarize_document()` + `summarize_sections_batch()`
4. Add `summarize_sections_batch()` method:
   - Takes list of (heading, section_text) pairs + doc_context
   - Groups sections into batches that fit under char threshold
   - Returns list of SectionSummarySuccess results
5. Keep existing `summarize_document()` and `summarize_section()` as fallbacks

#### File: `src/rag/pipeline/runner.py`
1. Update `_summarize_document()` to call `summarize_combined()` instead of
   separate doc + section calls
2. Remove the ThreadPoolExecutor for parallel section summarization (no longer needed)
3. Parse the combined result into existing SummarySuccess + SectionSummarySuccess types

#### File: `src/rag/results.py`
1. Add `CombinedSummarySuccess` result type containing doc summary + list of section summaries
2. Add `CombinedSummaryResult` discriminated union

#### File: `src/rag/protocols.py`
1. Add `summarize_combined()` to the `Summarizer` protocol

### Progress Timestamps
Add elapsed time to progress display lines.

#### File: `src/rag/cli.py`
1. Track start time per file in `_ProgressDisplay`
2. Show elapsed time on each progress line: `[  1/178] file.pdf  indexed (4 chunks) [12.3s]`
3. Show elapsed on parsing/summarizing status lines too

#### File: `src/rag/pipeline/runner.py`
1. Pass timing info through callbacks — `_ParsedFileResult.start` already tracks this
2. Compute elapsed in `_report_progress` and pass to callback

### Char Threshold Constant
- `_COMBINED_PROMPT_CHAR_LIMIT = 80_000` in summarizer.py
- Section text still individually capped at `MAX_EXCERPT_CHARS = 5000`
- Document excerpt still capped at `MAX_EXCERPT_CHARS = 5000`

### Test Coverage Required
- Unit tests for `summarize_combined()` with mock CLI
- Unit tests for adaptive splitting (under/over threshold)
- Unit tests for batch section grouping
- Unit tests for combined JSON parsing (valid, malformed, partial)
- Unit tests for fallback when combined call fails
- Unit tests for progress timestamp formatting
- Integration test: full pipeline with combined summarization
