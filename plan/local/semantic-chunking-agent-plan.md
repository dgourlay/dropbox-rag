# Agent Team Plan: Semantic Chunking Implementation

## Overview

3-agent team implementing `plan/local/semantic-chunking-spec.md`. Single shared branch (no worktrees — agents share types.py, config.py, chunker.py). Team is not disbanded until `make lint` and `make test` pass clean.

## Agents

| Agent | Type | Spec Steps | Files Owned |
|---|---|---|---|
| S1: core-builder | backend-developer | 1, 2, 3, 4, 6 | types.py, config.py, chunker_semantic.py (new), chunker.py |
| S2: integrator | backend-developer | 5, 7, 9 | runner.py, cli.py, dashboard.py, mcp/tools.py |
| S3: test-writer | qa-expert | 8 + validation | test_chunker_semantic.py (new), test_config.py updates |

## Phasing

```
Phase 1 (parallel):
  S1 — types + config + chunker_semantic.py + chunker.py dispatch
  S3 — test scaffolding, mock Embedder, code block tests, dispatch test
       (uses spec function signatures before S1 finishes)

Phase 2 (parallel, after S1 done):
  S2 — runner.py call sites, CLI progress/status/doctor, MCP get_sync_status
  S3 — complete remaining tests (topic detection, guardrails, etc.)

Phase 3 (serial):
  S3 — run `make lint` + `make test`, report pass/fail
  Team disbanded only after Phase 3 passes
```

## Key Decisions

- **No worktrees**: Files overlap too much; merge conflicts would waste time
- **Backward-compatible dispatch**: `chunk_document()` gets optional params with defaults so existing callers work without changes until S2 wires them in
- **Lazy spaCy import**: Only loaded when semantic strategy is selected
- **Mock embedder in tests**: Deterministic vectors — similar sentences get similar vectors via small perturbations of a base vector
- **S3 is the quality gate**: Team is not disbanded until S3 confirms lint + tests pass
