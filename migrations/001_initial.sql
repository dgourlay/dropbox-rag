PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=30000;

CREATE TABLE IF NOT EXISTS sync_state (
    id                  TEXT PRIMARY KEY,
    file_path           TEXT UNIQUE NOT NULL,
    file_name           TEXT NOT NULL,
    folder_path         TEXT NOT NULL,
    folder_ancestors    TEXT NOT NULL,
    file_type           TEXT NOT NULL,
    size_bytes          INTEGER,
    modified_at         TEXT NOT NULL,
    content_hash        TEXT NOT NULL,
    synced_at           TEXT DEFAULT (datetime('now')),
    process_status      TEXT DEFAULT 'pending'
                        CHECK (process_status IN ('pending','processing','done','error','poison')),
    error_message       TEXT,
    retry_count         INTEGER DEFAULT 0,
    is_deleted          INTEGER DEFAULT 0,
    created_at          TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sync_status ON sync_state (process_status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_sync_path ON sync_state (file_path);

CREATE TABLE IF NOT EXISTS documents (
    doc_id                  TEXT PRIMARY KEY,
    file_path               TEXT UNIQUE NOT NULL,
    folder_path             TEXT NOT NULL,
    folder_ancestors        TEXT NOT NULL,
    title                   TEXT,
    file_type               TEXT NOT NULL,
    modified_at             TEXT NOT NULL,
    indexed_at              TEXT DEFAULT (datetime('now')),
    parser_version          TEXT,
    raw_content_hash        TEXT NOT NULL,
    normalized_content_hash TEXT,
    duplicate_of_doc_id     TEXT,
    ocr_required            INTEGER DEFAULT 0,
    ocr_confidence          REAL,
    doc_type_guess          TEXT,
    key_topics              TEXT,
    summary_l1              TEXT,
    summary_l2              TEXT,
    summary_l3              TEXT,
    summary_content_hash    TEXT,
    embedding_model_version TEXT,
    chunker_version         TEXT
);
CREATE INDEX IF NOT EXISTS idx_doc_folder ON documents (folder_path);
CREATE INDEX IF NOT EXISTS idx_doc_content_hash ON documents (normalized_content_hash);

CREATE TABLE IF NOT EXISTS sections (
    section_id          TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_heading     TEXT,
    section_order       INTEGER NOT NULL,
    page_start          INTEGER,
    page_end            INTEGER,
    section_summary     TEXT,
    section_summary_l2  TEXT,
    embedding_model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_section_doc ON sections (doc_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id                TEXT PRIMARY KEY,
    doc_id                  TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_id              TEXT,
    chunk_order             INTEGER NOT NULL,
    chunk_text              TEXT NOT NULL,
    chunk_text_normalized   TEXT NOT NULL,
    page_start              INTEGER,
    page_end                INTEGER,
    section_heading         TEXT,
    citation_label          TEXT,
    token_count             INTEGER,
    embedding_model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunk_section ON chunks (section_id);

CREATE TABLE IF NOT EXISTS document_hashes (
    file_path           TEXT PRIMARY KEY,
    raw_hash            TEXT NOT NULL,
    normalized_hash     TEXT,
    canonical_doc_id    TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS processing_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          TEXT,
    file_path       TEXT,
    stage           TEXT NOT NULL,
    status          TEXT NOT NULL,
    duration_ms     INTEGER,
    details         TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_proclog_doc ON processing_log (doc_id);
