from __future__ import annotations

import hashlib
import re

from rag.types import NormalizedDocument, ParsedDocument, ParsedSection


def normalize(doc: ParsedDocument) -> NormalizedDocument:
    """Normalize a parsed document: clean whitespace, preserve headings, compute hash."""
    normalized_sections: list[ParsedSection] = []

    for section in doc.sections:
        normalized_text = _normalize_text(section.text)
        if not normalized_text:
            continue
        normalized_sections.append(
            ParsedSection(
                heading=section.heading.strip() if section.heading else None,
                order=section.order,
                text=normalized_text,
                page_start=section.page_start,
                page_end=section.page_end,
            )
        )

    # Compute hash on normalized full text
    full_text = "\n\n".join(s.text for s in normalized_sections)
    normalized_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

    return NormalizedDocument(
        doc_id=doc.doc_id,
        title=doc.title,
        file_type=doc.file_type,
        sections=normalized_sections,
        normalized_content_hash=normalized_hash,
        raw_content_hash=doc.raw_content_hash,
    )


def _normalize_text(text: str) -> str:
    """Clean whitespace while preserving structure."""
    # Strip null bytes (corrupt documents can contain them)
    text = text.replace("\x00", "")
    # Collapse multiple blank lines to max 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces to single space (but not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Strip leading/trailing whitespace
    text = text.strip()
    return text
