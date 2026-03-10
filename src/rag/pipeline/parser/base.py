from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.protocols import Parser
    from rag.types import FileType


def get_parser(file_type: FileType, parsers: list[Parser]) -> Parser | None:
    """Return the first parser that supports the given file type, or None."""
    for p in parsers:
        if file_type in p.supported_types:
            return p
    return None
