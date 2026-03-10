from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from rag.pipeline.parser.text_parser import TextParser
from rag.results import ParseError, ParseSuccess
from rag.types import FileType


@pytest.fixture
def parser() -> TextParser:
    return TextParser()


class TestSupportedTypes:
    def test_supported_types(self, parser: TextParser) -> None:
        assert parser.supported_types == {FileType.TXT, FileType.MD}


class TestParsePlainText:
    def test_single_section_no_heading(self, parser: TextParser, tmp_path: Path) -> None:
        f = tmp_path / "notes.txt"
        f.write_text("Hello world.\nSecond line.", encoding="utf-8")

        result = parser.parse(str(f), ocr_enabled=False)
        assert isinstance(result, ParseSuccess)
        doc = result.document
        assert doc.file_type == FileType.TXT
        assert doc.title == "notes"
        assert len(doc.sections) == 1
        assert doc.sections[0].heading is None
        assert doc.sections[0].order == 0
        assert "Hello world." in doc.sections[0].text
        assert doc.raw_content_hash  # non-empty hash


class TestParseMarkdown:
    def test_multiple_headings(self, parser: TextParser, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text(
            "# Introduction\nSome intro text.\n\n## Details\nMore details here.\n",
            encoding="utf-8",
        )

        result = parser.parse(str(md), ocr_enabled=False)
        assert isinstance(result, ParseSuccess)
        doc = result.document
        assert doc.file_type == FileType.MD
        assert len(doc.sections) == 2
        assert doc.sections[0].heading == "Introduction"
        assert doc.sections[0].order == 0
        assert "Some intro text." in doc.sections[0].text
        assert doc.sections[1].heading == "Details"
        assert doc.sections[1].order == 1
        assert "More details here." in doc.sections[1].text

    def test_preamble_before_first_heading(self, parser: TextParser, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text(
            "This is preamble text.\n\n# First Heading\nBody text.\n",
            encoding="utf-8",
        )

        result = parser.parse(str(md), ocr_enabled=False)
        assert isinstance(result, ParseSuccess)
        doc = result.document
        assert len(doc.sections) == 2
        assert doc.sections[0].heading is None
        assert "preamble" in doc.sections[0].text
        assert doc.sections[1].heading == "First Heading"

    def test_no_headings_single_section(self, parser: TextParser, tmp_path: Path) -> None:
        md = tmp_path / "plain.md"
        md.write_text("Just some markdown without headings.", encoding="utf-8")

        result = parser.parse(str(md), ocr_enabled=False)
        assert isinstance(result, ParseSuccess)
        assert len(result.document.sections) == 1
        assert result.document.sections[0].heading is None


class TestEncodingFallback:
    def test_latin1_fallback(self, parser: TextParser, tmp_path: Path) -> None:
        f = tmp_path / "latin.txt"
        f.write_bytes("caf\xe9 na\xefve".encode("latin-1"))

        result = parser.parse(str(f), ocr_enabled=False)
        assert isinstance(result, ParseSuccess)
        assert "caf" in result.document.sections[0].text


class TestErrors:
    def test_nonexistent_file(self, parser: TextParser) -> None:
        result = parser.parse("/nonexistent/file.txt", ocr_enabled=False)
        assert isinstance(result, ParseError)
        assert "not found" in result.error.lower()

    def test_empty_file(self, parser: TextParser, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        result = parser.parse(str(f), ocr_enabled=False)
        assert isinstance(result, ParseError)
        assert "empty" in result.error.lower()

    def test_whitespace_only_file(self, parser: TextParser, tmp_path: Path) -> None:
        f = tmp_path / "blank.txt"
        f.write_text("   \n\n  \t  ", encoding="utf-8")

        result = parser.parse(str(f), ocr_enabled=False)
        assert isinstance(result, ParseError)
        assert "empty" in result.error.lower()
