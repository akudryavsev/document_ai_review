"""Извлечение текста из DOCX и PDF."""

from extract.debug_dump import BlockDocumentDebugDump
from extract.extractors import (
    DocxExtractor,
    Extractor,
    FormatAwareExtractor,
    PdfExtractor,
)

__all__ = [
    "BlockDocumentDebugDump",
    "DocxExtractor",
    "Extractor",
    "FormatAwareExtractor",
    "PdfExtractor",
]
