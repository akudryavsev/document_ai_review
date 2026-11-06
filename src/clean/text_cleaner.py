"""Очистка блоков перед чанкингом."""

from __future__ import annotations

import re

from extract.debug_dump import BlockDocumentDebugDump
from models.documents import CleanedDocument, ExtractedBlock, ExtractedDocument
from ports.pipeline import Cleaner as CleanerPort


class Cleaner(CleanerPort):
    """Убирает колонтитулы и нормализует пробелы и точки, не меняя адреса абзацев.

    Attributes:
        debug_dump: Необязательный дамп `{cache}/debug/{doc_id}.clean.md`.
    """

    # Любой Unicode-пробел (категория Zs и ASCII whitespace), не только NBSP.
    _WHITESPACE = re.compile(r"\s+")
    _DOT_LEADERS = re.compile(r"\.{4,}")
    _MAIN_KINDS = {"heading", "paragraph", "table"}

    def __init__(self, debug_dump: BlockDocumentDebugDump | None = None) -> None:
        """Сохраняет необязательный отладочный дамп.

        Args:
            debug_dump: Писатель Markdown или None.
        """
        self.debug_dump = debug_dump

    def clean(self, document: ExtractedDocument) -> CleanedDocument:
        """Очищает блоки успешного извлечения.

        Args:
            document: ExtractedDocument.

        Returns:
            CleanedDocument с тем же doc_id и source_sha256.
        """
        without_sides = [
            block
            for block in document.blocks
            if block.kind not in {"header", "footer"}
        ]
        repeated = self._repeated_edge_texts(without_sides)
        kept: list[ExtractedBlock] = []
        for block in without_sides:
            if self._is_bare_page_number(block):
                continue
            if block.kind == "paragraph" and block.text in repeated:
                continue
            kept.append(block)
        normalized: list[ExtractedBlock] = []
        for block in kept:
            text = self._normalize_block(block)
            if text == "":
                continue
            normalized.append(block.model_copy(update={"text": text}))
        cleaned = CleanedDocument(
            doc_id=document.doc_id,
            source_sha256=document.source_sha256,
            blocks=normalized,
        )
        if self.debug_dump is not None:
            self.debug_dump.write(cleaned, "clean")
        return cleaned

    def _repeated_edge_texts(self, blocks: list[ExtractedBlock]) -> set[str]:
        """Ищет короткие строки, которые открывают или закрывают много страниц.

        Args:
            blocks: Блоки уже без header/footer.

        Returns:
            Тексты, которые нужно снять со всех paragraph.
        """
        by_page: dict[int, list[ExtractedBlock]] = {}
        for block in blocks:
            if block.kind not in self._MAIN_KINDS or block.page is None:
                continue
            by_page.setdefault(block.page, []).append(block)
        pages_by_text: dict[str, set[int]] = {}
        for page_number, page_blocks in by_page.items():
            edges = [page_blocks[0], page_blocks[-1]]
            seen_on_page: set[str] = set()
            for edge in edges:
                if edge.text in seen_on_page:
                    continue
                seen_on_page.add(edge.text)
                if len(edge.text) > 80:
                    continue
                pages_by_text.setdefault(edge.text, set()).add(page_number)
        repeated: set[str] = set()
        for text, pages in pages_by_text.items():
            if len(pages) >= 3:
                repeated.add(text)
        return repeated

    def _is_bare_page_number(self, block: ExtractedBlock) -> bool:
        """Проверяет, что абзац целиком равен номеру своей страницы.

        Args:
            block: Блок.

        Returns:
            True, если paragraph после обрезки краёв равен str(page).
        """
        if block.kind != "paragraph" or block.page is None:
            return False
        return block.text.strip() == str(block.page)

    def _normalize_block(self, block: ExtractedBlock) -> str:
        """Нормализует текст одного блока.

        Args:
            block: Блок.

        Returns:
            Новый текст. Пустая строка значит блок надо выбросить.
        """
        if block.kind == "table":
            return self._normalize_table(block.text)
        return self._normalize_plain(block.text)

    def _normalize_plain(self, text: str) -> str:
        """Схлопывает пробелы и точки, заменяет кавычки и тире.

        Args:
            text: Исходный текст блока.

        Returns:
            Нормализованный текст.
        """
        without_soft = text.replace("\u00ad", "")
        replaced = self._replace_marks(without_soft)
        collapsed = self._collapse_whitespace(replaced)
        return self._collapse_dot_leaders(collapsed)

    def _normalize_table(self, text: str) -> str:
        """Нормализует ячейки, сохраняя разделители строк и ` | `.

        Args:
            text: Markdown-подобная таблица извлекателя.

        Returns:
            Текст таблицы.
        """
        without_soft = text.replace("\u00ad", "")
        replaced = self._replace_marks(without_soft)
        trimmed = replaced.strip()
        if trimmed == "":
            return ""
        rows = trimmed.split("\n")
        normalized_rows: list[str] = []
        for row in rows:
            cells = row.split(" | ")
            normalized_cells = [
                self._collapse_dot_leaders(self._collapse_cell(cell))
                for cell in cells
            ]
            normalized_rows.append(" | ".join(normalized_cells))
        return "\n".join(normalized_rows)

    def _collapse_whitespace(self, text: str) -> str:
        """Схлопывает Unicode-пробелы в один обычный и обрезает края.

        Args:
            text: Текст после замены знаков.

        Returns:
            Текст без краевых и повторных пробелов.
        """
        collapsed = self._WHITESPACE.sub(" ", text)
        return collapsed.strip()

    def _collapse_cell(self, text: str) -> str:
        """Схлопывает пробелы внутри одной ячейки.

        Args:
            text: Текст ячейки.

        Returns:
            Ячейка без краевых пробелов.
        """
        return self._collapse_whitespace(text)

    def _collapse_dot_leaders(self, text: str) -> str:
        """Схлопывает четыре и более точек подряд до трёх.

        Args:
            text: Текст после схлопывания пробелов.

        Returns:
            Текст без «дорожек» точек оглавления.
        """
        return self._DOT_LEADERS.sub("...", text)

    def _replace_marks(self, text: str) -> str:
        """Заменяет кавычки и тире по таблице спецификации.

        Args:
            text: Текст без мягких переносов.

        Returns:
            Текст с обычными кавычками и дефисом.
        """
        result = text
        for char in "«»“”„":
            result = result.replace(char, '"')
        for char in "‘’":
            result = result.replace(char, "'")
        for char in "\u2010\u2011\u2013\u2014":
            result = result.replace(char, "-")
        return result
