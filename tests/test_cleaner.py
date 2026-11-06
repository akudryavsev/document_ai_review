"""T-CLN: очистка блоков."""

from __future__ import annotations

from pathlib import Path

from clean.text_cleaner import Cleaner
from extract.debug_dump import BlockDocumentDebugDump
from models.documents import CleanedDocument, ExtractedBlock, ExtractedDocument


class TestCleaner:
    """Каталог T-CLN-01…17."""

    def setup_method(self) -> None:
        """Готовит очиститель без дампа."""
        self.cleaner = Cleaner()

    def test_t_cln_01_headers_removed(self) -> None:
        """header и footer исчезают, основной текст остаётся."""
        document = self._document(
            [
                ExtractedBlock(kind="header", text="Колонтитул"),
                ExtractedBlock(
                    kind="paragraph",
                    text="Основной текст",
                    paragraph_index=1,
                ),
                ExtractedBlock(kind="footer", text="Низ"),
            ]
        )
        cleaned = self.cleaner.clean(document)
        assert all(block.kind not in {"header", "footer"} for block in cleaned.blocks)
        assert cleaned.blocks[0].text == "Основной текст"

    def test_t_cln_02_whitespace_and_case(self) -> None:
        """Лишние пробелы схлопываются, регистр сохраняется."""
        document = self._document(
            [ExtractedBlock(kind="paragraph", text="  Personal   Data \n")]
        )
        cleaned = self.cleaner.clean(document)
        assert cleaned.blocks[0].text == "Personal Data"

    def test_t_cln_03_latin_and_quotes(self) -> None:
        """Латиница остаётся, ёлочки становятся прямыми кавычками."""
        document = self._document(
            [ExtractedBlock(kind="paragraph", text="Personal «текст»")]
        )
        cleaned = self.cleaner.clean(document)
        assert cleaned.blocks[0].text == 'Personal "текст"'

    def test_t_cln_04_em_dash(self) -> None:
        """Тире U+2014 заменяется на дефис."""
        document = self._document(
            [ExtractedBlock(kind="paragraph", text="слово\u2014слово")]
        )
        cleaned = self.cleaner.clean(document)
        assert cleaned.blocks[0].text == "слово-слово"

    def test_t_cln_05_soft_hyphen(self) -> None:
        """Мягкий перенос удаляется, части слова сходятся."""
        document = self._document(
            [ExtractedBlock(kind="paragraph", text="пере\u00adнос")]
        )
        cleaned = self.cleaner.clean(document)
        assert cleaned.blocks[0].text == "перенос"

    def test_t_cln_06_empty_document(self) -> None:
        """Пустой список блоков даёт пустой CleanedDocument без исключения."""
        document = self._document([])
        cleaned = self.cleaner.clean(document)
        assert isinstance(cleaned, CleanedDocument)
        assert cleaned.blocks == []
        assert cleaned.doc_id == document.doc_id
        assert cleaned.source_sha256 == document.source_sha256

    def test_t_cln_07_already_clean_block_unchanged(self) -> None:
        """Чистый абзац сохраняет текст, путь, страницу, роль и номер."""
        block = ExtractedBlock(
            kind="paragraph",
            text="Срок хранения",
            page=4,
            section_path=["Сроки"],
            paragraph_index=3,
        )
        document = self._document([block])
        cleaned = self.cleaner.clean(document)
        assert len(cleaned.blocks) == 1
        result = cleaned.blocks[0]
        assert result.text == block.text
        assert result.section_path == block.section_path
        assert result.page == block.page
        assert result.kind == block.kind
        assert result.paragraph_index == block.paragraph_index
        assert cleaned.source_sha256 == document.source_sha256

    def test_t_cln_08_short_heading_kept(self) -> None:
        """Короткий заголовок не удаляется эвристикой колонтитула."""
        document = self._document(
            [
                ExtractedBlock(
                    kind="heading",
                    text="3.2",
                    heading_level=2,
                    section_path=["3.2"],
                )
            ]
        )
        cleaned = self.cleaner.clean(document)
        assert len(cleaned.blocks) == 1
        assert cleaned.blocks[0].text == "3.2"

    def test_t_cln_09_page_number_removed(self) -> None:
        """Строка «12» на странице 12 снимается."""
        document = self._document(
            [
                ExtractedBlock(kind="paragraph", text="12", page=12, paragraph_index=1),
                ExtractedBlock(
                    kind="paragraph",
                    text="Срок хранения",
                    page=12,
                    paragraph_index=2,
                ),
            ]
        )
        cleaned = self.cleaner.clean(document)
        assert [block.text for block in cleaned.blocks] == ["Срок хранения"]

    def test_t_cln_10_repeated_edge_removed(self) -> None:
        """Одна и та же короткая первая строка на трёх страницах снимается целиком."""
        blocks = []
        for page in (1, 2, 3):
            blocks.append(
                ExtractedBlock(
                    kind="paragraph",
                    text="Регламент 2024",
                    page=page,
                    paragraph_index=page,
                )
            )
            blocks.append(
                ExtractedBlock(
                    kind="paragraph",
                    text=f"Текст страницы {page}",
                    page=page,
                    paragraph_index=page + 10,
                )
            )
        cleaned = self.cleaner.clean(self._document(blocks))
        texts = [block.text for block in cleaned.blocks]
        assert "Регламент 2024" not in texts
        assert texts == ["Текст страницы 1", "Текст страницы 2", "Текст страницы 3"]

    def test_t_cln_11_paragraph_index_stable(self) -> None:
        """После снятия колонтитула paragraph_index=20 не перенумеровывается."""
        document = self._document(
            [
                ExtractedBlock(kind="header", text="Шапка"),
                ExtractedBlock(
                    kind="paragraph",
                    text="Абзац двадцать",
                    paragraph_index=20,
                ),
            ]
        )
        cleaned = self.cleaner.clean(document)
        assert cleaned.blocks[0].paragraph_index == 20

    def test_t_cln_12_footnote_marker_kept(self) -> None:
        """Маркер сноски остаётся, пробелы и кавычки внутри нормализуются."""
        document = self._document(
            [
                ExtractedBlock(
                    kind="paragraph",
                    text="Текст  [сноска:  «цитата»  ]   конец",
                )
            ]
        )
        cleaned = self.cleaner.clean(document)
        # Последовательности пробелов схлопываются в один, маркер не вырезается.
        assert cleaned.blocks[0].text == 'Текст [сноска: "цитата" ] конец'
        assert "[сноска:" in cleaned.blocks[0].text

    def test_t_cln_13_debug_dump_disabled(self, tmp_path: Path) -> None:
        """Выключенный дамп не создаёт clean.md."""
        dump = BlockDocumentDebugDump(enabled=False, cache_dir=str(tmp_path))
        cleaner = Cleaner(debug_dump=dump)
        cleaner.clean(self._document([ExtractedBlock(kind="paragraph", text="Текст")]))
        assert not (tmp_path / "debug" / "doc.docx.clean.md").exists()

    def test_t_cln_14_debug_dump_without_header(self, tmp_path: Path) -> None:
        """В clean.md нет колонтитула, текст тела и source_sha256 на месте."""
        dump = BlockDocumentDebugDump(enabled=True, cache_dir=str(tmp_path))
        cleaner = Cleaner(debug_dump=dump)
        document = self._document(
            [
                ExtractedBlock(kind="header", text="СЕКРЕТНЫЙ КОЛОНТИТУЛ"),
                ExtractedBlock(kind="paragraph", text="Тело документа"),
            ]
        )
        cleaner.clean(document)
        text = (tmp_path / "debug" / "doc.docx.clean.md").read_text(encoding="utf-8")
        assert text.startswith("# clean: doc.docx")
        assert document.source_sha256 in text
        assert "Тело документа" in text
        assert "СЕКРЕТНЫЙ КОЛОНТИТУЛ" not in text

    def test_t_cln_15_unicode_en_space(self) -> None:
        """En space схлопывается в обычный пробел наравне с ASCII-пробелами."""
        document = self._document(
            [ExtractedBlock(kind="paragraph", text="слово -\u2002 слово")]
        )
        cleaned = self.cleaner.clean(document)
        assert cleaned.blocks[0].text == "слово - слово"

    def test_t_cln_16_dot_leaders_collapsed(self) -> None:
        """Четыре и более точек подряд становятся многоточием из трёх."""
        document = self._document(
            [ExtractedBlock(kind="paragraph", text="Содержание..........5")]
        )
        cleaned = self.cleaner.clean(document)
        assert cleaned.blocks[0].text == "Содержание...5"

    def test_t_cln_17_three_dots_unchanged(self) -> None:
        """Ровно три точки не трогаются."""
        document = self._document(
            [ExtractedBlock(kind="paragraph", text="готово...")]
        )
        cleaned = self.cleaner.clean(document)
        assert cleaned.blocks[0].text == "готово..."

    def _document(self, blocks: list[ExtractedBlock]) -> ExtractedDocument:
        """Собирает вход очистки.

        Args:
            blocks: Блоки.

        Returns:
            ExtractedDocument.
        """
        return ExtractedDocument(
            doc_id="doc.docx",
            source_sha256="ab" * 32,
            has_text_layer=True,
            blocks=blocks,
        )
