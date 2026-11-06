"""T-EXT: извлечение DOCX и PDF."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from extract.debug_dump import BlockDocumentDebugDump
from extract.extractors import DocxExtractor, FormatAwareExtractor, PdfExtractor
from models.documents import DocumentRef, ExtractedDocument
from tests.office_samples import OfficeSamples


class TestExtractor:
    """Каталог T-EXT-01…24."""

    def setup_method(self) -> None:
        """Готовит сборщик файлов и извлекатели."""
        self.samples = OfficeSamples()
        self.docx = DocxExtractor()
        self.pdf = PdfExtractor()

    def test_t_ext_01_heading_paths_and_paragraph_indexes(self, tmp_path: Path) -> None:
        """Heading 1 / абзац / Heading 2 / абзац дают вложенные пути и номера 1 и 2."""
        ref = self.samples.docx_headings(tmp_path / "reglament.docx")
        extracted = self.docx.extract(ref)
        headings = [block for block in extracted.blocks if block.kind == "heading"]
        paragraphs = [block for block in extracted.blocks if block.kind == "paragraph"]
        assert [block.text for block in headings] == ["Регламент", "Сроки"]
        assert headings[0].heading_level == 1
        assert headings[0].section_path == ["Регламент"]
        assert headings[1].heading_level == 2
        assert headings[1].section_path == ["Регламент", "Сроки"]
        assert [block.paragraph_index for block in paragraphs] == [1, 2]
        assert paragraphs[0].section_path == ["Регламент"]
        assert paragraphs[1].section_path == ["Регламент", "Сроки"]

    def test_t_ext_02_russian_heading_style(self, tmp_path: Path) -> None:
        """Стиль «Заголовок 1» — заголовок уровня 1."""
        ref = self.samples.docx_russian_heading(tmp_path / "zagolovok.docx")
        extracted = self.docx.extract(ref)
        headings = [block for block in extracted.blocks if block.kind == "heading"]
        assert len(headings) == 1
        assert headings[0].heading_level == 1
        assert headings[0].text == "Введение"
        assert headings[0].section_path == ["Введение"]

    def test_t_ext_03_text_before_first_heading(self, tmp_path: Path) -> None:
        """Текст до первого заголовка не получает section_path."""
        ref = self.samples.docx_text_before_heading(tmp_path / "before.docx")
        extracted = self.docx.extract(ref)
        first = extracted.blocks[0]
        assert first.kind == "paragraph"
        assert first.section_path is None
        assert first.paragraph_index == 1

    def test_t_ext_04_no_page_breaks(self, tmp_path: Path) -> None:
        """Без явного разрыва страницы у абзаца page is None, номер абзаца есть."""
        ref = self.samples.docx_plain(tmp_path / "plain.docx")
        extracted = self.docx.extract(ref)
        paragraphs = [block for block in extracted.blocks if block.kind == "paragraph"]
        assert paragraphs
        assert all(block.page is None for block in paragraphs)
        assert all(block.paragraph_index is not None for block in paragraphs)

    def test_t_ext_05_header_not_in_main_text(self, tmp_path: Path) -> None:
        """Уникальная фраза колонтитула есть в header/footer и нет в основном тексте."""
        ref = self.samples.docx_header(tmp_path / "header.docx")
        extracted = self.docx.extract(ref)
        phrase = "УНИКАЛЬНЫЙ КОЛОНТИТУЛ"
        main = [
            block
            for block in extracted.blocks
            if block.kind in {"heading", "paragraph", "table"}
        ]
        sides = [
            block for block in extracted.blocks if block.kind in {"header", "footer"}
        ]
        assert sides
        assert any(phrase in block.text for block in sides)
        assert all(phrase not in block.text for block in main)
        kinds = [block.kind for block in extracted.blocks]
        seen_side = False
        for kind in kinds:
            if kind in {"header", "footer"}:
                seen_side = True
            elif seen_side:
                raise AssertionError("колонтитул стоит между блоками основного текста")

    def test_t_ext_06_table_cells(self, tmp_path: Path) -> None:
        """Таблица 2×2 — один блок table, четыре ячейки и свой paragraph_index."""
        ref = self.samples.docx_table(tmp_path / "table.docx")
        extracted = self.docx.extract(ref)
        tables = [block for block in extracted.blocks if block.kind == "table"]
        assert len(tables) == 1
        text = tables[0].text
        for cell in ("a", "b", "c", "d"):
            assert cell in text
        assert " | " in text
        assert "\n" in text
        assert tables[0].paragraph_index is not None
        assert tables[0].paragraph_index >= 1

    def test_t_ext_07_footnote_embedded(self, tmp_path: Path) -> None:
        """Текст сноски встроен в абзац, отдельного блока сноски нет."""
        ref = self.samples.docx_footnote(tmp_path / "note.docx")
        extracted = self.docx.extract(ref)
        paragraphs = [block for block in extracted.blocks if block.kind == "paragraph"]
        assert len(paragraphs) == 1
        assert (
            paragraphs[0].text
            == "Срок хранения [сноска: С момента увольнения] составляет 5 лет"
        )
        assert all("сноска" not in block.kind for block in extracted.blocks)

    def test_t_ext_08_pdf_heading_with_dot(self, tmp_path: Path) -> None:
        """Строка «3.2.1. Сроки» на странице 2 — заголовок уровня 3."""
        ref = self.samples.pdf_pages(
            tmp_path / "dotted.pdf",
            [["Введение документа"], ["3.2.1. Сроки"]],
        )
        extracted = self.pdf.extract(ref)
        heading = next(
            block for block in extracted.blocks if block.text == "3.2.1. Сроки"
        )
        assert heading.kind == "heading"
        assert heading.heading_level == 3
        assert heading.page == 2
        assert heading.section_path == ["3.2.1. Сроки"]

    def test_t_ext_09_pdf_heading_without_dot(self, tmp_path: Path) -> None:
        """Строка «3.2.1 Сроки» без точки после номера тоже заголовок уровня 3."""
        ref = self.samples.pdf_pages(
            tmp_path / "nodot.pdf",
            [["3.2.1 Сроки"]],
        )
        extracted = self.pdf.extract(ref)
        heading = extracted.blocks[0]
        assert heading.kind == "heading"
        assert heading.heading_level == 3
        assert heading.page == 1

    def test_t_ext_23_pdf_heading_wrap_keeps_full_paragraph(
        self, tmp_path: Path
    ) -> None:
        """Перенос нумерованной строки: заголовок — первая строка, абзац — вся фраза."""
        first = "1 ПОДГОТОВЛЕН Обществом «Институт развития информа-"
        second = "ционного общества» (ООО «ИРИО») на основе перевода."
        ref = self.samples.pdf_pages(tmp_path / "wrap.pdf", [[first, second]])
        extracted = self.pdf.extract(ref)
        headings = [block for block in extracted.blocks if block.kind == "heading"]
        paragraphs = [block for block in extracted.blocks if block.kind == "paragraph"]
        assert len(headings) == 1
        assert headings[0].text == first
        assert headings[0].heading_level == 1
        assert headings[0].section_path == [first]
        assert len(paragraphs) == 1
        expected = (
            "1 ПОДГОТОВЛЕН Обществом «Институт развития информационного общества» "
            "(ООО «ИРИО») на основе перевода."
        )
        assert paragraphs[0].text == expected
        assert paragraphs[0].section_path == [first]
        assert paragraphs[0].paragraph_index == 1
        assert paragraphs[0].page == 1

    def test_t_ext_24_pdf_heading_not_prefixed_to_next_sentence(
        self, tmp_path: Path
    ) -> None:
        """Предложение с заглавной после короткого заголовка остаётся отдельным абзацем."""
        heading_text = "3.2.1. Сроки"
        body = "Срок хранения составляет 5 лет."
        ref = self.samples.pdf_pages(
            tmp_path / "separate.pdf",
            [[heading_text, body]],
        )
        extracted = self.pdf.extract(ref)
        headings = [block for block in extracted.blocks if block.kind == "heading"]
        paragraphs = [block for block in extracted.blocks if block.kind == "paragraph"]
        assert len(headings) == 1
        assert headings[0].text == heading_text
        assert headings[0].heading_level == 3
        assert len(paragraphs) == 1
        assert paragraphs[0].text == body
        assert paragraphs[0].section_path == [heading_text]

    def test_t_ext_10_pdf_without_numbered_headings(self, tmp_path: Path) -> None:
        """Без нумерованных заголовков section_path пуст, абзацы пронумерованы."""
        ref = self.samples.pdf_pages(
            tmp_path / "plain.pdf",
            [["Просто абзац\n\nВторая мысль"]],
        )
        extracted = self.pdf.extract(ref)
        paragraphs = [block for block in extracted.blocks if block.kind == "paragraph"]
        assert len(paragraphs) >= 2
        assert all(block.section_path is None for block in paragraphs)
        assert [block.paragraph_index for block in paragraphs] == list(
            range(1, len(paragraphs) + 1)
        )

    def test_t_ext_11_pdf_images_only(self, tmp_path: Path) -> None:
        """PDF только с изображением даёт no_text_layer без исключения."""
        ref = self.samples.pdf_image_only(tmp_path / "scan.pdf")
        extracted = self.pdf.extract(ref)
        assert extracted.has_text_layer is False
        assert extracted.failure_reason == "no_text_layer"
        assert extracted.blocks == []
        assert extracted.joined_text() == ""

    def test_t_ext_12_empty_docx(self, tmp_path: Path) -> None:
        """Пустой DOCX — не no_text_layer и без блоков."""
        ref = self.samples.docx_empty(tmp_path / "empty.docx")
        extracted = self.docx.extract(ref)
        assert extracted.has_text_layer is True
        assert extracted.failure_reason is None
        assert extracted.blocks == []

    def test_t_ext_13_latin_preserved(self, tmp_path: Path) -> None:
        """Латиница и кириллица в одном абзаце сохраняются."""
        ref = self.samples.docx_plain(
            tmp_path / "latin.docx",
            text="Personal data и ПДн",
        )
        extracted = self.docx.extract(ref)
        paragraph = next(
            block for block in extracted.blocks if block.kind == "paragraph"
        )
        assert "Personal data" in paragraph.text
        assert "ПДн" in paragraph.text

    def test_t_ext_14_corrupt_file_raises(self, tmp_path: Path) -> None:
        """Повреждённый файл — исключение, а не no_text_layer."""
        docx_ref = self.samples.corrupt(tmp_path / "broken.docx")
        pdf_ref = self.samples.corrupt(tmp_path / "broken.pdf")
        with pytest.raises(Exception):
            self.docx.extract(docx_ref)
        with pytest.raises(Exception):
            self.pdf.extract(pdf_ref)

    def test_t_ext_15_format_aware_dispatches(self) -> None:
        """Диспетчер вызывает DOCX или PDF и не пересчитывает хеш сам."""

        class SpyDocx(DocxExtractor):
            """Запоминает вызов и возвращает готовый документ."""

            def __init__(self) -> None:
                """Готовит журнал."""
                super().__init__()
                self.calls: list[DocumentRef] = []

            def extract(self, document: DocumentRef) -> ExtractedDocument:
                """Фиксирует вызов.

                Args:
                    document: Ссылка.

                Returns:
                    Документ с заранее известным хешем.
                """
                self.calls.append(document)
                return ExtractedDocument(
                    doc_id=document.doc_id,
                    source_sha256="ab" * 32,
                    blocks=[],
                )

        class SpyPdf(PdfExtractor):
            """Запоминает вызов PDF."""

            def __init__(self) -> None:
                """Готовит журнал."""
                super().__init__()
                self.calls: list[DocumentRef] = []

            def extract(self, document: DocumentRef) -> ExtractedDocument:
                """Фиксирует вызов.

                Args:
                    document: Ссылка.

                Returns:
                    Документ с заранее известным хешем.
                """
                self.calls.append(document)
                return ExtractedDocument(
                    doc_id=document.doc_id,
                    source_sha256="cd" * 32,
                    blocks=[],
                )

        docx_spy = SpyDocx()
        pdf_spy = SpyPdf()
        dispatcher = FormatAwareExtractor(
            docx_extractor=docx_spy,
            pdf_extractor=pdf_spy,
        )
        docx_ref = DocumentRef(doc_id="a.docx", path="a.docx", suffix=".docx")
        pdf_ref = DocumentRef(doc_id="b.pdf", path="b.pdf", suffix=".pdf")
        docx_result = dispatcher.extract(docx_ref)
        pdf_result = dispatcher.extract(pdf_ref)
        assert docx_spy.calls == [docx_ref]
        assert pdf_spy.calls == [pdf_ref]
        assert docx_result.source_sha256 == "ab" * 32
        assert pdf_result.source_sha256 == "cd" * 32

    def test_t_ext_16_unknown_suffix(self) -> None:
        """Неизвестный суффикс — ValueError, файл не читается."""
        dispatcher = FormatAwareExtractor()
        ref = DocumentRef(doc_id="notes.txt", path="notes.txt", suffix=".txt")
        with pytest.raises(ValueError):
            dispatcher.extract(ref)

    def test_t_ext_17_same_file_same_hash(self, tmp_path: Path) -> None:
        """Два прогона одного файла дают один source_sha256 длиной 64 hex."""
        ref = self.samples.docx_plain(tmp_path / "same.docx", text="Один")
        first = self.docx.extract(ref)
        second = self.docx.extract(ref)
        assert first.source_sha256 == second.source_sha256
        assert re.fullmatch(r"[0-9a-f]{64}", first.source_sha256)

    def test_t_ext_18_same_bytes_different_name(self, tmp_path: Path) -> None:
        """Те же байты под другим именем дают тот же хеш и могут отличаться doc_id."""
        source = self.samples.docx_plain(tmp_path / "one.docx", text="Байты")
        payload = Path(source.path).read_bytes()
        other = tmp_path / "two.docx"
        other.write_bytes(payload)
        other_ref = self.samples.reference(other, doc_id="другое-имя.docx")
        first = self.docx.extract(source)
        second = self.docx.extract(other_ref)
        expected = hashlib.sha256(payload).hexdigest()
        assert first.source_sha256 == second.source_sha256 == expected
        assert first.doc_id != second.doc_id

    def test_t_ext_19_no_text_layer_keeps_hash(self, tmp_path: Path) -> None:
        """PDF без текста всё равно заполняет source_sha256."""
        ref = self.samples.pdf_image_only(tmp_path / "empty-layer.pdf")
        extracted = self.pdf.extract(ref)
        payload = Path(ref.path).read_bytes()
        assert extracted.failure_reason == "no_text_layer"
        assert extracted.source_sha256 == hashlib.sha256(payload).hexdigest()

    def test_t_ext_20_debug_dump_disabled(self, tmp_path: Path) -> None:
        """При debug_dump=false файл дампа не создаётся."""
        dump = BlockDocumentDebugDump(enabled=False, cache_dir=str(tmp_path))
        extractor = DocxExtractor(debug_dump=dump)
        ref = self.samples.docx_plain(tmp_path / "quiet.docx", text="Тишина")
        extractor.extract(ref)
        assert not (tmp_path / "debug" / "quiet.docx.extract.md").exists()

    def test_t_ext_21_debug_dump_two_blocks(self, tmp_path: Path) -> None:
        """Дамп содержит source_sha256, оба kind и оба текста."""
        dump = BlockDocumentDebugDump(enabled=True, cache_dir=str(tmp_path))
        extractor = DocxExtractor(debug_dump=dump)
        ref = self.samples.docx_headings(tmp_path / "reglament.docx")
        extracted = extractor.extract(ref)
        path = tmp_path / "debug" / "reglament.docx.extract.md"
        text = path.read_text(encoding="utf-8")
        assert extracted.source_sha256 in text
        assert "heading" in text
        assert "paragraph" in text
        assert "Регламент" in text
        assert "Первый абзац" in text

    def test_t_ext_22_debug_dump_no_text_layer(self, tmp_path: Path) -> None:
        """Дамп пустого слоя пишет failure_reason и не содержит тел блоков."""
        dump = BlockDocumentDebugDump(enabled=True, cache_dir=str(tmp_path))
        extractor = PdfExtractor(debug_dump=dump)
        ref = self.samples.pdf_image_only(tmp_path / "scan.pdf")
        extractor.extract(ref)
        text = (tmp_path / "debug" / "scan.pdf.extract.md").read_text(encoding="utf-8")
        assert "failure_reason: no_text_layer" in text
        assert "### [" not in text

    def test_partial_empty_pdf_pages_kept(self, tmp_path: Path) -> None:
        """Пустая страница в середине не валит документ и оставляет дыру в page_texts."""
        ref = self.samples.pdf_pages(
            tmp_path / "mixed.pdf",
            [["Первая страница"], [], ["Третья страница"]],
        )
        extracted = self.pdf.extract(ref)
        assert extracted.failure_reason is None
        assert extracted.has_text_layer is True
        pages = extracted.page_texts()
        assert pages[0] == "Первая страница"
        assert pages[1] == ""
        assert "Третья страница" in pages[2]

    def test_explicit_page_break_numbers_pages(self, tmp_path: Path) -> None:
        """Явный разрыв страницы даёт следующий номер со второй страницы."""
        ref = self.samples.docx_page_break(tmp_path / "pages.docx")
        extracted = self.docx.extract(ref)
        by_text = {block.text: block.page for block in extracted.blocks}
        assert by_text["Первая"] == 1
        assert by_text["Вторая"] == 2

    def test_flag_false_still_no_text_layer(self, tmp_path: Path) -> None:
        """Без OCR флаг pdf_text_layer_required=false не меняет no_text_layer."""
        ref = self.samples.pdf_image_only(tmp_path / "flag.pdf")
        extractor = PdfExtractor(pdf_text_layer_required=False)
        extracted = extractor.extract(ref)
        assert extracted.failure_reason == "no_text_layer"
