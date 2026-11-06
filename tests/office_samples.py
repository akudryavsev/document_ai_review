"""Сборка DOCX и PDF для тестов извлечения."""

from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_BREAK
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from models.documents import DocumentRef


class OfficeSamples:
    """Пишет небольшие офисные файлы с известным текстом."""

    def reference(self, path: Path, doc_id: str | None = None) -> DocumentRef:
        """Делает DocumentRef на уже записанный файл.

        Args:
            path: Путь к файлу.
            doc_id: Идентификатор; по умолчанию имя файла.

        Returns:
            DocumentRef.
        """
        return DocumentRef(
            doc_id=doc_id or path.name,
            path=str(path),
            suffix=path.suffix.lower(),
        )

    def docx_headings(self, path: Path) -> DocumentRef:
        """DOCX: Heading 1, абзац, Heading 2, абзац.

        Args:
            path: Куда сохранить.

        Returns:
            DocumentRef.
        """
        document = Document()
        document.add_heading("Регламент", level=1)
        document.add_paragraph("Первый абзац")
        document.add_heading("Сроки", level=2)
        document.add_paragraph("Второй абзац")
        document.save(path)
        return self.reference(path)

    def docx_russian_heading(self, path: Path) -> DocumentRef:
        """DOCX со стилем «Заголовок 1».

        Args:
            path: Куда сохранить.

        Returns:
            DocumentRef.
        """
        document = Document()
        style = document.styles.add_style("Заголовок 1", WD_STYLE_TYPE.PARAGRAPH)
        paragraph = document.add_paragraph("Введение")
        paragraph.style = style
        document.save(path)
        return self.reference(path)

    def docx_text_before_heading(self, path: Path) -> DocumentRef:
        """Абзац до первого заголовка.

        Args:
            path: Куда сохранить.

        Returns:
            DocumentRef.
        """
        document = Document()
        document.add_paragraph("Текст до структуры")
        document.add_heading("Глава", level=1)
        document.save(path)
        return self.reference(path)

    def docx_plain(self, path: Path, text: str = "Обычный абзац") -> DocumentRef:
        """Один абзац без разрыва страницы.

        Args:
            path: Куда сохранить.
            text: Текст абзаца.

        Returns:
            DocumentRef.
        """
        document = Document()
        document.add_paragraph(text)
        document.save(path)
        return self.reference(path)

    def docx_header(self, path: Path) -> DocumentRef:
        """Колонтитул с уникальной фразой и основной абзац.

        Args:
            path: Куда сохранить.

        Returns:
            DocumentRef.
        """
        document = Document()
        document.add_paragraph("Основной текст регламента")
        section = document.sections[0]
        section.header.paragraphs[0].text = "УНИКАЛЬНЫЙ КОЛОНТИТУЛ"
        section.footer.paragraphs[0].text = "УНИКАЛЬНЫЙ НИЖНИЙ"
        document.save(path)
        return self.reference(path)

    def docx_table(self, path: Path) -> DocumentRef:
        """Таблица 2×2.

        Args:
            path: Куда сохранить.

        Returns:
            DocumentRef.
        """
        document = Document()
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "a"
        table.cell(0, 1).text = "b"
        table.cell(1, 0).text = "c"
        table.cell(1, 1).text = "d"
        document.save(path)
        return self.reference(path)

    def docx_footnote(self, path: Path) -> DocumentRef:
        """Абзац со сноской «С момента увольнения».

        Args:
            path: Куда сохранить.

        Returns:
            DocumentRef.
        """
        document = Document()
        document.add_paragraph("PLACEHOLDER")
        document.save(path)
        self._inject_footnote(path)
        return self.reference(path)

    def docx_empty(self, path: Path) -> DocumentRef:
        """Пустой пакет DOCX.

        Args:
            path: Куда сохранить.

        Returns:
            DocumentRef.
        """
        document = Document()
        document.save(path)
        return self.reference(path)

    def docx_page_break(self, path: Path) -> DocumentRef:
        """Два абзаца, между ними явный разрыв страницы.

        Args:
            path: Куда сохранить.

        Returns:
            DocumentRef.
        """
        document = Document()
        first = document.add_paragraph("Первая")
        first.add_run().add_break(WD_BREAK.PAGE)
        document.add_paragraph("Вторая")
        document.save(path)
        return self.reference(path)

    def pdf_pages(self, path: Path, pages: list[list[str]]) -> DocumentRef:
        """PDF с заданными строками на каждой странице.

        Args:
            path: Куда сохранить.
            pages: Строки по страницам.

        Returns:
            DocumentRef.
        """
        writer = self._pdf_writer(pages)
        with path.open("wb") as handle:
            writer.write(handle)
        return self.reference(path)

    def pdf_image_only(self, path: Path) -> DocumentRef:
        """Страница с изображением и без текстовых операторов.

        Args:
            path: Куда сохранить.

        Returns:
            DocumentRef.
        """
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        page = writer.pages[0]
        image = DecodedStreamObject()
        image.set_data(b"\xff")
        image[NameObject("/Type")] = NameObject("/XObject")
        image[NameObject("/Subtype")] = NameObject("/Image")
        image[NameObject("/Width")] = NumberObject(1)
        image[NameObject("/Height")] = NumberObject(1)
        image[NameObject("/ColorSpace")] = NameObject("/DeviceGray")
        image[NameObject("/BitsPerComponent")] = NumberObject(8)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/XObject"): DictionaryObject({NameObject("/Im1"): image})}
        )
        content = DecodedStreamObject()
        content.set_data(b"q 100 0 0 100 0 0 cm /Im1 Do Q")
        page[NameObject("/Contents")] = content
        with path.open("wb") as handle:
            writer.write(handle)
        return self.reference(path)

    def corrupt(self, path: Path) -> DocumentRef:
        """Файл с расширением офисного документа, но без пакета.

        Args:
            path: Куда записать мусор.

        Returns:
            DocumentRef.
        """
        path.write_bytes(b"this is not a document")
        return self.reference(path)

    def _inject_footnote(self, path: Path) -> None:
        """Подменяет абзац-заглушку на абзац со сноской.

        Args:
            path: Уже сохранённый DOCX.
        """
        with ZipFile(path) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        document_xml = files["word/document.xml"].decode("utf-8")
        paragraph = (
            "<w:p>"
            '<w:r><w:t xml:space="preserve">Срок хранения</w:t></w:r>'
            '<w:r><w:footnoteReference w:id="1"/></w:r>'
            "<w:r><w:t>составляет 5 лет</w:t></w:r>"
            "</w:p>"
        )
        document_xml = re.sub(
            r"<w:p[\s>].*?</w:p>",
            paragraph,
            document_xml,
            count=1,
            flags=re.DOTALL,
        )
        files["word/document.xml"] = document_xml.encode("utf-8")
        files["word/footnotes.xml"] = self._footnotes_xml().encode("utf-8")
        rels = files["word/_rels/document.xml.rels"].decode("utf-8")
        relation = (
            '<Relationship Id="rIdFootnotes" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes" '
            'Target="footnotes.xml"/>'
        )
        rels = rels.replace("</Relationships>", relation + "</Relationships>")
        files["word/_rels/document.xml.rels"] = rels.encode("utf-8")
        content_types = files["[Content_Types].xml"].decode("utf-8")
        override = (
            '<Override PartName="/word/footnotes.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/>'
        )
        content_types = content_types.replace("</Types>", override + "</Types>")
        files["[Content_Types].xml"] = content_types.encode("utf-8")
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            for name, payload in files.items():
                archive.writestr(name, payload)

    def _footnotes_xml(self) -> str:
        """XML части footnotes.

        Returns:
            Документ footnotes.xml.
        """
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:t></w:t></w:r></w:p></w:footnote>
  <w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r><w:t></w:t></w:r></w:p></w:footnote>
  <w:footnote w:id="1"><w:p><w:r><w:t>С момента увольнения</w:t></w:r></w:p></w:footnote>
</w:footnotes>
"""

    def _pdf_writer(self, pages: list[list[str]]) -> PdfWriter:
        """Собирает PDF с ToUnicode, чтобы pypdf вернул исходные строки.

        Args:
            pages: Строки по страницам.

        Returns:
            PdfWriter.
        """
        seen: dict[str, int] = {}
        for page_lines in pages:
            for line in page_lines:
                for char in line:
                    if char not in seen:
                        seen[char] = len(seen) + 1
        bf_lines = [
            f"<{code:02X}> <{ord(char):04X}>" for char, code in seen.items()
        ]
        cmap_lines = [
            "/CIDInit /ProcSet findresource begin",
            "12 dict begin",
            "begincmap",
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
            "/CMapName /Custom def",
            "/CMapType 2 def",
            "1 begincodespacerange",
            "<00> <FF>",
            "endcodespacerange",
            f"{len(bf_lines)} beginbfchar",
            *bf_lines,
            "endbfchar",
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
        cmap_stream = DecodedStreamObject()
        cmap_stream.set_data("\n".join(cmap_lines).encode("ascii"))
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
                NameObject("/ToUnicode"): cmap_stream,
            }
        )
        writer = PdfWriter()
        for page_lines in pages:
            writer.add_blank_page(width=612, height=792)
            page = writer.pages[-1]
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            parts = ["BT /F1 12 Tf 72 720 Td"]
            first = True
            for line in page_lines:
                codes = "".join(f"{seen[char]:02X}" for char in line)
                if not first:
                    parts.append("0 -18 Td")
                first = False
                parts.append(f"<{codes}> Tj")
            parts.append("ET")
            stream = DecodedStreamObject()
            stream.set_data(" ".join(parts).encode("ascii"))
            page[NameObject("/Contents")] = stream
        return writer


class BlockFinder:
    """Ищет блоки по роли в результате extract."""

    def of_kind(self, blocks: list, kind: str) -> list:
        """Возвращает блоки указанной роли.

        Args:
            blocks: Список ExtractedBlock.
            kind: Роль.

        Returns:
            Подсписок.
        """
        return [block for block in blocks if block.kind == kind]
