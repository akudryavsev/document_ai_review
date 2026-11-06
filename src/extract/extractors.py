"""Извлечение блоков из DOCX и PDF."""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from extract.debug_dump import BlockDocumentDebugDump
from models.documents import DocumentRef, ExtractedBlock, ExtractedDocument
from ports.pipeline import Extractor as ExtractorPort


class Extractor(ExtractorPort):
    """Общая сборка блоков: путь заголовков, номера абзацев, отпечаток файла.

    Attributes:
        debug_dump: Необязательный Markdown-дамп после extract.
    """

    def __init__(self, debug_dump: BlockDocumentDebugDump | None = None) -> None:
        """Сохраняет необязательный дамп.

        Args:
            debug_dump: Писатель Markdown или None.
        """
        self.debug_dump = debug_dump

    def extract(self, document: DocumentRef) -> ExtractedDocument:
        """Читает документ. Наследник обязан заменить метод.

        Args:
            document: Ссылка на файл.

        Returns:
            ExtractedDocument.

        Raises:
            NotImplementedError: Вызван сам базовый класс.
        """
        raise NotImplementedError

    def read_file(self, path: str) -> bytes:
        """Читает сырые байты файла.

        Args:
            path: Путь из DocumentRef.

        Returns:
            Содержимое файла.

        Raises:
            OSError: Файл не читается.
        """
        return Path(path).read_bytes()

    def digest_bytes(self, payload: bytes) -> str:
        """Считает SHA-256 сырых байтов.

        Args:
            payload: Байты файла до разбора.

        Returns:
            Нижний регистр hex.
        """
        return hashlib.sha256(payload).hexdigest()

    def assemble(
        self,
        doc_id: str,
        digest: str,
        main_blocks: list[ExtractedBlock],
        side_blocks: list[ExtractedBlock],
        has_text_layer: bool,
        failure_reason: str | None,
    ) -> ExtractedDocument:
        """Проставляет пути и номера абзацев и собирает документ.

        Args:
            doc_id: Идентификатор документа.
            digest: source_sha256.
            main_blocks: Заголовки, абзацы и таблицы в порядке чтения.
            side_blocks: Колонтитулы; ставятся после основного текста.
            has_text_layer: Признак текстового слоя.
            failure_reason: Причина отказа или None.

        Returns:
            ExtractedDocument.
        """
        combined = list(main_blocks) + list(side_blocks)
        with_paths = self._apply_section_paths(combined)
        with_indexes = self._apply_paragraph_indexes(with_paths)
        return ExtractedDocument(
            doc_id=doc_id,
            source_sha256=digest,
            has_text_layer=has_text_layer,
            failure_reason=failure_reason,
            blocks=with_indexes,
        )

    def publish(self, extracted: ExtractedDocument) -> ExtractedDocument:
        """Пишет отладочный дамп, если он подключён, и возвращает документ.

        Args:
            extracted: Готовый документ.

        Returns:
            Тот же документ.
        """
        if self.debug_dump is not None:
            self.debug_dump.write(extracted, "extract")
        return extracted

    def _apply_section_paths(
        self,
        blocks: list[ExtractedBlock],
    ) -> list[ExtractedBlock]:
        """Обновляет section_path по заголовкам.

        До первого заголовка путь — None. Колонтитул путь не меняет.

        Args:
            blocks: Блоки в порядке чтения.

        Returns:
            Копии блоков с путями.
        """
        current: list[str] | None = None
        result: list[ExtractedBlock] = []
        for block in blocks:
            if block.kind == "heading":
                level = block.heading_level or 1
                base = [] if current is None else list(current)
                # Обрезаем путь до родителя и добавляем текст этого заголовка.
                trimmed = base[: max(level - 1, 0)]
                trimmed.append(block.text)
                current = trimmed
                result.append(
                    block.model_copy(
                        update={
                            "section_path": list(current),
                            "paragraph_index": None,
                        }
                    )
                )
                continue
            if block.kind in {"paragraph", "table"}:
                path = None if current is None else list(current)
                result.append(block.model_copy(update={"section_path": path}))
                continue
            result.append(
                block.model_copy(
                    update={"section_path": None, "paragraph_index": None}
                )
            )
        return result

    def _apply_paragraph_indexes(
        self,
        blocks: list[ExtractedBlock],
    ) -> list[ExtractedBlock]:
        """Нумерует paragraph и table с 1, не трогая заголовки и колонтитулы.

        Args:
            blocks: Блоки с уже проставленными путями.

        Returns:
            Копии с paragraph_index.
        """
        number = 1
        result: list[ExtractedBlock] = []
        for block in blocks:
            if block.kind in {"paragraph", "table"}:
                result.append(block.model_copy(update={"paragraph_index": number}))
                number += 1
                continue
            result.append(block.model_copy(update={"paragraph_index": None}))
        return result


class DocxExtractor(Extractor):
    """Читает пакет DOCX: стили заголовков, таблицы, сноски, явные разрывы страниц."""

    _HEADING_NAME = re.compile(r"^(?:Heading|Заголовок) ([1-9])$")
    _FOOTNOTES_REL = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"
    )
    _ENDNOTES_REL = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/endnotes"
    )
    _CONTAINERS = {
        qn("w:r"),
        qn("w:hyperlink"),
        qn("w:ins"),
        qn("w:smartTag"),
        qn("w:sdt"),
        qn("w:sdtContent"),
    }

    def extract(self, document: DocumentRef) -> ExtractedDocument:
        """Разбирает DOCX в блоки.

        Args:
            document: Ссылка на файл .docx.

        Returns:
            ExtractedDocument. Пустой пакет — has_text_layer=true и blocks=[].

        Raises:
            Exception: Файл не читается или пакет повреждён.
        """
        payload = self.read_file(document.path)
        digest = self.digest_bytes(payload)
        parsed = Document(BytesIO(payload))
        footnotes = self._note_map(parsed, self._FOOTNOTES_REL)
        endnotes = self._note_map(parsed, self._ENDNOTES_REL)
        page_mode = self._has_explicit_page_break(parsed)
        main_blocks = self._main_blocks(parsed, footnotes, endnotes, page_mode)
        side_blocks = self._side_blocks(parsed, footnotes, endnotes)
        extracted = self.assemble(
            doc_id=document.doc_id,
            digest=digest,
            main_blocks=main_blocks,
            side_blocks=side_blocks,
            has_text_layer=True,
            failure_reason=None,
        )
        return self.publish(extracted)

    def _note_map(self, document: Document, reltype: str) -> dict[str, str]:
        """Читает тексты сносок по id.

        Args:
            document: Открытый пакет.
            reltype: Тип связи footnotes или endnotes.

        Returns:
            Словарь id → текст. Разделительные сноски пропускаются.
        """
        blob = self._related_blob(document, reltype)
        if blob is None:
            return {}
        root = parse_xml(blob)
        notes: dict[str, str] = {}
        for tag in (qn("w:footnote"), qn("w:endnote")):
            for note in root.findall(tag):
                note_type = note.get(qn("w:type"))
                if note_type in {
                    "separator",
                    "continuationSeparator",
                    "continuationNotice",
                }:
                    continue
                note_id = note.get(qn("w:id"))
                if note_id is None:
                    continue
                parts: list[str] = []
                for node in note.iter(qn("w:t")):
                    if node.text:
                        parts.append(node.text)
                notes[note_id] = "".join(parts).strip()
        return notes

    def _related_blob(self, document: Document, reltype: str) -> bytes | None:
        """Достаёт XML связанной части пакета.

        Args:
            document: Открытый пакет.
            reltype: Тип связи OPC.

        Returns:
            Байты части или None.
        """
        for relation in document.part.rels.values():
            if relation.reltype == reltype:
                return relation.target_part.blob
        return None

    def _has_explicit_page_break(self, document: Document) -> bool:
        """Ищет явный разрыв страницы в теле документа.

        Args:
            document: Открытый пакет.

        Returns:
            True, если есть w:br type=page или w:pageBreakBefore.
        """
        for node in document.element.body.iter():
            if node.tag == qn("w:br") and node.get(qn("w:type")) == "page":
                return True
            if node.tag == qn("w:pageBreakBefore"):
                return True
        return False

    def _main_blocks(
        self,
        document: Document,
        footnotes: dict[str, str],
        endnotes: dict[str, str],
        page_mode: bool,
    ) -> list[ExtractedBlock]:
        """Обходит тело в порядке абзацев и таблиц.

        Args:
            document: Открытый пакет.
            footnotes: Тексты сносок.
            endnotes: Тексты концевых сносок.
            page_mode: Нумеровать ли страницы.

        Returns:
            Блоки основного текста.
        """
        blocks: list[ExtractedBlock] = []
        current_page: int | None = 1 if page_mode else None
        for child in document.element.body:
            if child.tag == qn("w:p"):
                paragraph = Paragraph(child, document)
                current_page = self._append_paragraph(
                    blocks,
                    paragraph,
                    footnotes,
                    endnotes,
                    page_mode,
                    current_page,
                )
                continue
            if child.tag == qn("w:tbl"):
                table = Table(child, document)
                text = self._table_text(table)
                if text.strip() == "":
                    continue
                blocks.append(
                    ExtractedBlock(kind="table", text=text, page=current_page)
                )
        return blocks

    def _append_paragraph(
        self,
        blocks: list[ExtractedBlock],
        paragraph: Paragraph,
        footnotes: dict[str, str],
        endnotes: dict[str, str],
        page_mode: bool,
        current_page: int | None,
    ) -> int | None:
        """Добавляет куски абзаца и возвращает страницу после него.

        Args:
            blocks: Накопитель блоков.
            paragraph: Абзац DOCX.
            footnotes: Тексты сносок.
            endnotes: Тексты концевых сносок.
            page_mode: Нумеровать ли страницы.
            current_page: Текущий номер или None.

        Returns:
            Номер страницы для следующих блоков.
        """
        if page_mode and current_page is not None and self._has_page_break_before(
            paragraph
        ):
            current_page += 1
        pieces = self._paragraph_pieces(paragraph, footnotes, endnotes)
        level = self._heading_level(paragraph)
        kind = "heading" if level is not None else "paragraph"
        for index, piece in enumerate(pieces):
            if index > 0 and page_mode and current_page is not None:
                current_page += 1
            if piece == "":
                continue
            blocks.append(
                ExtractedBlock(
                    kind=kind,
                    text=piece,
                    page=current_page,
                    heading_level=level if kind == "heading" else None,
                )
            )
        return current_page

    def _has_page_break_before(self, paragraph: Paragraph) -> bool:
        """Проверяет свойство pageBreakBefore.

        Args:
            paragraph: Абзац.

        Returns:
            True, если абзац начинается с новой страницы.
        """
        properties = paragraph._p.find(qn("w:pPr"))
        if properties is None:
            return False
        return properties.find(qn("w:pageBreakBefore")) is not None

    def _paragraph_pieces(
        self,
        paragraph: Paragraph,
        footnotes: dict[str, str],
        endnotes: dict[str, str],
    ) -> list[str]:
        """Собирает текст абзаца и режет его по разрывам страницы.

        Args:
            paragraph: Абзац.
            footnotes: Тексты сносок.
            endnotes: Тексты концевых сносок.

        Returns:
            Куски текста. Пустая строка между ними — сам разрыв.
        """
        pieces = [""]
        for child in paragraph._p:
            self._consume_node(child, footnotes, endnotes, pieces)
        return pieces

    def _consume_node(
        self,
        node: object,
        footnotes: dict[str, str],
        endnotes: dict[str, str],
        pieces: list[str],
    ) -> None:
        """Добавляет текст узла в текущий кусок абзаца.

        Args:
            node: Узел OOXML.
            footnotes: Тексты сносок.
            endnotes: Тексты концевых сносок.
            pieces: Накопитель кусков.
        """
        tag = node.tag  # type: ignore[attr-defined]
        if tag == qn("w:t"):
            pieces[-1] = pieces[-1] + (node.text or "")  # type: ignore[attr-defined]
            return
        if tag == qn("w:tab"):
            pieces[-1] = pieces[-1] + "\t"
            return
        if tag == qn("w:br"):
            if node.get(qn("w:type")) == "page":  # type: ignore[attr-defined]
                pieces.append("")
                return
            pieces[-1] = pieces[-1] + "\n"
            return
        if tag == qn("w:cr"):
            pieces[-1] = pieces[-1] + "\n"
            return
        if tag == qn("w:footnoteReference"):
            note_id = node.get(qn("w:id"))  # type: ignore[attr-defined]
            note = footnotes.get(note_id or "")
            if note:
                pieces[-1] = pieces[-1] + f" [сноска: {note}] "
            return
        if tag == qn("w:endnoteReference"):
            note_id = node.get(qn("w:id"))  # type: ignore[attr-defined]
            note = endnotes.get(note_id or "")
            if note:
                pieces[-1] = pieces[-1] + f" [сноска: {note}] "
            return
        if tag not in self._CONTAINERS:
            return
        for child in node:  # type: ignore[attr-defined]
            self._consume_node(child, footnotes, endnotes, pieces)

    def _heading_level(self, paragraph: Paragraph) -> int | None:
        """Узнаёт уровень стиля Heading N или Заголовок N.

        Args:
            paragraph: Абзац.

        Returns:
            Уровень 1…9 или None.
        """
        style = paragraph.style
        if style is None:
            return None
        name = style.name or ""
        match = self._HEADING_NAME.fullmatch(name)
        if match is None:
            return None
        return int(match.group(1))

    def _table_text(self, table: Table) -> str:
        """Собирает видимый текст ячеек: строки через перевод, ячейки через ` | `.

        Args:
            table: Таблица DOCX.

        Returns:
            Текст одного блока table.
        """
        rows: list[str] = []
        for row in table.rows:
            cells: list[str] = []
            for cell in row.cells:
                cell_text = cell.text.replace("\n", " ").strip()
                cells.append(cell_text)
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _side_blocks(
        self,
        document: Document,
        footnotes: dict[str, str],
        endnotes: dict[str, str],
    ) -> list[ExtractedBlock]:
        """Собирает непустые колонтитулы, не вставляя их в основной поток.

        Args:
            document: Открытый пакет.
            footnotes: Тексты сносок.
            endnotes: Тексты концевых сносок.

        Returns:
            Блоки header и footer.
        """
        blocks: list[ExtractedBlock] = []
        for section in document.sections:
            blocks.extend(
                self._part_blocks(section.header, "header", footnotes, endnotes)
            )
            blocks.extend(
                self._part_blocks(section.footer, "footer", footnotes, endnotes)
            )
            if section.different_first_page_header_footer:
                blocks.extend(
                    self._part_blocks(
                        section.first_page_header,
                        "header",
                        footnotes,
                        endnotes,
                    )
                )
                blocks.extend(
                    self._part_blocks(
                        section.first_page_footer,
                        "footer",
                        footnotes,
                        endnotes,
                    )
                )
        return blocks

    def _part_blocks(
        self,
        part: object,
        kind: str,
        footnotes: dict[str, str],
        endnotes: dict[str, str],
    ) -> list[ExtractedBlock]:
        """Читает абзацы одной части колонтитула.

        Args:
            part: Header или Footer.
            kind: header или footer.
            footnotes: Тексты сносок.
            endnotes: Тексты концевых сносок.

        Returns:
            Непустые блоки.
        """
        if part.is_linked_to_previous:  # type: ignore[attr-defined]
            return []
        blocks: list[ExtractedBlock] = []
        for paragraph in part.paragraphs:  # type: ignore[attr-defined]
            pieces = self._paragraph_pieces(paragraph, footnotes, endnotes)
            text = "".join(piece for piece in pieces if piece != "")
            if text.strip() == "":
                continue
            blocks.append(ExtractedBlock(kind=kind, text=text))  # type: ignore[arg-type]
        return blocks


class PdfExtractor(Extractor):
    """Читает текстовый слой PDF постранично. OCR в этой итерации нет.

    Attributes:
        pdf_text_layer_required: Флаг политики. Без OCR оба значения
            дают no_text_layer на полностью пустом документе.
    """

    _HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")

    def __init__(
        self,
        pdf_text_layer_required: bool = True,
        debug_dump: BlockDocumentDebugDump | None = None,
    ) -> None:
        """Сохраняет флаг текстового слоя и дамп.

        Args:
            pdf_text_layer_required: Требовать ли текстовый слой.
            debug_dump: Писатель Markdown или None.
        """
        super().__init__(debug_dump)
        self.pdf_text_layer_required = pdf_text_layer_required

    def extract(self, document: DocumentRef) -> ExtractedDocument:
        """Разбирает PDF. Пустой слой не бросает исключение.

        Args:
            document: Ссылка на файл .pdf.

        Returns:
            ExtractedDocument. Если текста нет ни на одной странице —
            failure_reason=no_text_layer.

        Raises:
            Exception: Файл не читается или повреждён.
        """
        payload = self.read_file(document.path)
        digest = self.digest_bytes(payload)
        reader = PdfReader(BytesIO(payload))
        page_texts = [page.extract_text() or "" for page in reader.pages]
        if not any(self._has_visible_text(text) for text in page_texts):
            extracted = self.assemble(
                doc_id=document.doc_id,
                digest=digest,
                main_blocks=[],
                side_blocks=[],
                has_text_layer=False,
                failure_reason=self._empty_failure_reason(),
            )
            return self.publish(extracted)
        blocks: list[ExtractedBlock] = []
        for page_number, text in enumerate(page_texts, start=1):
            if not self._has_visible_text(text):
                continue
            blocks.extend(self._blocks_on_page(text, page_number))
        extracted = self.assemble(
            doc_id=document.doc_id,
            digest=digest,
            main_blocks=blocks,
            side_blocks=[],
            has_text_layer=True,
            failure_reason=None,
        )
        return self.publish(extracted)

    def _empty_failure_reason(self) -> str:
        """Возвращает причину полностью пустого PDF.

        Returns:
            Всегда no_text_layer: OCR в итерации 1 нет и при флаге false.
        """
        if self.pdf_text_layer_required:
            return "no_text_layer"
        return "no_text_layer"

    def _has_visible_text(self, text: str) -> bool:
        """Проверяет, остался ли текст после снятия пробелов.

        Args:
            text: Текст страницы.

        Returns:
            True, если есть непробельный символ.
        """
        for char in text:
            if not char.isspace():
                return True
        return False

    def _blocks_on_page(self, text: str, page_number: int) -> list[ExtractedBlock]:
        """Делит страницу на заголовки и абзацы.

        Пустая строка начинает новый абзац. Перенос нумерованной строки
        остаётся в заголовке только первой строкой, а абзац получает фразу целиком.

        Args:
            text: Текст страницы из текстового слоя.
            page_number: Номер страницы с 1.

        Returns:
            Блоки этой страницы.
        """
        blocks: list[ExtractedBlock] = []
        paragraph_lines: list[str] = []
        lines = text.splitlines()
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped == "":
                self._flush_paragraph(blocks, paragraph_lines, page_number)
                index += 1
                continue
            level = self._heading_level(stripped)
            if level is None:
                paragraph_lines.append(stripped)
                index += 1
                continue
            self._flush_paragraph(blocks, paragraph_lines, page_number)
            blocks.append(
                ExtractedBlock(
                    kind="heading",
                    text=stripped,
                    page=page_number,
                    heading_level=level,
                )
            )
            continuation, index = self._take_heading_continuation(lines, index)
            if not continuation:
                continue
            # Первая строка нужна и в пути раздела, и в цитируемом абзаце.
            full_text = self._join_lines([stripped, *continuation])
            blocks.append(
                ExtractedBlock(
                    kind="paragraph",
                    text=full_text,
                    page=page_number,
                )
            )
        self._flush_paragraph(blocks, paragraph_lines, page_number)
        return blocks

    def _flush_paragraph(
        self,
        blocks: list[ExtractedBlock],
        lines: list[str],
        page_number: int,
    ) -> None:
        """Кладёт накопленные строки одним абзацем и очищает накопитель.

        Args:
            blocks: Уже собранные блоки страницы.
            lines: Строки текущего абзаца. Очищается здесь.
            page_number: Номер страницы с 1.
        """
        if not lines:
            return
        paragraph_text = self._join_lines(lines)
        lines.clear()
        if paragraph_text == "":
            return
        blocks.append(
            ExtractedBlock(
                kind="paragraph",
                text=paragraph_text,
                page=page_number,
            )
        )

    def _take_heading_continuation(
        self,
        lines: list[str],
        heading_index: int,
    ) -> tuple[list[str], int]:
        """Забирает строки переноса сразу после заголовка.

        Args:
            lines: Все строки страницы, без обрезки краёв.
            heading_index: Индекс первой строки заголовка.

        Returns:
            Строки продолжения и индекс первой строки, которая к ним не относится.
            Без продолжения индекс указывает на строку сразу после заголовка.
        """
        continuation: list[str] = []
        previous = lines[heading_index].strip()
        index = heading_index + 1
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped == "":
                break
            if self._heading_level(stripped) is not None:
                break
            if not self._continues_heading(previous, stripped):
                break
            continuation.append(stripped)
            previous = stripped
            index += 1
        if not continuation:
            return [], heading_index + 1
        return continuation, index

    def _continues_heading(self, previous: str, line: str) -> bool:
        """Проверяет, что строка продолжает фразу заголовка, а не новый абзац.

        Args:
            previous: Предыдущая строка фразы без краевых пробелов.
            line: Кандидат без краевых пробелов.

        Returns:
            True, если строка — перенос: со строчной буквы или после дефиса.
        """
        if previous.endswith("-"):
            return True
        first = line[:1]
        return first.islower()

    def _join_lines(self, lines: list[str]) -> str:
        """Склеивает строки абзаца, снимая перенос слова на дефисе.

        Args:
            lines: Строки уже без краевых пробелов.

        Returns:
            Одна строка. Дефис перед строчной буквой исчезает, иначе между
            строками один пробел.
        """
        if not lines:
            return ""
        merged = lines[0]
        for line in lines[1:]:
            # «информа-» + «ционного» — один перенос, не два слова.
            if merged.endswith("-") and line[:1].islower():
                merged = merged[:-1] + line
                continue
            merged = f"{merged} {line}"
        return merged.strip()

    def _heading_level(self, line: str) -> int | None:
        """Уровень нумерованного заголовка PDF.

        Args:
            line: Строка без краевых пробелов.

        Returns:
            Число компонент номера или None.
        """
        match = self._HEADING_RE.match(line)
        if match is None:
            return None
        components = match.group(1).split(".")
        return len(components)


class FormatAwareExtractor(Extractor):
    """Выбирает извлекатель только по суффиксу файла.

    Attributes:
        docx_extractor: Извлекатель DOCX.
        pdf_extractor: Извлекатель PDF.
    """

    def __init__(
        self,
        docx_extractor: DocxExtractor | None = None,
        pdf_extractor: PdfExtractor | None = None,
        debug_dump: BlockDocumentDebugDump | None = None,
    ) -> None:
        """Собирает наследников. Свой дамп передаётся им, если они не заданы снаружи.

        Args:
            docx_extractor: Готовый DocxExtractor или None.
            pdf_extractor: Готовый PdfExtractor или None.
            debug_dump: Дамп для создаваемых по умолчанию наследников.
        """
        super().__init__(debug_dump)
        self.docx_extractor = docx_extractor or DocxExtractor(debug_dump=debug_dump)
        self.pdf_extractor = pdf_extractor or PdfExtractor(debug_dump=debug_dump)

    def extract(self, document: DocumentRef) -> ExtractedDocument:
        """Отдаёт файл .docx или .pdf соответствующему наследнику.

        Args:
            document: Ссылка на файл.

        Returns:
            ExtractedDocument выбранного наследника.

        Raises:
            ValueError: Суффикс не .docx и не .pdf.
        """
        suffix = document.suffix.lower()
        if suffix == ".docx":
            return self.docx_extractor.extract(document)
        if suffix == ".pdf":
            return self.pdf_extractor.extract(document)
        raise ValueError(f"Неизвестный суффикс: {suffix}")
