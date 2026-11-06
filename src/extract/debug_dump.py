"""Человекочитаемый Markdown-снимок блоков документа."""

from __future__ import annotations

from pathlib import Path

from models.documents import CleanedDocument, ExtractedBlock, ExtractedDocument


class BlockDocumentDebugDump:
    """Пишет `{cache}/debug/{doc_id}.{stage}.md`.

    Attributes:
        enabled: При false write ничего не создаёт.
        cache_dir: Корень, внутри которого лежит каталог debug.
    """

    def __init__(self, enabled: bool = False, cache_dir: str = "./.cache") -> None:
        """Сохраняет флаг и каталог.

        Args:
            enabled: Писать ли файл.
            cache_dir: runtime.cache_dir.
        """
        self.enabled = enabled
        self.cache_dir = cache_dir

    def render(
        self,
        document: ExtractedDocument | CleanedDocument,
        stage: str,
    ) -> str:
        """Собирает Markdown без экранирования текста блоков.

        Args:
            document: Документ с блоками.
            stage: `extract` или `clean`.

        Returns:
            Текст файла дампа.
        """
        failure = getattr(document, "failure_reason", None)
        failure_text = "null" if failure is None else str(failure)
        has_layer = getattr(document, "has_text_layer", True)
        lines = [
            f"# {stage}: {document.doc_id}",
            "",
            f"- source_sha256: {document.source_sha256}",
            f"- has_text_layer: {str(has_layer).lower()}",
            f"- failure_reason: {failure_text}",
            f"- blocks: {len(document.blocks)}",
            "",
            "## Блоки",
            "",
        ]
        for index, block in enumerate(document.blocks, start=1):
            lines.append(self._section_heading(index, block))
            lines.append(f"path: {self._path_line(block)}")
            lines.append("")
            lines.append(block.text)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def write(
        self,
        document: ExtractedDocument | CleanedDocument,
        stage: str,
    ) -> str | None:
        """Пишет дамп, если флаг включён.

        Args:
            document: Документ с блоками.
            stage: `extract` или `clean`.

        Returns:
            Путь к файлу или None, если дамп выключен.
        """
        if not self.enabled:
            return None
        directory = Path(self.cache_dir) / "debug"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{document.doc_id}.{stage}.md"
        path.write_text(self.render(document, stage), encoding="utf-8")
        return str(path)

    def _section_heading(self, index: int, block: ExtractedBlock) -> str:
        """Заголовок секции одного блока.

        Args:
            index: Порядковый номер в файле с 1.
            block: Блок.

        Returns:
            Строка `###`.
        """
        page = "—" if block.page is None else str(block.page)
        paragraph = "—" if block.paragraph_index is None else str(block.paragraph_index)
        if block.kind == "heading":
            level = block.heading_level or 0
            return (
                f"### [{index}] heading · lvl {level} · стр. {page} · абз. {paragraph}"
            )
        return f"### [{index}] {block.kind} · стр. {page} · абз. {paragraph}"

    def _path_line(self, block: ExtractedBlock) -> str:
        """Форматирует section_path для строки path.

        Args:
            block: Блок.

        Returns:
            Элементы через ` / ` или длинное тире.
        """
        if not block.section_path:
            return "—"
        return " / ".join(block.section_path)
