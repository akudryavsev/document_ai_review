"""Отладочный Markdown нарезки документа."""

from __future__ import annotations

from pathlib import Path

from models.documents import ChunkRecord


class ChunkDebugDump:
    """Пишет `{cache}/debug/{doc_id}.chunks.md`.

    Attributes:
        enabled: При false write ничего не создаёт.
        cache_dir: Корень с каталогом debug.
    """

    def __init__(self, enabled: bool = False, cache_dir: str = "./.cache") -> None:
        """Сохраняет флаг и каталог.

        Args:
            enabled: Писать ли файл.
            cache_dir: runtime.cache_dir.
        """
        self.enabled = enabled
        self.cache_dir = cache_dir

    def render(self, doc_id: str, chunks: list[ChunkRecord]) -> str:
        """Собирает Markdown чанков по chunk_index.

        Args:
            doc_id: Документ.
            chunks: Чанки документа, возможно пустые.

        Returns:
            Текст файла.
        """
        if not chunks:
            return f"# chunks: {doc_id}\n\nЧанков нет.\n"
        ordered = sorted(chunks, key=lambda chunk: chunk.chunk_index)
        sections: list[str] = []
        for chunk in ordered:
            if chunk.section_path:
                section = " / ".join(chunk.section_path)
            else:
                section = "null"
            block = "\n".join(
                [
                    f"## {chunk.chunk_id}",
                    "",
                    f"- pages: {chunk.page_from}–{chunk.page_to}",
                    f"- section: {section}",
                    "",
                    chunk.text,
                ]
            )
            sections.append(block)
        body = "\n\n".join(sections)
        return f"# chunks: {doc_id}\n\n{body}\n"

    def write(self, doc_id: str, chunks: list[ChunkRecord]) -> str | None:
        """Пишет дамп, если флаг включён.

        Args:
            doc_id: Документ.
            chunks: Чанки, в том числе пустой список.

        Returns:
            Путь к файлу или None, если дамп выключен.
        """
        if not self.enabled:
            return None
        directory = Path(self.cache_dir) / "debug"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{doc_id}.chunks.md"
        path.write_text(self.render(doc_id, chunks), encoding="utf-8")
        return str(path)
