"""Markdown-дамп собранного ContextBundle."""

from __future__ import annotations

from pathlib import Path

from models.artifacts import ContextBundle, ContextCandidate


class ContextDebugDump:
    """Пишет `{cache}/debug/{doc_id}/{question_id}.context.md`.

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

    def path_for(self, doc_id: str, question_id: str) -> Path:
        """Строит путь файла дампа.

        Args:
            doc_id: Документ.
            question_id: Вопрос.

        Returns:
            Путь к Markdown.
        """
        return Path(self.cache_dir) / "debug" / doc_id / f"{question_id}.context.md"

    def render(self, bundle: ContextBundle) -> str:
        """Собирает Markdown бандла без экранирования текста.

        Args:
            bundle: Бандл после проставления хешей фазой.

        Returns:
            Текст файла.
        """
        lines = [
            f"# context: {bundle.doc_id} · {bundle.question_id}",
            "",
            f"- empty_search: {str(bundle.empty_search).lower()}",
            f"- reason: {self._reason(bundle)}",
            f"- category: {bundle.category}",
            f"- budget_used_tokens: {bundle.budget_used_tokens}",
            f"- top_k: {bundle.retrieval_meta.get('top_k', '')}",
            f"- after_rerank: {bundle.retrieval_meta.get('after_rerank', '')}",
            f"- sections_used: {self._sections(bundle)}",
            f"- chunking_hash: {bundle.chunking_hash}",
            f"- retrieval_config_hash: {bundle.retrieval_config_hash}",
            f"- reranker_config_hash: {bundle.reranker_config_hash}",
            "",
            "## Синонимы",
            "",
        ]
        lines.extend(self._synonym_lines(bundle))
        lines.extend(["", "## Фрагменты", ""])
        if not bundle.candidates:
            lines.append("Фрагментов нет.")
            lines.append("")
            return "\n".join(lines)
        for candidate in bundle.candidates:
            lines.extend(self._fragment_lines(candidate))
        return "\n".join(lines)

    def write(self, bundle: ContextBundle) -> str | None:
        """Записывает дамп, если флаг включён.

        Args:
            bundle: Бандл для снимка.

        Returns:
            Путь файла или None при выключенном флаге.
        """
        if not self.enabled:
            return None
        path = self.path_for(bundle.doc_id, bundle.question_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(bundle), encoding="utf-8")
        return str(path)

    def _reason(self, bundle: ContextBundle) -> str:
        """Печатает причину или null.

        Args:
            bundle: Бандл.

        Returns:
            Строка для шапки.
        """
        if bundle.reason is None:
            return "null"
        return bundle.reason

    def _sections(self, bundle: ContextBundle) -> str:
        """Склеивает sections_used или ставит прочерк.

        Args:
            bundle: Бандл.

        Returns:
            Строка шапки.
        """
        raw = bundle.retrieval_meta.get("sections_used") or []
        if not raw:
            return "—"
        return " | ".join(str(item) for item in raw)

    def _synonym_lines(self, bundle: ContextBundle) -> list[str]:
        """Строки секции синонимов в исходном порядке.

        Args:
            bundle: Бандл.

        Returns:
            Маркированный список или один прочерк.
        """
        if not bundle.used_synonyms:
            return ["- —"]
        return [f"- {item}" for item in bundle.used_synonyms]

    def _fragment_lines(self, candidate: ContextCandidate) -> list[str]:
        """Секция одного фрагмента и его текст как есть.

        Args:
            candidate: Элемент бандла.

        Returns:
            Строки Markdown, включая тело.
        """
        truncated = "true" if candidate.truncated else "false"
        path = "—"
        if candidate.section_path:
            path = " / ".join(candidate.section_path)
        chunks = ", ".join(candidate.chunks_included)
        heading = (
            f"### Фрагмент {candidate.rank_after_rerank}"
            f" · rank {candidate.rank_after_rerank}"
            f" · score {candidate.rerank_score:g}"
            f" · {self._pages(candidate.pages)}"
            f" · truncated {truncated}"
        )
        return [
            heading,
            f"path: {path}",
            f"chunks: {chunks}",
            "",
            candidate.text,
            "",
        ]

    def _pages(self, pages: list[int]) -> str:
        """Форматирует страницы фрагмента.

        Args:
            pages: Номера по возрастанию.

        Returns:
            Подпись вида стр. 12–13, стр. 12 или стр. —.
        """
        if not pages:
            return "стр. —"
        if len(pages) == 1:
            return f"стр. {pages[0]}"
        return f"стр. {pages[0]}–{pages[-1]}"
