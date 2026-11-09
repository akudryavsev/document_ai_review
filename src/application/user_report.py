"""Короткий ответ для чтения: вопрос, ответ и пояснение."""

from __future__ import annotations

from pathlib import Path

from models.artifacts import AnswerRecord


class UserAnswerReport:
    """Собирает results/<doc_id>.md из строк ответа.

    Attributes:
        missing_context: Фраза, если контекст пары не собран.
        unavailable: Фраза, если модель не дала годный JSON.
        empty: Фраза для остальных пустых ответов.
    """

    missing_context = "Контекст для ответа не собран."
    unavailable = "Модель недоступна."
    empty = "Ответа нет."

    def render(self, records: list[AnswerRecord]) -> str:
        """Собирает Markdown без цитат и промпта.

        Args:
            records: Строки одного документа в порядке jsonl.

        Returns:
            Текст файла с завершающим переводом строки.
        """
        counts: dict[str, int] = {}
        for record in records:
            counts[record.category] = counts.get(record.category, 0) + 1
        blocks: list[str] = []
        for record in records:
            lines = [f"## {record.category}", ""]
            if counts[record.category] > 1:
                lines.append(record.answer_model)
                lines.append("")
            lines.append(self._paragraph(record))
            explanation = self._explanation(record)
            if explanation:
                lines.append("")
                lines.append(explanation)
            blocks.append("\n".join(lines))
        if not blocks:
            return "\n"
        return "\n\n".join(blocks) + "\n"

    def write(self, records: list[AnswerRecord], path: Path) -> str:
        """Перезаписывает файл ответа.

        Args:
            records: Строки документа.
            path: Путь results/<doc_id>.md.

        Returns:
            Строковый путь.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(records), encoding="utf-8")
        return str(path)

    def _paragraph(self, record: AnswerRecord) -> str:
        """Выбирает абзац ответа.

        Args:
            record: Строка jsonl.

        Returns:
            Непустой текст для читателя. Поле answer записи не меняется.
        """
        if record.answer:
            return record.answer
        if record.reason == "missing_context":
            return self.missing_context
        if record.reason == "model_unavailable":
            return self.unavailable
        return self.empty

    def _explanation(self, record: AnswerRecord) -> str:
        """Возвращает пояснение для md.

        Args:
            record: Строка jsonl.

        Returns:
            Обрезанный explanation или пустая строка, если пояснения нет.
        """
        return record.explanation.strip()
