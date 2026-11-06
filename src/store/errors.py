"""Ошибки связки текста и вектора."""

from __future__ import annotations

from models.artifacts import EmbeddingBindingReport


class EmbeddingBindingError(Exception):
    """Текст чанка или вопроса не сходится с рядом вектора.

    Attributes:
        report: Подробный отчёт сверки.
    """

    def __init__(self, report: EmbeddingBindingReport) -> None:
        """Сохраняет отчёт и собирает короткое сообщение.

        Args:
            report: Отчёт с ok=false.
        """
        self.report = report
        parts: list[str] = []
        if report.size_error:
            parts.append(report.size_error)
        if report.mismatched_ids:
            parts.append("чужие идентификаторы")
        if report.mismatched_hashes:
            parts.append("разные text_sha256")
        message = "; ".join(parts) or "рассинхрон эмбеддингов"
        super().__init__(message)
