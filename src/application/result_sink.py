"""Файловая запись результатов прогона."""

from __future__ import annotations

import json
from pathlib import Path

from application.user_report import UserAnswerReport
from models.artifacts import AnswerRecord
from models.summary import RunSummary
from ports.pipeline import ResultSink


class FileResultSink(ResultSink):
    """Пишет jsonl/md/summary в каталог results.

    Attributes:
        results_dir: Корневой каталог результатов.
        answers: Журнал записанных ответов (для сводки).
    """

    def __init__(self, results_dir: str | Path) -> None:
        """Создаёт каталог результатов.

        Args:
            results_dir: Путь results/.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.answers: list[AnswerRecord] = []

    def write_answer(self, record: AnswerRecord) -> None:
        """Заменяет строку того же документа, вопроса и тройки и пересобирает md.

        Args:
            record: AnswerRecord.
        """
        self.answers = [
            item
            for item in self.answers
            if not self._same_triple(item, record)
        ]
        self.answers.append(record)
        path = self.results_dir / f"{record.doc_id}.jsonl"
        kept = self._kept_records(path, record)
        kept.append(record)
        lines = [item.model_dump_json() for item in kept]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        UserAnswerReport().write(kept, self.results_dir / f"{record.doc_id}.md")

    def _kept_records(self, path: Path, record: AnswerRecord) -> list[AnswerRecord]:
        """Оставляет записи файла, кроме той же тройки и вопроса.

        Args:
            path: Файл jsonl документа.
            record: Новая запись.

        Returns:
            Записи, которые остаются в файле, в прежнем порядке.
        """
        if not path.is_file():
            return []
        kept: list[AnswerRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            existing = AnswerRecord.model_validate_json(line)
            if self._same_triple(existing, record):
                continue
            kept.append(existing)
        return kept

    def _same_triple(self, left: AnswerRecord, right: AnswerRecord) -> bool:
        """Сравнивает документ, вопрос и тройку моделей.

        Args:
            left: Уже записанная строка.
            right: Новая строка.

        Returns:
            True, если это та же пара и та же тройка.
        """
        return (
            left.doc_id == right.doc_id
            and left.question_id == right.question_id
            and left.embedding_model == right.embedding_model
            and left.reranker_model == right.reranker_model
            and left.answer_model == right.answer_model
        )

    def write_summary(self, summary: object) -> None:
        """Пишет run_summary.json.

        Args:
            summary: RunSummary.
        """
        assert isinstance(summary, RunSummary)
        path = self.results_dir / "run_summary.json"
        path.write_text(
            json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
