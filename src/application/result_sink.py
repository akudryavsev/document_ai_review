"""Файловая запись результатов прогона."""

from __future__ import annotations

import json
from pathlib import Path

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
        """Дописывает строку в results/<doc_id>.jsonl.

        Args:
            record: AnswerRecord.
        """
        self.answers.append(record)
        path = self.results_dir / f"{record.doc_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

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
