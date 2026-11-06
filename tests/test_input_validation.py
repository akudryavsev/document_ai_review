"""T-IN: валидация файла вопросов."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from models.questions import Question, QuestionSet
from ports.file_sources import FileQuestionSource
from ports.sources import QuestionSource
from tests.helpers import ServiceFactory
from tests.support import build_harness


class RaisingQuestionSource(QuestionSource):
    """Источник, который падает при любой загрузке.

    Attributes:
        calls: Сколько раз вызвали load.
    """

    def __init__(self) -> None:
        """Начинает счётчик вызовов с нуля."""
        self.calls = 0

    def load(self) -> QuestionSet:
        """Фиксирует вызов и отказывается отдать вопросы.

        Returns:
            Не возвращает набор.

        Raises:
            AssertionError: Файл вопросов для этого прогона не нужен.
        """
        self.calls += 1
        raise AssertionError("файл вопросов не нужен для фазы A1")


class TestInputValidation:
    """Каталог T-IN-01…04."""

    def test_t_in_01_empty_questions_file(self, tmp_path: Path) -> None:
        """Пустой список вопросов — ошибка загрузки до старта фаз."""
        path = tmp_path / "questions.json"
        path.write_text(
            json.dumps({"version": "1.0", "questions": []}),
            encoding="utf-8",
        )
        source = FileQuestionSource(path)
        with pytest.raises((ValueError, ValidationError)):
            source.load()

    def test_t_in_02_duplicate_id(self, tmp_path: Path) -> None:
        """Дубликат id — ошибка загрузки."""
        path = tmp_path / "questions.json"
        payload = {
            "version": "1.0",
            "questions": [
                {
                    "id": "Q001",
                    "category": "A",
                    "synonyms": [],
                },
                {
                    "id": "Q001",
                    "category": "B",
                    "synonyms": [],
                },
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        source = FileQuestionSource(path)
        with pytest.raises((ValueError, ValidationError)):
            source.load()

    def test_t_in_03_duplicate_category(self, tmp_path: Path) -> None:
        """Дубликат category — ошибка загрузки."""
        path = tmp_path / "questions.json"
        payload = {
            "version": "1.0",
            "questions": [
                {
                    "id": "Q001",
                    "category": "Same",
                    "synonyms": [],
                },
                {
                    "id": "Q002",
                    "category": "Same",
                    "synonyms": [],
                },
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        source = FileQuestionSource(path)
        with pytest.raises((ValueError, ValidationError)):
            source.load()

    def test_t_in_04_empty_synonyms_allowed(self) -> None:
        """Пустой synonyms допустим: один вариант запроса = category."""
        question_set = QuestionSet(
            questions=[
                Question(id="Q001", category="Срок", synonyms=[]),
            ]
        )
        assert question_set.questions[0].synonyms == []
        assert question_set.questions[0].category == "Срок"

    def test_t_in_05_phase_a1_without_questions(self) -> None:
        """Только A1 стартует без файла вопросов, пишет чанки и не трогает кэш вопросов."""
        harness = build_harness()
        question_source = RaisingQuestionSource()
        harness.question_source = question_source
        service = ServiceFactory().build(harness)
        summary = service.run(harness.run_request(phases=["A1"]))
        first_batch = harness.store.load_chunks("doc_a.docx")
        second_batch = harness.store.load_chunks("doc_b.pdf")
        assert question_source.calls == 0
        assert summary.questions == 0
        assert summary.phases["A1"].computed == 2
        assert first_batch is not None and first_batch.chunks
        assert second_batch is not None and second_batch.chunks
        assert harness.store.has_questions_cache() is False
