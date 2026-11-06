"""T-OBS: декоратор Observed."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from fakes.trace import RecordingTraceSink
from observe import Observed


class SamplePayload(BaseModel):
    """Модель с doc_id и question_id для извлечения полей."""

    doc_id: str
    question_id: str


class SampleService:
    """Сервис для проверки Observed."""

    def __init__(self, trace_sink: RecordingTraceSink) -> None:
        """Сохраняет TraceSink.

        Args:
            trace_sink: RecordingTraceSink.
        """
        self.trace_sink = trace_sink

    @Observed(stage="sample_ok")
    def succeed(self, payload: SamplePayload) -> str:
        """Успешный метод.

        Args:
            payload: Модель с идентификаторами.

        Returns:
            Строка ok.
        """
        return "ok"

    @Observed(stage="sample_fail")
    def fail(self, payload: SamplePayload) -> str:
        """Падающий метод.

        Args:
            payload: Модель с идентификаторами.

        Returns:
            Не возвращает — всегда бросает.

        Raises:
            RuntimeError: Всегда.
        """
        raise RuntimeError("boom")


class TestObserved:
    """Каталог T-OBS-01…02."""

    def test_t_obs_01_success_emits_started_and_ok(self) -> None:
        """Успешный метод пишет started и ok с duration_ms."""
        sink = RecordingTraceSink()
        service = SampleService(sink)
        result = service.succeed(SamplePayload(doc_id="d1", question_id="Q1"))
        assert result == "ok"
        statuses = [event.status for event in sink.events if event.stage == "sample_ok"]
        assert statuses == ["started", "ok"]
        ok_event = next(event for event in sink.events if event.status == "ok")
        assert ok_event.duration_ms is not None
        assert ok_event.doc_id == "d1"
        assert ok_event.question_id == "Q1"

    def test_t_obs_02_failure_emits_failed_and_reraises(self) -> None:
        """Падение пишет failed с error и пробрасывает исключение."""
        sink = RecordingTraceSink()
        service = SampleService(sink)
        with pytest.raises(RuntimeError, match="boom"):
            service.fail(SamplePayload(doc_id="d1", question_id="Q1"))
        failed = next(event for event in sink.events if event.status == "failed")
        assert failed.stage == "sample_fail"
        assert failed.error == "boom"
        assert failed.duration_ms is not None
