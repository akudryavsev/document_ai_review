"""T-OBS: декоратор Observed и LoggingTraceSink."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel, ValidationError

from fakes.trace import RecordingTraceSink
from models.config import RuntimeConfig
from models.observe import TraceEvent
from observe import Observed
from observe.logging_sink import LoggingTraceSink


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


class ListLogHandler(logging.Handler):
    """Копит LogRecord для проверки уровня sink.

    Attributes:
        records: Записи, которые принял handler.
    """

    def __init__(self) -> None:
        """Создаёт пустой список записей."""
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Сохраняет запись.

        Args:
            record: LogRecord.
        """
        self.records.append(record)


class TestObserved:
    """Каталог T-OBS-01…05."""

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

    def test_t_obs_04_failed_is_warning_and_debug_payload_is_gated(self) -> None:
        """failed пишется как warning, debug_payload виден только на DEBUG."""
        logger = logging.getLogger("observe.trace.t_obs_04")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = ListLogHandler()
        logger.addHandler(handler)
        sink = LoggingTraceSink(logger)
        sink.emit(
            TraceEvent(
                stage="A1",
                status="failed",
                doc_id="doc.pdf",
                error="no_text_layer",
                debug_payload="скрытый текст",
            )
        )
        sink.emit(
            TraceEvent(
                stage="A1",
                status="ok",
                debug_payload="скрытый текст",
            )
        )
        assert handler.records[0].levelno == logging.WARNING
        assert "no_text_layer" in handler.records[0].getMessage()
        assert "скрытый текст" not in handler.records[0].getMessage()
        assert handler.records[1].levelno == logging.INFO
        assert "скрытый текст" not in handler.records[1].getMessage()
        logger.setLevel(logging.DEBUG)
        sink.emit(
            TraceEvent(
                stage="A1",
                status="ok",
                debug_payload="видимый текст",
            )
        )
        assert "видимый текст" in handler.records[2].getMessage()

    def test_t_obs_05_unknown_log_level_rejected(self) -> None:
        """Чужое имя уровня не проходит конфигурацию."""
        with pytest.raises(ValidationError, match="log_level"):
            RuntimeConfig(log_level="VERBOSE")
