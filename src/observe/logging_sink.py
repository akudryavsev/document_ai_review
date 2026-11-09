"""TraceSink на structured log."""

from __future__ import annotations

import logging

from models.observe import TraceEvent
from ports.trace import TraceSink


class LoggingTraceSink(TraceSink):
    """Пишет TraceEvent в стандартный logger.

    Отказ документа идёт как warning, границы стадий — как info.
    debug_payload добавляется в строку только при уровне DEBUG.

    Attributes:
        logger: Logger для событий.
        levels: Допустимые имена runtime.log_level.
    """

    levels = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Сохраняет logger.

        Args:
            logger: Целевой logger; по умолчанию observe.trace.
        """
        self.logger = logger or logging.getLogger("observe.trace")

    def configure(self, level_name: str) -> None:
        """Ставит уровень и обработчик консоли на логгер трассы.

        Args:
            level_name: Имя из levels.

        Raises:
            ValueError: Имя уровня не из пятёрки.
        """
        if level_name not in self.levels:
            names = ", ".join(sorted(self.levels))
            raise ValueError(f"runtime.log_level должен быть одним из: {names}")
        level = getattr(logging, level_name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setLevel(level)
        self.logger.addHandler(handler)

    def emit(self, event: TraceEvent) -> None:
        """Логирует событие. failed — warning, остальные статусы — info.

        Args:
            event: TraceEvent.
        """
        message = (
            "trace stage=%s status=%s duration_ms=%s doc_id=%s "
            "question_id=%s error=%s"
        )
        values: list[object] = [
            event.stage,
            event.status,
            event.duration_ms,
            event.doc_id,
            event.question_id,
            event.error,
        ]
        if event.debug_payload and self.logger.isEnabledFor(logging.DEBUG):
            message = f"{message} debug_payload=%s"
            values.append(event.debug_payload)
        if event.status == "failed":
            self.logger.warning(message, *values)
            return
        self.logger.info(message, *values)
