"""TraceSink на structured log."""

from __future__ import annotations

import logging

from models.observe import TraceEvent
from ports.trace import TraceSink


class LoggingTraceSink(TraceSink):
    """Пишет TraceEvent в стандартный logger.

    Attributes:
        logger: Logger для событий.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Сохраняет logger.

        Args:
            logger: Целевой logger; по умолчанию observe.trace.
        """
        self.logger = logger or logging.getLogger("observe.trace")

    def emit(self, event: TraceEvent) -> None:
        """Логирует событие как JSON-подобную запись.

        Args:
            event: TraceEvent.
        """
        self.logger.info(
            "trace stage=%s status=%s duration_ms=%s doc_id=%s question_id=%s error=%s",
            event.stage,
            event.status,
            event.duration_ms,
            event.doc_id,
            event.question_id,
            event.error,
        )
