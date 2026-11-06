"""Подделки TraceSink."""

from __future__ import annotations

from models.observe import TraceEvent
from ports.trace import TraceSink


class RecordingTraceSink(TraceSink):
    """Записывает все TraceEvent в список.

    Attributes:
        events: Журнал событий.
    """

    def __init__(self) -> None:
        """Инициализирует пустой журнал."""
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        """Добавляет событие в журнал.

        Args:
            event: TraceEvent.
        """
        self.events.append(event)
