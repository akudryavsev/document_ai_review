"""Порт приёмника событий наблюдаемости."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.observe import TraceEvent


class TraceSink(ABC):
    """Приёмник TraceEvent."""

    @abstractmethod
    def emit(self, event: TraceEvent) -> None:
        """Принимает событие стадии.

        Args:
            event: TraceEvent.
        """
