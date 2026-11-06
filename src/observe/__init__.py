"""Наблюдаемость публичных методов."""

from observe.decorator import Observed, ObservedMethod
from observe.logging_sink import LoggingTraceSink

__all__ = ["LoggingTraceSink", "Observed", "ObservedMethod"]
