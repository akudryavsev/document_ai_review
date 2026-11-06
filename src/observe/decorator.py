"""Декоратор наблюдаемости публичных методов."""

from __future__ import annotations

import time
from typing import Any, Callable

from models.observe import TraceEvent
from ports.trace import TraceSink


class Observed:
    """Класс-декоратор: пишет started/ok/failed в TraceSink.

    Attributes:
        stage: Имя стадии.
        trace_sink: Приёмник событий; может быть задан позже через bind.
    """

    def __init__(self, stage: str, trace_sink: TraceSink | None = None) -> None:
        """Сохраняет имя стадии и опциональный sink.

        Args:
            stage: Имя стадии.
            trace_sink: TraceSink или None до bind.
        """
        self.stage = stage
        self.trace_sink = trace_sink
        self._func: Callable[..., Any] | None = None

    def bind(self, trace_sink: TraceSink) -> None:
        """Привязывает TraceSink после сборки графа.

        Args:
            trace_sink: Приёмник событий.
        """
        self.trace_sink = trace_sink

    def __call__(self, func: Callable[..., Any]) -> ObservedMethod:
        """Возвращает дескриптор-обёртку метода.

        Args:
            func: Оборачиваемый метод.

        Returns:
            ObservedMethod.
        """
        self._func = func
        return ObservedMethod(self, func)


class ObservedMethod:
    """Дескриптор метода с записью TraceEvent.

    Attributes:
        observed: Родительский Observed.
        func: Исходный метод.
    """

    def __init__(self, observed: Observed, func: Callable[..., Any]) -> None:
        """Сохраняет декоратор и функцию.

        Args:
            observed: Observed.
            func: Метод.
        """
        self.observed = observed
        self.func = func
        self.__name__ = getattr(func, "__name__", "wrapped")
        self.__doc__ = getattr(func, "__doc__", None)

    def __get__(
        self,
        instance: object | None,
        owner: type | None = None,
    ) -> Callable[..., Any] | ObservedMethod:
        """Возвращает bound-обёртку для экземпляра.

        Args:
            instance: Экземпляр класса.
            owner: Класс.

        Returns:
            Bound callable или сам дескриптор.
        """
        if instance is None:
            return self

        def bound(*args: Any, **kwargs: Any) -> Any:
            return self._invoke(instance, *args, **kwargs)

        bound.__name__ = self.__name__
        bound.__doc__ = self.__doc__
        return bound

    def _invoke(self, instance: object, *args: Any, **kwargs: Any) -> Any:
        """Исполняет метод и пишет события.

        Args:
            instance: Экземпляр.
            *args: Позиционные аргументы.
            **kwargs: Именованные аргументы.

        Returns:
            Результат исходного метода.

        Raises:
            Exception: Пробрасывает исключение после записи failed.
        """
        trace_sink = self._resolve_trace_sink(instance)
        doc_id, question_id = self._extract_ids(args, kwargs)
        if trace_sink is not None:
            trace_sink.emit(
                TraceEvent(
                    stage=self.observed.stage,
                    status="started",
                    duration_ms=None,
                    doc_id=doc_id,
                    question_id=question_id,
                )
            )
        started_at = time.perf_counter()
        try:
            result = self.func(instance, *args, **kwargs)
        except Exception as error:
            duration_ms = (time.perf_counter() - started_at) * 1000
            if trace_sink is not None:
                trace_sink.emit(
                    TraceEvent(
                        stage=self.observed.stage,
                        status="failed",
                        duration_ms=duration_ms,
                        doc_id=doc_id,
                        question_id=question_id,
                        error=str(error),
                    )
                )
            raise
        duration_ms = (time.perf_counter() - started_at) * 1000
        if trace_sink is not None:
            trace_sink.emit(
                TraceEvent(
                    stage=self.observed.stage,
                    status="ok",
                    duration_ms=duration_ms,
                    doc_id=doc_id,
                    question_id=question_id,
                )
            )
        return result

    def _resolve_trace_sink(self, instance: object) -> TraceSink | None:
        """Берёт TraceSink из декоратора или атрибута экземпляра.

        Args:
            instance: Экземпляр класса.

        Returns:
            TraceSink или None.
        """
        if self.observed.trace_sink is not None:
            return self.observed.trace_sink
        return getattr(instance, "trace_sink", None)

    def _extract_ids(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """Читает doc_id и question_id из аргументов-моделей.

        Args:
            args: Позиционные аргументы.
            kwargs: Именованные аргументы.

        Returns:
            Пара (doc_id, question_id).
        """
        doc_id: str | None = None
        question_id: str | None = None
        values = list(args) + list(kwargs.values())
        for value in values:
            if doc_id is None:
                doc_id = self._read_field(value, "doc_id")
            if question_id is None:
                question_id = self._read_field(value, "question_id")
        return doc_id, question_id

    def _read_field(self, value: Any, field_name: str) -> str | None:
        """Достаёт строковое поле из модели или dict.

        Args:
            value: Аргумент.
            field_name: Имя поля.

        Returns:
            Значение или None.
        """
        if value is None:
            return None
        if hasattr(value, field_name):
            field_value = getattr(value, field_name)
            if isinstance(field_value, str):
                return field_value
        if isinstance(value, dict):
            field_value = value.get(field_name)
            if isinstance(field_value, str):
                return field_value
        return None
