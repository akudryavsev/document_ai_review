"""События наблюдаемости."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """Событие стадии для TraceSink.

    Attributes:
        stage: Имя стадии из Observed.
        status: started, ok или failed.
        duration_ms: Длительность; для started — None.
        doc_id: Идентификатор документа, если есть.
        question_id: Идентификатор вопроса, если есть.
        counters: Числовые счётчики стадии.
        error: Текст исключения при failed.
        debug_payload: Тело промпта/документа только в DEBUG.
    """

    stage: str
    status: Literal["started", "ok", "failed"]
    duration_ms: float | None = None
    doc_id: str | None = None
    question_id: str | None = None
    counters: dict[str, int | float] = Field(default_factory=dict)
    error: str | None = None
    debug_payload: str | None = None
