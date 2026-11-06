"""Модели очереди заданий."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from models.requests import EvalRequest, RunRequest


class Job(BaseModel):
    """Единица очереди: analyze или eval.

    Attributes:
        job_id: Уникальный идентификатор задания.
        kind: Тип задания.
        request: RunRequest или EvalRequest по kind.
    """

    job_id: str
    kind: Literal["analyze", "eval"]
    request: RunRequest | EvalRequest

    @model_validator(mode="after")
    def validate_kind_matches_request(self) -> Job:
        """Проверяет согласованность kind и типа request.

        Returns:
            Тот же экземпляр при успехе.

        Raises:
            ValueError: kind не соответствует типу request.
        """
        if self.kind == "analyze" and not isinstance(self.request, RunRequest):
            raise ValueError("kind=analyze требует RunRequest")
        if self.kind == "eval" and not isinstance(self.request, EvalRequest):
            raise ValueError("kind=eval требует EvalRequest")
        return self

    def to_json(self) -> str:
        """Сериализует задание в JSON.

        Returns:
            JSON-строка.
        """
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> Job:
        """Восстанавливает задание из JSON.

        Args:
            payload: JSON-строка.

        Returns:
            Экземпляр Job.
        """
        return TypeAdapter(cls).validate_json(payload)


class JobReceipt(BaseModel):
    """Квитанция принятого задания.

    Attributes:
        job_id: Идентификатор задания.
        accepted_at: Время принятия.
    """

    job_id: str
    accepted_at: datetime = Field(default_factory=datetime.utcnow)
