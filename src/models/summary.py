"""Сводка прогона и оболочка eval-отчёта."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from models.requests import PhaseName


class PhaseCounters(BaseModel):
    """Счётчики одной фазы в RunSummary.

    Attributes:
        computed: Пересчитано.
        skipped: Пропущено.
        failed: Упало.
    """

    computed: int = 0
    skipped: int = 0
    failed: int = 0


class ModelVersions(BaseModel):
    """Версии моделей прогона.

    Attributes:
        embedding: Эмбеддер.
        reranker: Реранкер.
        answer: LLM ответа.
        verification: Модель верификации.
    """

    embedding: str = ""
    reranker: str = ""
    answer: str = ""
    verification: str = ""


class RunSummary(BaseModel):
    """Агрегат пакетного прогона.

    Attributes:
        documents: Число документов.
        questions: Число вопросов.
        error_rate: Доля неуспешных пар или документов.
        average_latency_ms: Средняя latency ответа.
        average_prompt_tokens: Средние токены промпта.
        average_completion_tokens: Средние токены ответа.
        model_versions: Версии моделей.
        phases: Счётчики по каждой запрошенной фазе.
        requested_phases: Фазы, которые были запрошены.
        details: Дополнительные детали failures.
    """

    documents: int = 0
    questions: int = 0
    error_rate: float = 0.0
    average_latency_ms: float = 0.0
    average_prompt_tokens: float = 0.0
    average_completion_tokens: float = 0.0
    model_versions: ModelVersions = Field(default_factory=ModelVersions)
    phases: dict[PhaseName, PhaseCounters] = Field(default_factory=dict)
    requested_phases: list[PhaseName] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_requested_phase_counters(self) -> RunSummary:
        """Требует счётчики для каждой запрошенной фазы.

        Returns:
            Тот же экземпляр при успехе.

        Raises:
            ValueError: Нет счётчиков для запрошенной фазы.
        """
        for phase_name in self.requested_phases:
            if phase_name not in self.phases:
                raise ValueError(
                    f"RunSummary без счётчиков фазы {phase_name}"
                )
        return self


class EvalReport(BaseModel):
    """Оболочка отчёта eval до отдельной спецификации метрик.

    Attributes:
        ground_truth_path: Путь к gt.json.
        predictions_dir: Каталог предсказаний.
        metrics: Зарезервировано под будущие метрики.
        note: Пометка об отложенной спеке eval.
    """

    ground_truth_path: str = ""
    predictions_dir: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    note: str = "Состав метрик уточняется в спецификации eval"
