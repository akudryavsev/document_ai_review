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
    """Имена файлов, которые обошёл прогон.

    Attributes:
        embedding: Эмбеддеры в порядке конфига.
        reranker: Реранкеры в порядке конфига.
        answer: Модели ответа в порядке конфига.
        verification: Модель верификации. В цикл не входит.
    """

    embedding: list[str] = Field(default_factory=list)
    reranker: list[str] = Field(default_factory=list)
    answer: list[str] = Field(default_factory=list)
    verification: str = ""


class CombinationSummary(BaseModel):
    """Счётчики одной тройки моделей.

    Attributes:
        embedding_model: Имя файла эмбеддера.
        reranker_model: Имя файла реранкера.
        answer_model: Имя файла модели ответа.
        phases: Счётчики фаз, запрошенных для этой тройки.
    """

    embedding_model: str
    reranker_model: str
    answer_model: str
    phases: dict[PhaseName, PhaseCounters] = Field(default_factory=dict)


class RunSummary(BaseModel):
    """Агрегат пакетного прогона.

    Attributes:
        documents: Число документов.
        questions: Число вопросов.
        error_rate: Доля неуспешных пар или документов.
        average_latency_ms: Средняя latency ответа.
        average_prompt_tokens: Средние токены промпта.
        average_completion_tokens: Средние токены ответа.
        model_versions: Списки имён файлов прогона.
        combinations: Тройки в порядке цикла.
        phases: Суммарные счётчики по каждой запрошенной фазе.
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
    combinations: list[CombinationSummary] = Field(default_factory=list)
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
    """Отчёт одной тройки моделей.

    Attributes:
        embedding_model: Эмбеддер группы.
        reranker_model: Реранкер группы.
        answer_model: Модель ответа группы.
        cosine_embedding_model: Первый эмбеддер конфига для косинуса.
        ground_truth_path: Путь к gt.json.
        predictions_dir: Каталог предсказаний.
        index_dir: Корень кандидатов.
        contexts_dir: Корень контекстов.
        n_matched: Число сопоставленных пар эталона и группы.
        n_missing_candidates: Пары без файла кандидатов по префиксу тройки.
        candidates_path: Путь кандидатов первой сопоставленной пары.
        metrics: Зарезервировано под формулы спеки eval.
        note: Пометка о формулах, которые считает отдельная спека eval.
    """

    embedding_model: str = ""
    reranker_model: str = ""
    answer_model: str = ""
    cosine_embedding_model: str = ""
    ground_truth_path: str = ""
    predictions_dir: str = ""
    index_dir: str = ""
    contexts_dir: str = ""
    n_matched: int = 0
    n_missing_candidates: int = 0
    candidates_path: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    note: str = "Формулы метрик задаёт спецификация eval"
