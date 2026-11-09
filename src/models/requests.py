"""Запросы CLI и фасада."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from models.config import AppConfig
from models.model_list import ModelListError


PhaseName = Literal["A1", "A2", "B"]


class RunOverrides(BaseModel):
    """Переопределения модели и каталогов из CLI.

    Attributes:
        answer_model: Модель ответа.
        reranker_model: Модель реранкера.
        embedding_model: Модель эмбеддера.
        cache_dir: Каталог кэша эмбеддингов.
        contexts_dir: Каталог context-бандлов.
        results_dir: Каталог результатов.
        index_dir: Каталог index.
    """

    answer_model: str | None = None
    reranker_model: str | None = None
    embedding_model: str | None = None
    cache_dir: str | None = None
    contexts_dir: str | None = None
    results_dir: str | None = None
    index_dir: str | None = None


class RunRequest(BaseModel):
    """Запрос пакетного прогона анализа.

    Attributes:
        config: Загруженный AppConfig.
        config_path: Исходный путь к YAML, если был.
        phases: Список фаз к исполнению.
        force_phases: Фазы принудительного пересчёта.
        overrides: Переопределения из CLI.
    """

    config: AppConfig
    config_path: str | None = None
    phases: list[PhaseName] = Field(default_factory=lambda: ["A1", "A2", "B"])
    force_phases: set[PhaseName] = Field(default_factory=set)
    overrides: RunOverrides = Field(default_factory=RunOverrides)

    def resolved_config(self) -> AppConfig:
        """Применяет overrides к копии конфига.

        Флаг модели оставляет имя, которое уже есть в списке. Записи
        params при этом не выкидываются: ini строится по полному конфигу.

        Returns:
            AppConfig с учётом переопределений.

        Raises:
            ModelListError: Имени из флага нет в списке конфига.
        """
        config = self.config.model_copy(deep=True)
        overrides = self.overrides
        self._keep_named(config.answer, overrides.answer_model)
        self._keep_named(config.reranker, overrides.reranker_model)
        self._keep_named(config.embedding, overrides.embedding_model)
        if overrides.cache_dir is not None:
            config.runtime.cache_dir = overrides.cache_dir
        if overrides.contexts_dir is not None:
            config.phases.contexts_dir = overrides.contexts_dir
        if overrides.results_dir is not None:
            config.runtime.results_dir = overrides.results_dir
        if overrides.index_dir is not None:
            config.phases.index_dir = overrides.index_dir
        return config

    def _keep_named(self, role: object, selected: str | None) -> None:
        """Оставляет в роли одну запись из уже заданного списка.

        Args:
            role: Секция embedding, reranker или answer.
            selected: Имя файла из флага CLI или None.

        Raises:
            ModelListError: Имени нет в списке конфига.
        """
        if selected is None:
            return
        names = list(getattr(role, "models") or [])
        if selected not in names:
            raise ModelListError(f"имя {selected} нет в списке конфига")
        role.model = selected
        role.models = [selected]


class EvalRequest(BaseModel):
    """Запрос оценки на ground truth.

    Attributes:
        config: Загруженный AppConfig.
        config_path: Исходный путь к YAML, если был.
        ground_truth_path: Путь к gt.json.
        predictions_dir: Каталог предсказаний.
    """

    config: AppConfig
    config_path: str | None = None
    ground_truth_path: str
    predictions_dir: str
