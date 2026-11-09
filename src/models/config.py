"""Конфигурация приложения и вычисляемые хеши артефактов."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from models.model_list import ModelListResolver


class InputConfig(BaseModel):
    """Пути к входным документам и вопросам.

    Attributes:
        documents_dir: Каталог документов (рекурсивный обход).
        questions_file: Путь к JSON-файлу вопросов.
    """

    documents_dir: str = "./docs"
    questions_file: str = "./questions.json"


class ExtractionConfig(BaseModel):
    """Параметры извлечения текста из документов.

    Attributes:
        pdf_text_layer_required: Без текстового слоя PDF документ помечается failed.
        debug_dump: Писать отладочные Markdown после extract и clean.
    """

    pdf_text_layer_required: bool = True
    debug_dump: bool = False


class ChunkingConfig(BaseModel):
    """Параметры чанкинга.

    Attributes:
        max_tokens: Максимальный размер чанка в токенах.
        overlap_tokens: Перекрытие соседних чанков.
        min_chunk_tokens: Минимальный размер чанка.
        preserve_section_path: Сохранять иерархию заголовков.
    """

    max_tokens: int = 512
    overlap_tokens: int = 64
    min_chunk_tokens: int = 80
    preserve_section_path: bool = True


class EmbeddingConfig(BaseModel):
    """Параметры эмбеддера.

    Attributes:
        model: Имя файла текущей комбинации. Первый элемент models.
        models: Имена GGUF в каталоге models/. Одно имя — список из одного.
        batch_size: Размер батча.
        dtype: Тип данных на устройстве.
        device: Устройство загрузки.
        cache_dir: Каталог кэша эмбеддингов чанков.
        questions_cache: Путь к кэшу эмбеддингов вопросов.
        persist_chunk_index: Сохранять эмбеддинги чанков на диск.
        persist_bm25: Сохранять BM25-индекс на диск.
    """

    model: str | None = Field(default=None, exclude=True)
    models: list[str] | None = None
    batch_size: int = 8
    dtype: str = "fp16"
    device: str = "cuda:0"
    cache_dir: str = "./.cache/emb"
    questions_cache: str = "./.cache/questions.parquet"
    persist_chunk_index: bool = True
    persist_bm25: bool = False

    @model_validator(mode="before")
    @classmethod
    def reject_both_model_keys(cls, value: object) -> object:
        """Запрещает в YAML сразу и model, и models.

        Args:
            value: Сырой словарь секции или уже собранный конфиг.

        Returns:
            Тот же вход.

        Raises:
            ValueError: В словаре заданы оба ключа.
        """
        if isinstance(value, dict) and value.get("model") is not None and value.get("models") is not None:
            raise ValueError("одновременно заданы model и models")
        return value

    @model_validator(mode="after")
    def resolve_models(self) -> EmbeddingConfig:
        """Сводит model и models к списку имён файлов.

        Returns:
            Конфиг с заполненными model и models.

        Raises:
            ValueError: Список пуст.
        """
        if self.models and self.model == self.models[0]:
            return self
        current, names = ModelListResolver("embed").resolve(self.model, self.models)
        self.model = current
        self.models = names
        return self


class HybridConfig(BaseModel):
    """Веса гибридного поиска.

    Attributes:
        enabled: Включён ли гибридный поиск.
        bm25_weight: Вес BM25.
        vector_weight: Вес векторного поиска.
    """

    enabled: bool = True
    bm25_weight: float = 0.3
    vector_weight: float = 0.7


class RetrievalConfig(BaseModel):
    """Параметры retrieval.

    Attributes:
        top_k: Число кандидатов ветви до объединения.
        score_threshold: Порог косинуса; 0 выключает отсечение.
        query_strategy: Стратегия объединения вариантов запроса.
        hybrid: Настройки гибридного поиска.
        persist_candidates: Сохранять candidates.jsonl до реранка.
        debug_dump: Писать промежуточный JSON в cache/debug.
    """

    top_k: int = 20
    score_threshold: float = 0.0
    query_strategy: str = "max_over_variants"
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    persist_candidates: bool = True
    debug_dump: bool = False


class RerankerConfig(BaseModel):
    """Параметры реранкера.

    Attributes:
        enabled: Включён ли реранкер.
        model: Имя файла текущей комбинации. Первый элемент models.
        models: Имена GGUF реранкеров.
        top_n: Число кандидатов после реранка.
        score_threshold: Порог реранка; 0 выключает отсечение.
        device: Устройство загрузки.
    """

    enabled: bool = True
    model: str | None = Field(default=None, exclude=True)
    models: list[str] | None = None
    top_n: int = 5
    score_threshold: float = 0.0
    device: str = "cuda:0"

    @model_validator(mode="before")
    @classmethod
    def reject_both_model_keys(cls, value: object) -> object:
        """Запрещает в YAML сразу и model, и models.

        Args:
            value: Сырой словарь секции или уже собранный конфиг.

        Returns:
            Тот же вход.

        Raises:
            ValueError: В словаре заданы оба ключа.
        """
        if isinstance(value, dict) and value.get("model") is not None and value.get("models") is not None:
            raise ValueError("одновременно заданы model и models")
        return value

    @model_validator(mode="after")
    def resolve_models(self) -> RerankerConfig:
        """Сводит model и models к списку имён файлов.

        Returns:
            Конфиг с заполненными model и models.

        Raises:
            ValueError: Список пуст.
        """
        if self.models and self.model == self.models[0]:
            return self
        current, names = ModelListResolver("rerank").resolve(self.model, self.models)
        self.model = current
        self.models = names
        return self


class VerificationConfig(BaseModel):
    """Параметры LLM-верификации.

    Attributes:
        enabled: Включена ли верификация.
        model: Идентификатор модели верификации.
        batch_size: Размер батча.
        device: Устройство загрузки.
    """

    enabled: bool = True
    model: str = "small-llm"
    batch_size: int = 8
    device: str = "auto"


class ContextConfig(BaseModel):
    """Параметры сборки контекста.

    Attributes:
        expansion: Стратегия расширения кандидатов.
        neighbors_fallback: Число соседей при отсутствии section_path.
        max_total_tokens: Бюджет финального промпта.
        max_section_tokens: Обрезка длинных секций.
        early_exit_threshold: Порог раннего выхода; 0 выключает.
        compression: Сжатие бандлов на диске.
    """

    expansion: Literal["section", "neighbors", "none"] = "section"
    neighbors_fallback: int = 2
    max_total_tokens: int = 6000
    max_section_tokens: int = 3000
    early_exit_threshold: float = 0.0
    compression: Literal["none", "gzip"] = "none"


class AnswerConfig(BaseModel):
    """Параметры генерации ответа.

    Attributes:
        model: Имя файла текущей комбинации. Первый элемент models.
        models: Имена GGUF моделей ответа.
        device: Устройство загрузки.
        temperature: Температура генерации.
        max_output_tokens: Лимит токенов ответа.
        require_citations: Требовать цитаты в ответе.
    """

    model: str | None = Field(default=None, exclude=True)
    models: list[str] | None = None
    device: str = "auto"
    temperature: float = 0
    max_output_tokens: int = 800
    require_citations: bool = True

    @model_validator(mode="before")
    @classmethod
    def reject_both_model_keys(cls, value: object) -> object:
        """Запрещает в YAML сразу и model, и models.

        Args:
            value: Сырой словарь секции или уже собранный конфиг.

        Returns:
            Тот же вход.

        Raises:
            ValueError: В словаре заданы оба ключа.
        """
        if isinstance(value, dict) and value.get("model") is not None and value.get("models") is not None:
            raise ValueError("одновременно заданы model и models")
        return value

    @model_validator(mode="after")
    def resolve_models(self) -> AnswerConfig:
        """Сводит model и models к списку имён файлов.

        Returns:
            Конфиг с заполненными model и models.

        Raises:
            ValueError: Список пуст.
        """
        if self.models and self.model == self.models[0]:
            return self
        current, names = ModelListResolver("answer").resolve(self.model, self.models)
        self.model = current
        self.models = names
        return self


class PhasesConfig(BaseModel):
    """Какие фазы исполнять и где лежат артефакты.

    Attributes:
        run_a1: Исполнять индексацию.
        run_a2: Исполнять retrieval.
        run_b: Исполнять ответы.
        contexts_dir: Каталог context-бандлов.
        index_dir: Каталог candidates/index.
        skip_if_artifacts_valid: Пропускать фазу при валидных хешах.
    """

    run_a1: bool = True
    run_a2: bool = True
    run_b: bool = True
    contexts_dir: str = "./contexts"
    index_dir: str = "./index"
    skip_if_artifacts_valid: bool = True


class InferenceConfig(BaseModel):
    """Адрес контейнера llama-server.

    Attributes:
        base_url: Базовый URL без завершающего слэша.
        request_timeout_sec: Таймаут одного HTTP-вызова и ожидания load.
        poll_interval_sec: Пауза между опросами готовности пресета.
    """

    base_url: str = "http://inference:8080"
    request_timeout_sec: float = 120
    poll_interval_sec: float = 0.5


class RuntimeConfig(BaseModel):
    """Рантайм-параметры прогона.

    Attributes:
        retries: Число вызовов LLM, включая первый. Меньше 1 — ошибка.
        retry_pause_sec: Пауза между попытками, секунды.
        question_parallelism: Параллелизм вопросов (в PoC = 1).
        results_dir: Каталог результатов.
        log_level: Уровень логов.
        cache_dir: Корень отладочных дампов и артефактов (.cache).
    """

    retries: int = 2
    retry_pause_sec: float = 0.0
    question_parallelism: int = 1
    results_dir: str = "./results"
    log_level: str = "INFO"
    cache_dir: str = "./.cache"

    @model_validator(mode="after")
    def validate_retries(self) -> RuntimeConfig:
        """Проверяет, что число попыток задано до старта вопросов.

        Returns:
            Тот же конфиг.

        Raises:
            ValueError: retries меньше 1.
        """
        if self.retries < 1:
            raise ValueError("runtime.retries должен быть не меньше 1")
        return self


class AppConfig(BaseModel):
    """Зеркало YAML-конфигурации пайплайна.

    Attributes:
        input: Входные пути.
        extraction: Извлечение текста.
        chunking: Чанкинг.
        embedding: Эмбеддер.
        retrieval: Retrieval.
        reranker: Реранкер.
        verification: Верификация.
        context: Сборка контекста. Бюджет не меньше
            reranker.top_n * chunking.max_tokens.
        answer: Генерация ответа.
        inference: Адрес llama-server. Секцию можно не писать в YAML.
        phases: Флаги фаз и каталоги артефактов.
        runtime: Рантайм.
        schema_version: Версия схемы meta-артефактов.
    """

    input: InputConfig = Field(default_factory=InputConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    phases: PhasesConfig = Field(default_factory=PhasesConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    schema_version: str = "1.0"

    @model_validator(mode="after")
    def validate_context_budget(self) -> AppConfig:
        """Проверяет, что бюджет вмещает top_n чанков номинального размера.

        Returns:
            Тот же конфиг.

        Raises:
            ValueError: max_total_tokens меньше top_n * max_tokens.
        """
        floor = self.reranker.top_n * self.chunking.max_tokens
        if self.context.max_total_tokens < floor:
            raise ValueError(
                "max_total_tokens меньше reranker.top_n * chunking.max_tokens"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Читает YAML и возвращает проверенный конфиг.

        Args:
            path: Путь к YAML-файлу.

        Returns:
            Экземпляр AppConfig.
        """
        raw_text = Path(path).read_text(encoding="utf-8")
        payload: dict[str, Any] = yaml.safe_load(raw_text) or {}
        return cls.model_validate(payload)

    @property
    def chunking_hash(self) -> str:
        """Хеш параметров чанкинга для инвалидации A1.

        Returns:
            sha256 от max_tokens, overlap, min_chunk и preserve_section_path.
        """
        chunking = self.chunking
        material = (
            f"{chunking.max_tokens}+{chunking.overlap_tokens}+"
            f"{chunking.min_chunk_tokens}+{chunking.preserve_section_path}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @property
    def retrieval_config_hash(self) -> str:
        """Хеш параметров retrieval для инвалидации A2.

        Returns:
            sha256 от top_k, score_threshold, hybrid и query_strategy.
        """
        retrieval = self.retrieval
        hybrid = retrieval.hybrid
        material = (
            f"{retrieval.top_k}+{retrieval.score_threshold}+"
            f"{hybrid.enabled}+{hybrid.bm25_weight}+{hybrid.vector_weight}+"
            f"{retrieval.query_strategy}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @property
    def reranker_config_hash(self) -> str:
        """Хеш параметров реранкера для инвалидации A2.

        Returns:
            sha256 от model, top_n и score_threshold.
        """
        reranker = self.reranker
        material = (
            f"{reranker.model}+{reranker.top_n}+{reranker.score_threshold}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()
