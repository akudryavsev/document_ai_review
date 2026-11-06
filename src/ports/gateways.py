"""Абстрактные шлюзы моделей."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.gateways import (
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    RerankRequest,
    RerankResponse,
)


class ModelGateway(ABC):
    """Общий контракт загрузки и выгрузки модели.

    Attributes:
        name: Имя шлюза для GpuLease и логов.
        loaded: Загружена ли модель.
    """

    name: str = "gateway"

    def __init__(self) -> None:
        """Инициализирует флаг загрузки."""
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        """Загружает модель на устройство."""

    @abstractmethod
    def unload(self) -> None:
        """Выгружает модель и освобождает ресурсы."""


class EmbeddingGateway(ModelGateway):
    """Шлюз эмбеддера."""

    name = "embedding"

    @abstractmethod
    def infer(self, request: EmbedRequest) -> EmbedResponse:
        """Считает эмбеддинги.

        Args:
            request: Тексты и идентификаторы.

        Returns:
            Идентификаторы и факт успеха без тензоров.
        """


class RerankerGateway(ModelGateway):
    """Шлюз реранкера."""

    name = "reranker"

    @abstractmethod
    def infer(self, request: RerankRequest) -> RerankResponse:
        """Скорирует пары query × passage.

        Args:
            request: Пары текстов.

        Returns:
            Скоры и ранги.
        """


class LlmGateway(ModelGateway):
    """Шлюз LLM."""

    name = "llm"

    @abstractmethod
    def infer(self, request: GenerateRequest) -> GenerateResponse:
        """Генерирует текст ответа.

        Args:
            request: Промпт или сообщения.

        Returns:
            Текст, usage и latency.
        """
