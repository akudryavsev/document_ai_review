"""Оболочки запросов и ответов шлюзов без тензоров."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbedItem(BaseModel):
    """Один текст для эмбеддинга.

    Attributes:
        item_id: Идентификатор текста.
        text: Текст.
        cache_key: Опциональный ключ кэша.
    """

    item_id: str
    text: str
    cache_key: str | None = None


class EmbedRequest(BaseModel):
    """Запрос к EmbeddingGateway.

    Attributes:
        items: Тексты для эмбеддинга.
    """

    items: list[EmbedItem] = Field(default_factory=list)


class EmbedResponseItem(BaseModel):
    """Результат эмбеддинга одного текста.

    Attributes:
        item_id: Идентификатор текста.
        success: Успех. Истина только вместе с непустыми байтами вектора.
        dim: Число компонент. При неуспехе 0.
        vector: Little-endian float16 длины dim * 2. При неуспехе пустые байты.
        cache_key: Эхо ключа из запроса. Шлюз по нему ничего не ищет.
    """

    item_id: str
    success: bool = True
    dim: int = 0
    vector: bytes = b""
    cache_key: str | None = None


class EmbedResponse(BaseModel):
    """Ответ EmbeddingGateway.

    Attributes:
        items: Результаты по идентификаторам.
    """

    items: list[EmbedResponseItem] = Field(default_factory=list)


class RerankPair(BaseModel):
    """Пара query × passage для реранка.

    Attributes:
        pair_id: Идентификатор пары.
        query: Текст запроса.
        passage: Текст пассажа.
        passage_id: Идентификатор пассажа.
    """

    pair_id: str
    query: str
    passage: str
    passage_id: str


class RerankRequest(BaseModel):
    """Запрос к RerankerGateway.

    Attributes:
        pairs: Пары для скоринга.
    """

    pairs: list[RerankPair] = Field(default_factory=list)


class RerankResponseItem(BaseModel):
    """Скор одной пары после реранка.

    Attributes:
        pair_id: Идентификатор пары.
        score: Скор реранкера.
        rank: Ранг среди пар запроса.
    """

    pair_id: str
    score: float
    rank: int


class RerankResponse(BaseModel):
    """Ответ RerankerGateway.

    Attributes:
        items: Скоры и ранги.
    """

    items: list[RerankResponseItem] = Field(default_factory=list)


class ChatMessage(BaseModel):
    """Сообщение для LLM.

    Attributes:
        role: Роль отправителя.
        content: Текст сообщения.
    """

    role: str
    content: str


class GenerateRequest(BaseModel):
    """Запрос к LlmGateway.

    Attributes:
        messages: Сообщения промпта.
        prompt: Альтернативный плоский промпт.
        max_output_tokens: Лимит ответа. None — лимит в запрос не ставится.
        temperature: Температура.
        doc_id: Документ, если известен.
        question_id: Вопрос, если известен.
    """

    messages: list[ChatMessage] = Field(default_factory=list)
    prompt: str | None = None
    max_output_tokens: int | None = None
    temperature: float = 0.0
    doc_id: str | None = None
    question_id: str | None = None


class GenerateResponse(BaseModel):
    """Ответ LlmGateway.

    Attributes:
        text: Текст ответа.
        prompt_tokens: Токены промпта.
        completion_tokens: Токены ответа.
        latency_ms: Задержка.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
