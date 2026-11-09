"""Артефакты прогона: context-бандлы и ответы."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ContextCandidate(BaseModel):
    """Кандидат внутри ContextBundle после реранка и expansion.

    Attributes:
        rank_after_rerank: Ранг после реранка.
        rerank_score: Скор реранкера.
        section_path: Иерархия заголовков или None.
        pages: Страницы.
        text: Текст секции или усечённый фрагмент.
        chunks_included: Идентификаторы вошедших чанков.
        truncated: Была ли обрезка по бюджету.
    """

    rank_after_rerank: int
    rerank_score: float
    section_path: list[str] | None = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    text: str = ""
    chunks_included: list[str] = Field(default_factory=list)
    truncated: bool = False


class ContextBundle(BaseModel):
    """Контекст пары документ × вопрос для фазы B.

    Attributes:
        doc_id: Идентификатор документа.
        question_id: Идентификатор вопроса.
        category: Канонический вопрос для LLM.
        used_synonyms: Синонимы, участвовавшие в поиске.
        empty_search: True, если после retrieval/rerank нет кандидатов.
        reason: Код причины пустого поиска или None.
        budget_used_tokens: Занятый бюджет токенов.
        candidates: Кандидаты после реранка/expansion.
        retrieval_meta: Метаданные поиска.
        chunking_hash: Хеш чанкинга upstream.
        retrieval_config_hash: Хеш retrieval.
        reranker_config_hash: Хеш реранкера.
    """

    doc_id: str
    question_id: str
    category: str
    used_synonyms: list[str] = Field(default_factory=list)
    empty_search: bool = False
    reason: str | None = None
    budget_used_tokens: int = 0
    candidates: list[ContextCandidate] = Field(default_factory=list)
    retrieval_meta: dict[str, Any] = Field(default_factory=dict)
    chunking_hash: str = ""
    retrieval_config_hash: str = ""
    reranker_config_hash: str = ""

    @model_validator(mode="after")
    def validate_empty_search(self) -> ContextBundle:
        """Связывает empty_search с пустыми кандидатами и reason.

        Returns:
            Тот же экземпляр при успехе.

        Raises:
            ValueError: Нарушен инвариант empty_search.
        """
        if self.empty_search:
            if self.candidates:
                raise ValueError(
                    "empty_search=true требует candidates=[]"
                )
            if self.reason != "no_relevant_chunks":
                raise ValueError(
                    "empty_search=true требует reason=no_relevant_chunks"
                )
        return self


class Citation(BaseModel):
    """Цитата в ответе.

    Attributes:
        chunk_ids: Все чанки фрагмента, на который ссылается модель.
        page_from: Начальная страница фрагмента.
        page_to: Конечная страница фрагмента.
        section_path: Иерархия заголовков фрагмента или None.
        quote: Текст цитаты из ответа модели.
    """

    chunk_ids: list[str] = Field(default_factory=list)
    page_from: int = 0
    page_to: int = 0
    section_path: list[str] | None = None
    quote: str = ""


class TokenUsage(BaseModel):
    """Учёт токенов LLM.

    Attributes:
        prompt: Токены промпта.
        completion: Токены ответа.
    """

    prompt: int = 0
    completion: int = 0


class RetrievalStats(BaseModel):
    """Статистика retrieval в AnswerRecord.

    Attributes:
        candidates: Число кандидатов до реранка.
        after_rerank: Число после реранка.
        after_verify: Число после верификации.
        sections_used: Использованные секции.
    """

    candidates: int = 0
    after_rerank: int = 0
    after_verify: int = 0
    sections_used: list[str] = Field(default_factory=list)


class AnswerRecord(BaseModel):
    """Строка results/<doc_id>.jsonl.

    Attributes:
        doc_id: Идентификатор документа.
        question_id: Идентификатор вопроса.
        category: Канонический вопрос.
        used_synonyms: Синонимы поиска.
        found: Нашёл ли LLM ответ.
        answer: Текст ответа.
        explanation: Пояснение отдельно от ответа.
        citations: Цитаты.
        confidence: Уверенность. До формулы не заполняется.
        reason: Причина found=false или None.
        retrieval: Статистика поиска.
        tokens: Учёт токенов.
        model: Имя файла модели ответа. Совпадает с answer_model.
        embedding_model: Имя файла эмбеддера комбинации.
        reranker_model: Имя файла реранкера комбинации.
        answer_model: Имя файла модели ответа комбинации.
        latency_ms: Задержка ответа.
        status: Статус пары.
        verification_skipped: Верификация была пропущена.
        prepared_context: Промпт с фрагментами, если модель не ответила.
    """

    doc_id: str
    question_id: str
    category: str
    used_synonyms: list[str] = Field(default_factory=list)
    found: bool = False
    answer: str = ""
    explanation: str = ""
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None
    reason: str | None = None
    retrieval: RetrievalStats = Field(default_factory=RetrievalStats)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    embedding_model: str = ""
    reranker_model: str = ""
    answer_model: str = ""
    latency_ms: float = 0.0
    status: Literal["ok", "failed"] = "ok"
    verification_skipped: bool = False
    prepared_context: str | None = None

    @model_validator(mode="after")
    def validate_found(self) -> AnswerRecord:
        """Запрещает найденный ответ без цитат и вместе с пустым поиском.

        Returns:
            Тот же экземпляр при успехе.

        Raises:
            ValueError: found=true без цитат или вместе с no_relevant_chunks.
        """
        if self.found and self.reason == "no_relevant_chunks":
            raise ValueError(
                "пустой поиск и found=true вместе не сохраняются"
            )
        if self.found and not self.citations:
            raise ValueError("found=true требует непустой список citations")
        return self


class ArtifactMeta(BaseModel):
    """Метаданные артефакта для ArtifactGate.

    Attributes:
        schema_version: Версия схемы.
        chunking_hash: Хеш чанкинга, если применим.
        retrieval_config_hash: Хеш retrieval, если применим.
        reranker_config_hash: Хеш реранкера, если применим.
        embedding_model: Модель эмбеддера.
        embedding_dim: Размерность вектора.
        embedding_dtype: Тип вектора; в v1 float16.
        n_chunks: Число чанков.
        source_sha256: Отпечаток исходного файла.
        doc_id: Идентификатор документа.
        vectors_index_sha256: SHA-256 файла индекса векторов.
        extra: Прочие поля meta.
    """

    schema_version: str = "1.0"
    chunking_hash: str | None = None
    retrieval_config_hash: str | None = None
    reranker_config_hash: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    embedding_dtype: str | None = None
    n_chunks: int | None = None
    source_sha256: str | None = None
    doc_id: str | None = None
    vectors_index_sha256: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class QuestionVariantRecord(BaseModel):
    """Один вариант вопроса в кэше эмбеддингов.

    Attributes:
        question_id: Идентификатор вопроса.
        variant: Метка варианта: category или synonym:N.
        text: Текст, ушедший в эмбеддер.
        text_sha256: SHA-256 этого текста.
        item_id: Ключ эмбеддинга.
    """

    question_id: str
    variant: str
    text: str
    text_sha256: str = ""
    item_id: str


class ChunkVectorMatrix(BaseModel):
    """Оболочка матрицы эмбеддингов чанков без тензора фреймворка.

    Attributes:
        chunk_ids: Идентификаторы чанков по рядам.
        text_sha256s: Хеши текстов, на которых посчитаны векторы.
        dim: Размерность.
        payload: Сырые little-endian float16.
    """

    chunk_ids: list[str]
    text_sha256s: list[str]
    dim: int
    payload: bytes


class QuestionVectorMatrix(BaseModel):
    """Оболочка матрицы эмбеддингов вариантов вопросов.

    Attributes:
        item_ids: Идентификаторы вариантов по рядам.
        text_sha256s: Хеши текстов.
        dim: Размерность.
        payload: Сырые little-endian float16.
    """

    item_ids: list[str]
    text_sha256s: list[str]
    dim: int
    payload: bytes


class EmbeddingBindingReport(BaseModel):
    """Отчёт сверки текста чанка или вопроса с рядом вектора.

    Attributes:
        ok: Все проверки прошли.
        n_chunks: Число текстовых записей.
        n_vector_rows: Число рядов индекса.
        mismatched_ids: Идентификаторы, которые есть только с одной стороны.
        mismatched_hashes: Идентификаторы с разным text_sha256.
        size_error: Описание неверного размера матрицы, если было.
    """

    ok: bool
    n_chunks: int
    n_vector_rows: int
    mismatched_ids: list[str] = Field(default_factory=list)
    mismatched_hashes: list[str] = Field(default_factory=list)
    size_error: str | None = None


class CachedEmbeddingRecord(BaseModel):
    """Паспорт одного вектора в embed_cache.

    Attributes:
        text_sha256: Хеш текста.
        model: Имя модели.
        dtype: Тип данных.
        dim: Размерность.
        created_at: Время записи в ISO.
    """

    text_sha256: str
    model: str
    dtype: str
    dim: int
    created_at: str


class GroundTruthItem(BaseModel):
    """Эталон для eval.

    Attributes:
        doc_id: Идентификатор документа.
        question_id: Идентификатор вопроса.
        expected_found: Ожидаемое found.
        expected_answer: Эталонный ответ.
        expected_pages: Эталонные страницы.
        expected_sections: Эталонные секции.
    """

    doc_id: str
    question_id: str
    expected_found: bool
    expected_answer: str = ""
    expected_pages: list[int] = Field(default_factory=list)
    expected_sections: list[str] = Field(default_factory=list)
