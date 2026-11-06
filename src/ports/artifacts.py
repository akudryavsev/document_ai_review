"""Порт персистентности артефактов."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.artifacts import (
    ArtifactMeta,
    ChunkVectorMatrix,
    ContextBundle,
    EmbeddingBindingReport,
    QuestionVectorMatrix,
)
from models.documents import CandidateBatch, ChunkBatch
from models.questions import QuestionSet


class ArtifactStore(ABC):
    """Хранилище JSONL/float16-артефактов через pydantic-модели."""

    @abstractmethod
    def save_chunks(self, batch: ChunkBatch, meta: ArtifactMeta) -> None:
        """Сохраняет чанки документа.

        Args:
            batch: Пакет чанков.
            meta: Метаданные A1.
        """

    @abstractmethod
    def load_chunks(self, doc_id: str) -> ChunkBatch | None:
        """Читает чанки документа.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            ChunkBatch или None, если нет.
        """

    @abstractmethod
    def load_chunks_meta(self, doc_id: str) -> ArtifactMeta | None:
        """Читает meta чанков.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            ArtifactMeta или None.
        """

    @abstractmethod
    def save_chunk_vectors(
        self,
        doc_id: str,
        rows: list[tuple[str, str]],
        dim: int,
        payload: bytes,
    ) -> None:
        """Пишет vectors.f16 и индекс, затем проверяет связку.

        Args:
            doc_id: Идентификатор документа.
            rows: Пары (chunk_id, text_sha256) в порядке рядов.
            dim: Размерность.
            payload: Сырые little-endian float16.

        Raises:
            ValueError: Пары или размер не сходятся с чанками.
            EmbeddingBindingError: Связка не прошла проверку.
        """

    @abstractmethod
    def verify_chunk_embeddings(self, doc_id: str) -> EmbeddingBindingReport:
        """Сверяет чанки, индекс и размер матрицы.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            Отчёт с ok=true.

        Raises:
            EmbeddingBindingError: Есть рассинхрон.
        """

    @abstractmethod
    def load_chunk_vectors(self, doc_id: str) -> ChunkVectorMatrix | None:
        """Читает матрицу чанков после verify.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            Оболочка с ids, хешами, dim и payload или None.
        """

    @abstractmethod
    def embedding_cache_key(self, model: str, dtype: str, text_sha256: str) -> str:
        """Строит ключ embed_cache.

        Args:
            model: Имя модели.
            dtype: Тип данных.
            text_sha256: Хеш текста.

        Returns:
            Hex-ключ.
        """

    @abstractmethod
    def get_cached_embedding(self, key: str) -> bytes | None:
        """Читает вектор из embed_cache.

        Args:
            key: Ключ кэша.

        Returns:
            Байты вектора или None.
        """

    @abstractmethod
    def put_cached_embedding(
        self,
        key: str,
        payload: bytes,
        text_sha256: str,
        model: str,
        dtype: str,
        dim: int,
    ) -> None:
        """Кладёт вектор в embed_cache.

        Args:
            key: Ключ кэша.
            payload: Байты одного вектора.
            text_sha256: Хеш текста.
            model: Имя модели.
            dtype: Тип данных.
            dim: Размерность.
        """

    @abstractmethod
    def save_question_vectors(
        self,
        rows: list[tuple[str, str]],
        dim: int,
        payload: bytes,
    ) -> None:
        """Пишет матрицу эмбеддингов вопросов и проверяет связку.

        Args:
            rows: Пары (item_id, text_sha256).
            dim: Размерность.
            payload: Little-endian float16.
        """

    @abstractmethod
    def verify_question_embeddings(self) -> EmbeddingBindingReport:
        """Сверяет кэш вопросов с матрицей.

        Returns:
            Отчёт с ok=true.

        Raises:
            EmbeddingBindingError: Есть рассинхрон.
        """

    @abstractmethod
    def load_question_vectors(self) -> QuestionVectorMatrix | None:
        """Читает матрицу вопросов после verify.

        Returns:
            Оболочка или None.
        """

    @abstractmethod
    def save_candidates(self, batch: CandidateBatch, meta: ArtifactMeta) -> None:
        """Сохраняет candidates документа.

        Args:
            batch: Кандидаты.
            meta: Метаданные A2.
        """

    @abstractmethod
    def load_candidates(self, doc_id: str) -> CandidateBatch | None:
        """Читает candidates документа.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            CandidateBatch или None.
        """

    @abstractmethod
    def load_candidates_meta(self, doc_id: str) -> ArtifactMeta | None:
        """Читает meta candidates.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            ArtifactMeta или None.
        """

    @abstractmethod
    def save_context(self, bundle: ContextBundle) -> None:
        """Сохраняет context-бандл.

        Args:
            bundle: ContextBundle.
        """

    @abstractmethod
    def load_context(self, doc_id: str, question_id: str) -> ContextBundle | None:
        """Читает context-бандл.

        Args:
            doc_id: Идентификатор документа.
            question_id: Идентификатор вопроса.

        Returns:
            ContextBundle или None.
        """

    @abstractmethod
    def save_questions_cache(
        self,
        questions: QuestionSet,
        meta: ArtifactMeta,
    ) -> None:
        """Кэширует эмбеддинги вариантов вопросов.

        Args:
            questions: Набор вопросов.
            meta: Метаданные кэша.
        """

    @abstractmethod
    def load_questions_meta(self) -> ArtifactMeta | None:
        """Читает meta кэша вопросов.

        Returns:
            ArtifactMeta или None.
        """

    @abstractmethod
    def has_questions_cache(self) -> bool:
        """Проверяет наличие кэша вопросов.

        Returns:
            True, если кэш есть.
        """

    @abstractmethod
    def save_answers_meta(self, doc_id: str, meta: ArtifactMeta) -> None:
        """Сохраняет meta успешных ответов документа.

        Args:
            doc_id: Идентификатор документа.
            meta: Upstream-хеши фазы B.
        """

    @abstractmethod
    def load_answers_meta(self, doc_id: str) -> ArtifactMeta | None:
        """Читает meta ответов документа.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            ArtifactMeta или None.
        """
