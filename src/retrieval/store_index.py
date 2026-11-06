"""Индекс поиска из матриц ArtifactStore."""

from __future__ import annotations

from collections.abc import Sequence

from gateways.float16 import Float16Codec
from models.artifacts import ChunkVectorMatrix, QuestionVectorMatrix
from retrieval.vectors import EmbeddingIndex


class StoreEmbeddingIndex(EmbeddingIndex):
    """Карта item_id текущего документа и вариантов вопросов.

    Attributes:
        doc_id: Документ, чьи чанки сейчас в карте.
        vectors: item_id → компоненты float.
        codec: Декодер float16.
    """

    def __init__(self) -> None:
        """Создаёт пустую карту."""
        self.doc_id: str | None = None
        self.vectors: dict[str, list[float]] = {}
        self.codec = Float16Codec()

    def activate(
        self,
        doc_id: str,
        chunk_matrix: ChunkVectorMatrix | None,
        question_matrix: QuestionVectorMatrix | None,
    ) -> None:
        """Заменяет карту векторами одного документа и вопросов прогона.

        Args:
            doc_id: Документ, для которого сейчас идёт поиск.
            chunk_matrix: Матрица чанков после verify или None.
            question_matrix: Матрица вариантов вопросов или None.

        Raises:
            ValueError: Длина payload не сходится с числом рядов и dim.
        """
        self.vectors = {}
        self.doc_id = doc_id
        if chunk_matrix is not None:
            self._put_rows(chunk_matrix.chunk_ids, chunk_matrix.dim, chunk_matrix.payload)
        if question_matrix is not None:
            self._put_rows(
                question_matrix.item_ids,
                question_matrix.dim,
                question_matrix.payload,
            )

    def get_vector(self, item_id: str) -> Sequence[float] | None:
        """Возвращает вектор или None, если ключа в текущей карте нет.

        Args:
            item_id: Идентификатор чанка или варианта вопроса.

        Returns:
            Список float или None.
        """
        values = self.vectors.get(item_id)
        if values is None:
            return None
        return values

    def _put_rows(self, item_ids: list[str], dim: int, payload: bytes) -> None:
        """Кладёт ряды матрицы в карту.

        Args:
            item_ids: Идентификаторы по рядам.
            dim: Размерность.
            payload: Склейка little-endian float16.

        Raises:
            ValueError: Размер матрицы не сходится.
        """
        width = dim * 2
        if dim < 1 or len(payload) != len(item_ids) * width:
            raise ValueError("размер матрицы эмбеддингов")
        for index, item_id in enumerate(item_ids):
            start = index * width
            self.vectors[item_id] = self.codec.decode(payload[start : start + width])
