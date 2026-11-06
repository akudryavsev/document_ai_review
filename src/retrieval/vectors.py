"""Индекс эмбеддингов по item_id и косинусное сходство."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence


class EmbeddingIndex(ABC):
    """Доступ к векторам по строковому идентификатору."""

    @abstractmethod
    def get_vector(self, item_id: str) -> Sequence[float] | None:
        """Возвращает вектор или None, если ключа нет.

        Args:
            item_id: Идентификатор чанка или варианта вопроса.

        Returns:
            Последовательность float или None.
        """


class DictEmbeddingIndex(EmbeddingIndex):
    """Индекс из словаря в памяти.

    Attributes:
        vectors: Карта item_id → вектор.
    """

    def __init__(self, vectors: Mapping[str, Sequence[float]] | None = None) -> None:
        """Сохраняет копию карты.

        Args:
            vectors: Начальные векторы.
        """
        self.vectors: dict[str, list[float]] = {}
        if vectors is not None:
            for item_id, values in vectors.items():
                self.vectors[item_id] = [float(value) for value in values]

    def get_vector(self, item_id: str) -> Sequence[float] | None:
        """Читает вектор из словаря.

        Args:
            item_id: Ключ.

        Returns:
            Вектор или None.
        """
        values = self.vectors.get(item_id)
        if values is None:
            return None
        return values

    def put(self, item_id: str, values: Sequence[float]) -> None:
        """Записывает вектор.

        Args:
            item_id: Ключ.
            values: Компоненты.
        """
        self.vectors[item_id] = [float(value) for value in values]


class CosineSimilarity:
    """Считает косинус двух векторов."""

    def score(self, left: Sequence[float], right: Sequence[float]) -> float:
        """Возвращает косинусное сходство.

        Args:
            left: Первый вектор.
            right: Второй вектор.

        Returns:
            Скалярное произведение, делённое на произведение длин.

        Raises:
            ValueError: Разные размерности или нулевая длина.
        """
        if len(left) != len(right):
            raise ValueError("Размерности векторов не совпадают")
        dot = 0.0
        left_norm_sq = 0.0
        right_norm_sq = 0.0
        for left_value, right_value in zip(left, right, strict=True):
            dot += left_value * right_value
            left_norm_sq += left_value * left_value
            right_norm_sq += right_value * right_value
        if left_norm_sq == 0.0 or right_norm_sq == 0.0:
            raise ValueError("Нулевая длина вектора")
        return dot / (math.sqrt(left_norm_sq) * math.sqrt(right_norm_sq))
