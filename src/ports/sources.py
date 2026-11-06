"""Порты источников документов, вопросов и ground truth."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.artifacts import GroundTruthItem
from models.documents import DocumentRef
from models.questions import QuestionSet


class DocumentSource(ABC):
    """Источник списка документов прогона."""

    @abstractmethod
    def list_documents(self) -> list[DocumentRef]:
        """Возвращает документы каталога.

        Returns:
            Список DocumentRef.
        """


class QuestionSource(ABC):
    """Источник и валидатор файла вопросов."""

    @abstractmethod
    def load(self) -> QuestionSet:
        """Загружает и проверяет вопросы.

        Returns:
            QuestionSet без дубликатов.

        Raises:
            ValueError: Схема или инварианты нарушены.
        """


class GroundTruthSource(ABC):
    """Источник эталонов eval."""

    @abstractmethod
    def load(self) -> list[GroundTruthItem]:
        """Загружает gt.json.

        Returns:
            Список эталонов.
        """
