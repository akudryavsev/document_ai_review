"""Порты пайплайна: extract/clean/chunk/retrieve/answer."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.artifacts import AnswerRecord, ContextBundle
from models.documents import (
    CandidateBatch,
    ChunkBatch,
    CleanedDocument,
    DocumentRef,
    ExtractedDocument,
)
from models.questions import Question, QuestionSet


class Extractor(ABC):
    """Извлекает текст из документа."""

    @abstractmethod
    def extract(self, document: DocumentRef) -> ExtractedDocument:
        """Читает документ и возвращает извлечённый текст.

        Args:
            document: Ссылка на файл.

        Returns:
            ExtractedDocument; при отсутствии текстового слоя —
            failure_reason заполнен.
        """


class Cleaner(ABC):
    """Очищает извлечённый текст."""

    @abstractmethod
    def clean(self, document: ExtractedDocument) -> CleanedDocument:
        """Очищает текст документа.

        Args:
            document: Извлечённый документ.

        Returns:
            CleanedDocument.
        """


class Chunker(ABC):
    """Делит очищенный документ на чанки."""

    @abstractmethod
    def chunk(self, document: CleanedDocument) -> ChunkBatch:
        """Строит чанки.

        Args:
            document: Очищенный документ.

        Returns:
            ChunkBatch.
        """


class HybridRetriever(ABC):
    """Гибридный поиск кандидатов."""

    @abstractmethod
    def search(
        self,
        doc_id: str,
        question: Question,
        chunks: ChunkBatch,
    ) -> CandidateBatch:
        """Ищет кандидатов по category и synonyms.

        Args:
            doc_id: Идентификатор документа.
            question: Вопрос с synonyms.
            chunks: Чанки документа.

        Returns:
            CandidateBatch до реранка.
        """


class ContextAssembler(ABC):
    """Собирает ContextBundle из кандидатов после реранка."""

    @abstractmethod
    def assemble(
        self,
        doc_id: str,
        question: Question,
        candidates: CandidateBatch,
        empty_search: bool,
        chunks: ChunkBatch,
    ) -> ContextBundle:
        """Строит бандл для фазы B.

        Args:
            doc_id: Идентификатор документа.
            question: Вопрос.
            candidates: Кандидаты после реранка.
            empty_search: Признак пустого поиска.
            chunks: Чанки документа для расширения секции.

        Returns:
            ContextBundle.
        """


class AnswerVerifier(ABC):
    """Верифицирует кандидатов перед ответом."""

    @abstractmethod
    def verify(
        self,
        bundle: ContextBundle,
        question: Question,
    ) -> ContextBundle:
        """Фильтрует кандидатов бандла.

        Args:
            bundle: Контекст с кандидатами.
            question: Вопрос (category).

        Returns:
            Бандл с отфильтрованными кандидатами.

        Raises:
            Exception: При сбое — фаза делает graceful degradation.
        """


class AnswerGenerator(ABC):
    """Генерирует AnswerRecord через LlmGateway."""

    @abstractmethod
    def generate(
        self,
        bundle: ContextBundle,
        question: Question,
        questions: QuestionSet | None = None,
    ) -> AnswerRecord:
        """Строит ответ для пары документ × вопрос.

        Args:
            bundle: Контекст (возможно пустой).
            question: Вопрос; текст для LLM = category.
            questions: Полный набор, если нужен генератору.

        Returns:
            AnswerRecord.
        """


class ResultSink(ABC):
    """Пишет результаты прогона на диск или в память."""

    @abstractmethod
    def write_answer(self, record: AnswerRecord) -> None:
        """Пишет одну строку ответа.

        Args:
            record: AnswerRecord.
        """

    @abstractmethod
    def write_summary(self, summary: object) -> None:
        """Пишет RunSummary.

        Args:
            summary: Сводка прогона.
        """
