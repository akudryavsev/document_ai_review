"""Фасад DocumentAnalysisService."""

from __future__ import annotations

from models.questions import QuestionSet
from models.requests import EvalRequest, RunRequest
from models.summary import EvalReport, RunSummary
from observe import Observed
from phases.answering import AnsweringPhase
from phases.indexing import IndexingPhase
from phases.retrieval import RetrievalPhase
from ports.pipeline import ResultSink
from ports.sources import DocumentSource, QuestionSource
from ports.trace import TraceSink
from gpu import GpuLease
from runs.analysis import AnalysisRun


class DocumentAnalysisService:
    """Единственный фасад, который вызывают входные точки.

    Attributes:
        document_source: Источник документов.
        question_source: Источник вопросов.
        indexing: Фаза A1.
        retrieval: Фаза A2.
        answering: Фаза B.
        result_sink: Запись результатов.
        lease: GpuLease.
        trace_sink: TraceSink.
    """

    def __init__(
        self,
        document_source: DocumentSource,
        question_source: QuestionSource,
        indexing: IndexingPhase,
        retrieval: RetrievalPhase,
        answering: AnsweringPhase,
        result_sink: ResultSink,
        lease: GpuLease,
        trace_sink: TraceSink,
    ) -> None:
        """Сохраняет зависимости фасада.

        Args:
            document_source: DocumentSource.
            question_source: QuestionSource.
            indexing: IndexingPhase.
            retrieval: RetrievalPhase.
            answering: AnsweringPhase.
            result_sink: ResultSink.
            lease: GpuLease.
            trace_sink: TraceSink.
        """
        self.document_source = document_source
        self.question_source = question_source
        self.indexing = indexing
        self.retrieval = retrieval
        self.answering = answering
        self.result_sink = result_sink
        self.lease = lease
        self.trace_sink = trace_sink

    @Observed(stage="service_run")
    def run(self, request: RunRequest) -> RunSummary:
        """Создаёт AnalysisRun и исполняет запрошенные фазы.

        Args:
            request: RunRequest.

        Returns:
            RunSummary.
        """
        questions = self._load_questions(request)
        documents = self.document_source.list_documents()
        analysis = AnalysisRun(
            request=request,
            documents=documents,
            questions=questions,
            indexing=self.indexing,
            retrieval=self.retrieval,
            answering=self.answering,
            result_sink=self.result_sink,
            lease=self.lease,
            trace_sink=self.trace_sink,
        )
        return analysis.execute()

    def evaluate(self, request: EvalRequest) -> EvalReport:
        """Оболочка eval до отдельной спецификации метрик.

        Args:
            request: EvalRequest.

        Returns:
            EvalReport-заглушка.

        Raises:
            NotImplementedError: Метрики отложены спецификацией eval.
        """
        raise NotImplementedError(
            "EvaluationRun появится после спецификации eval"
        )

    def _load_questions(self, request: RunRequest) -> QuestionSet | None:
        """Загружает вопросы, если в прогоне есть поиск или ответы.

        Для одного A1 файл вопросов не нужен: чанкинг от него не зависит.

        Args:
            request: Запрос с перечнем фаз.

        Returns:
            QuestionSet для A2/B либо None, если запрошен только A1.

        Raises:
            ValueError: Схема вопросов не прошла проверку, когда загрузка обязательна.
        """
        requested = set(request.phases)
        needs_questions = "A2" in requested or "B" in requested
        if not needs_questions:
            return None
        # Валидация вопросов — до старта фаз, только когда они нужны.
        return self.question_source.load()
