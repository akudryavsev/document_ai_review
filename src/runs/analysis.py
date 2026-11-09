"""Пакетный прогон AnalysisRun."""

from __future__ import annotations

from models.documents import DocumentRef
from models.config import AppConfig
from models.phases import PhaseContext, PhaseReport
from models.questions import QuestionSet
from models.requests import PhaseName, RunRequest
from models.summary import CombinationSummary, ModelVersions, PhaseCounters, RunSummary
from observe import Observed
from phases.answering import AnsweringPhase
from phases.indexing import IndexingPhase
from phases.retrieval import RetrievalPhase
from ports.pipeline import ResultSink
from ports.trace import TraceSink
from gpu import GpuLease


class AnalysisRun:
    """Пакетный прогон троек моделей в порядке A1 → A2 → B.

    Attributes:
        request: RunRequest.
        documents: Документы прогона.
        questions: Вопросы прогона или None, если запрошен только A1.
        indexing: IndexingPhase.
        retrieval: RetrievalPhase.
        answering: AnsweringPhase.
        result_sink: ResultSink.
        lease: GpuLease.
        trace_sink: TraceSink.
    """

    def __init__(
        self,
        request: RunRequest,
        documents: list[DocumentRef],
        questions: QuestionSet | None,
        indexing: IndexingPhase,
        retrieval: RetrievalPhase,
        answering: AnsweringPhase,
        result_sink: ResultSink,
        lease: GpuLease,
        trace_sink: TraceSink,
    ) -> None:
        """Сохраняет зависимости прогона.

        Args:
            request: Запрос CLI/API.
            documents: Список документов.
            questions: Набор вопросов или None для прогона только A1.
            indexing: Фаза A1.
            retrieval: Фаза A2.
            answering: Фаза B.
            result_sink: Запись результатов.
            lease: Аренда GPU.
            trace_sink: Наблюдаемость.
        """
        self.request = request
        self.documents = documents
        self.questions = questions
        self.indexing = indexing
        self.retrieval = retrieval
        self.answering = answering
        self.result_sink = result_sink
        self.lease = lease
        self.trace_sink = trace_sink

    @Observed(stage="analysis_run")
    def execute(self) -> RunSummary:
        """Обходит тройки моделей и для каждой держит порядок A1 → A2 → B.

        A1 считается один раз на эмбеддер, A2 — на пару с реранкером,
        B — на каждую модель ответа.

        Returns:
            RunSummary со списком комбинаций.
        """
        config = self.request.resolved_config()
        requested = list(self.request.phases)
        totals = {
            phase_name: PhaseCounters() for phase_name in requested
        }
        combinations: list[CombinationSummary] = []
        details: dict[str, object] = {}
        zero_chunks: list[str] = []
        embedding_names = list(config.embedding.models or [])
        reranker_names = list(config.reranker.models or [])
        answer_names = list(config.answer.models or [])

        for embedding_name in embedding_names:
            indexing_report = None
            if "A1" in requested:
                self._bind(embedding_name, reranker_names[0], answer_names[0])
                scoped = self._scoped(
                    config, embedding_name, reranker_names[0], answer_names[0]
                )
                indexing_report = self._execute_phase("A1", scoped)
                self._add_counters(totals, "A1", indexing_report)
                zero_chunks.extend(self._zero_chunk_ids(indexing_report))
            for reranker_name in reranker_names:
                retrieval_report = None
                if "A2" in requested:
                    self._bind(embedding_name, reranker_name, answer_names[0])
                    scoped = self._scoped(
                        config, embedding_name, reranker_name, answer_names[0]
                    )
                    retrieval_report = self._execute_phase("A2", scoped)
                    self._add_counters(totals, "A2", retrieval_report)
                    zero_chunks.extend(self._zero_chunk_ids(retrieval_report))
                for answer_name in answer_names:
                    answering_report = None
                    if "B" in requested:
                        self._bind(embedding_name, reranker_name, answer_name)
                        scoped = self._scoped(
                            config, embedding_name, reranker_name, answer_name
                        )
                        answering_report = self._execute_phase("B", scoped)
                        self._add_counters(totals, "B", answering_report)
                        zero_chunks.extend(self._zero_chunk_ids(answering_report))
                    phase_counters: dict[PhaseName, PhaseCounters] = {}
                    if indexing_report is not None:
                        phase_counters["A1"] = self._counters_of(indexing_report)
                    if retrieval_report is not None:
                        phase_counters["A2"] = self._counters_of(retrieval_report)
                    if answering_report is not None:
                        phase_counters["B"] = self._counters_of(answering_report)
                    combinations.append(
                        CombinationSummary(
                            embedding_model=embedding_name,
                            reranker_model=reranker_name,
                            answer_model=answer_name,
                            phases=phase_counters,
                        )
                    )

        if zero_chunks:
            details["zero_chunks"] = zero_chunks

        answers = getattr(self.result_sink, "answers", [])
        failed_answers = [
            answer for answer in answers if getattr(answer, "status", "ok") == "failed"
        ]
        total_pairs = max(len(answers), 1) if answers else 1
        error_rate = len(failed_answers) / total_pairs if answers else 0.0
        if answers:
            average_latency = sum(
                getattr(answer, "latency_ms", 0.0) for answer in answers
            ) / len(answers)
            average_prompt = sum(
                getattr(answer.tokens, "prompt", 0) for answer in answers
            ) / len(answers)
            average_completion = sum(
                getattr(answer.tokens, "completion", 0) for answer in answers
            ) / len(answers)
        else:
            average_latency = 0.0
            average_prompt = 0.0
            average_completion = 0.0

        question_count = 0
        if self.questions is not None:
            question_count = len(self.questions.questions)
        summary = RunSummary(
            documents=len(self.documents),
            questions=question_count,
            error_rate=error_rate,
            average_latency_ms=average_latency,
            average_prompt_tokens=average_prompt,
            average_completion_tokens=average_completion,
            model_versions=ModelVersions(
                embedding=list(config.embedding.models or []),
                reranker=list(config.reranker.models or []),
                answer=list(config.answer.models or []),
                verification=config.verification.model,
            ),
            combinations=combinations,
            phases=totals,
            requested_phases=requested,
            details=details,
        )
        self.result_sink.write_summary(summary)
        return summary

    def _execute_phase(self, phase_name: PhaseName, config: AppConfig) -> PhaseReport:
        """Планирует и исполняет одну фазу для текущей тройки.

        Args:
            phase_name: A1, A2 или B.
            config: Конфиг, в котором model ролей указывает на текущие файлы.

        Returns:
            Отчёт фазы.
        """
        phase_map = {
            "A1": self.indexing,
            "A2": self.retrieval,
            "B": self.answering,
        }
        phase = phase_map[phase_name]
        context = PhaseContext(
            config=config,
            documents=self.documents,
            force=phase_name in self.request.force_phases,
            questions=self.questions,
            requested_phases=list(self.request.phases),
        )
        plan = phase.plan(context)
        report = phase.execute(plan, context)
        # Следующая роль не должна застать предыдущую модель на GPU.
        self.lease.release()
        return report

    def _scoped(
        self,
        config: AppConfig,
        embedding_name: str,
        reranker_name: str,
        answer_name: str,
    ) -> AppConfig:
        """Копирует конфиг и подставляет имена текущей тройки.

        Args:
            config: Исходный конфиг прогона.
            embedding_name: Файл эмбеддера.
            reranker_name: Файл реранкера.
            answer_name: Файл модели ответа.

        Returns:
            Копия, у которой список роли и model — текущий файл.
        """
        scoped = config.model_copy(deep=True)
        scoped.embedding.model = embedding_name
        scoped.embedding.models = [embedding_name]
        scoped.reranker.model = reranker_name
        scoped.reranker.models = [reranker_name]
        scoped.answer.model = answer_name
        scoped.answer.models = [answer_name]
        self._rebind_workers(embedding_name, reranker_name, answer_name)
        return scoped

    def _rebind_workers(
        self,
        embedding_name: str,
        reranker_name: str,
        answer_name: str,
    ) -> None:
        """Подставляет текущие файлы в шлюзы, сборщик и генератор.

        Args:
            embedding_name: Файл эмбеддера.
            reranker_name: Файл реранкера.
            answer_name: Файл модели ответа.
        """
        embedding = getattr(self.indexing, "embedding", None)
        if hasattr(embedding, "preset"):
            embedding.preset = embedding_name
        reranker = getattr(self.retrieval, "reranker", None)
        if hasattr(reranker, "preset"):
            reranker.preset = reranker_name
        llm = getattr(self.answering, "llm", None)
        if hasattr(llm, "preset"):
            llm.preset = answer_name
        assembler = getattr(self.retrieval, "assembler", None)
        if hasattr(assembler, "embedding_model"):
            assembler.embedding_model = embedding_name
        if hasattr(assembler, "reranker_model"):
            assembler.reranker_model = reranker_name
        generator = getattr(self.answering, "generator", None)
        if hasattr(generator, "embedding_model"):
            generator.embedding_model = embedding_name
        if hasattr(generator, "reranker_model"):
            generator.reranker_model = reranker_name
        answer_config = getattr(generator, "answer_config", None)
        if answer_config is not None and hasattr(answer_config, "model"):
            answer_config.model = answer_name
        if hasattr(generator, "answer_model"):
            generator.answer_model = answer_name

    def _bind(
        self,
        embedding_name: str,
        reranker_name: str,
        answer_name: str,
    ) -> None:
        """Включает префиксы хранилища для текущей тройки.

        Args:
            embedding_name: Файл эмбеддера.
            reranker_name: Файл реранкера.
            answer_name: Файл модели ответа.
        """
        for store in (
            getattr(self.indexing, "store", None),
            getattr(self.retrieval, "store", None),
            getattr(self.answering, "store", None),
        ):
            if hasattr(store, "bind_models"):
                store.bind_models(embedding_name, reranker_name, answer_name)

    def _add_counters(
        self,
        totals: dict[PhaseName, PhaseCounters],
        phase_name: PhaseName,
        report: PhaseReport,
    ) -> None:
        """Прибавляет счётчики одного исполнения к сводке прогона.

        Args:
            totals: Счётчики запрошенных фаз.
            phase_name: Фаза отчёта.
            report: Отчёт одного исполнения.
        """
        current = totals[phase_name]
        totals[phase_name] = PhaseCounters(
            computed=current.computed + report.computed,
            skipped=current.skipped + report.skipped,
            failed=current.failed + report.failed,
        )

    def _counters_of(self, report: PhaseReport) -> PhaseCounters:
        """Копирует счётчики отчёта в элемент комбинации.

        Args:
            report: Отчёт фазы.

        Returns:
            Счётчики computed, skipped и failed.
        """
        return PhaseCounters(
            computed=report.computed,
            skipped=report.skipped,
            failed=report.failed,
        )

    def _zero_chunk_ids(self, report: PhaseReport) -> list[str]:
        """Собирает документы с причиной zero_chunks.

        Args:
            report: Отчёт фазы.

        Returns:
            Идентификаторы документов.
        """
        return [
            outcome.doc_id
            for outcome in report.outcomes
            if outcome.reason == "zero_chunks"
        ]
