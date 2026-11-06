"""Пакетный прогон AnalysisRun."""

from __future__ import annotations

from models.documents import DocumentRef
from models.phases import PhaseContext
from models.questions import QuestionSet
from models.requests import PhaseName, RunRequest
from models.summary import ModelVersions, PhaseCounters, RunSummary
from observe import Observed
from phases.answering import AnsweringPhase
from phases.indexing import IndexingPhase
from phases.retrieval import RetrievalPhase
from ports.pipeline import ResultSink
from ports.trace import TraceSink
from gpu import GpuLease


class AnalysisRun:
    """Один пакетный прогон фаз A1 → A2 → B.

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
        """Исполняет запрошенные фазы последовательно.

        Returns:
            RunSummary.
        """
        config = self.request.resolved_config()
        phase_map = {
            "A1": self.indexing,
            "A2": self.retrieval,
            "B": self.answering,
        }
        phase_counters: dict[PhaseName, PhaseCounters] = {}
        details: dict[str, object] = {}
        zero_chunks: list[str] = []

        for phase_name in self.request.phases:
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
            # Между фазами GPU отпускается.
            self.lease.release()
            phase_counters[phase_name] = PhaseCounters(
                computed=report.computed,
                skipped=report.skipped,
                failed=report.failed,
            )
            for outcome in report.outcomes:
                if outcome.reason == "zero_chunks":
                    zero_chunks.append(outcome.doc_id)

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
                embedding=config.embedding.model,
                reranker=config.reranker.model,
                answer=config.answer.model,
                verification=config.verification.model,
            ),
            phases=phase_counters,
            requested_phases=list(self.request.phases),
            details=details,
        )
        self.result_sink.write_summary(summary)
        return summary
