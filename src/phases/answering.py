"""Фаза B: верификация и генерация ответов."""

from __future__ import annotations

from models.artifacts import AnswerRecord, ArtifactMeta
from models.phases import (
    DocumentPhaseOutcome,
    PhaseContext,
    PhasePlan,
    PhasePlanItem,
    PhaseReport,
)
from observe import Observed
from phases.base import Phase
from ports.artifacts import ArtifactStore
from ports.gateways import LlmGateway
from ports.pipeline import AnswerGenerator, AnswerVerifier, ResultSink
from ports.trace import TraceSink
from gate import ArtifactGate
from gpu import GpuLease


class AnsweringPhase(Phase):
    """Ответы LLM по каждой паре документ × вопрос.

    Верификатор хранится для следующего запуска и в этом прогоне не вызывается.
    Пустой поиск закрывает генератор без LLM.

    Attributes:
        llm: LlmGateway.
        verifier: AnswerVerifier или None. В первом запуске в граф не входит
            и не вызывается.
        generator: AnswerGenerator.
        result_sink: ResultSink.
        store: ArtifactStore.
        gate: ArtifactGate.
        lease: GpuLease.
        trace_sink: TraceSink.
    """

    def __init__(
        self,
        llm: LlmGateway,
        generator: AnswerGenerator,
        result_sink: ResultSink,
        store: ArtifactStore,
        gate: ArtifactGate,
        lease: GpuLease,
        trace_sink: TraceSink,
        verifier: AnswerVerifier | None = None,
    ) -> None:
        """Сохраняет порты фазы.

        Args:
            llm: Шлюз LLM.
            generator: Генератор ответов.
            verifier: Верификатор. В первом запуске не передаётся.
            result_sink: Запись результатов.
            store: Хранилище контекстов и meta.
            gate: ArtifactGate.
            lease: GpuLease.
            trace_sink: TraceSink.
        """
        self.llm = llm
        self.verifier = verifier
        self.generator = generator
        self.result_sink = result_sink
        self.store = store
        self.gate = gate
        self.lease = lease
        self.trace_sink = trace_sink

    def plan(self, context: PhaseContext) -> PhasePlan:
        """Решает compute/skip по meta ответов и upstream-хешам.

        Args:
            context: Контекст фазы.

        Returns:
            PhasePlan.
        """
        items: list[PhasePlanItem] = []
        for document in context.documents:
            meta = self.store.load_answers_meta(document.doc_id)
            decision = self.gate.decide(
                meta,
                schema_version=context.config.schema_version,
                force=context.force,
                chunking_hash=context.config.chunking_hash,
                retrieval_config_hash=context.config.retrieval_config_hash,
                reranker_config_hash=context.config.reranker_config_hash,
            )
            items.append(PhasePlanItem(document=document, decision=decision))
        return PhasePlan(phase="B", items=items)

    @Observed(stage="answering_execute")
    def execute(self, plan: PhasePlan, context: PhaseContext) -> PhaseReport:
        """Генерирует ответы. Верификатор в первом запуске не вызывается.

        Args:
            plan: PhasePlan.
            context: Контекст с вопросами.

        Returns:
            PhaseReport.
        """
        outcomes: list[DocumentPhaseOutcome] = []
        computed = 0
        skipped = 0
        failed = 0
        needs_compute = any(
            item.decision.action == "compute" for item in plan.items
        )
        if needs_compute:
            self.lease.acquire(self.llm)

        questions = context.questions.questions if context.questions else []
        for item in plan.items:
            doc_id = item.document.doc_id
            if item.decision.action == "skip":
                skipped += 1
                outcomes.append(
                    DocumentPhaseOutcome(
                        doc_id=doc_id,
                        status="skipped",
                        reason=item.decision.reason,
                    )
                )
                continue
            doc_failed = False
            pair_failures = 0
            for question in questions:
                bundle = self.store.load_context(doc_id, question.id)
                if bundle is None:
                    pair_failures += 1
                    doc_failed = True
                    answer_name = context.config.answer.model or ""
                    record = AnswerRecord(
                        doc_id=doc_id,
                        question_id=question.id,
                        category=question.category,
                        found=False,
                        answer="",
                        reason="missing_context",
                        status="failed",
                        model=answer_name,
                        embedding_model=context.config.embedding.model or "",
                        reranker_model=context.config.reranker.model or "",
                        answer_model=answer_name,
                    )
                    self.result_sink.write_answer(record)
                    continue
                # Первый запуск не фильтрует фрагменты: verify не вызывается.
                record = self.generator.generate(
                    bundle,
                    question,
                    context.questions,
                )
                self.result_sink.write_answer(record)
                if record.status == "failed":
                    pair_failures += 1
                    doc_failed = True
            if doc_failed and pair_failures == len(questions):
                failed += 1
                outcomes.append(
                    DocumentPhaseOutcome(
                        doc_id=doc_id,
                        status="failed",
                        reason="pairs_failed",
                    )
                )
            else:
                computed += 1
                outcomes.append(
                    DocumentPhaseOutcome(doc_id=doc_id, status="computed")
                )
                self.store.save_answers_meta(
                    doc_id,
                    ArtifactMeta(
                        schema_version=context.config.schema_version,
                        chunking_hash=context.config.chunking_hash,
                        retrieval_config_hash=(
                            context.config.retrieval_config_hash
                        ),
                        reranker_config_hash=(
                            context.config.reranker_config_hash
                        ),
                    ),
                )
        return PhaseReport(
            phase="B",
            computed=computed,
            skipped=skipped,
            failed=failed,
            outcomes=outcomes,
        )
