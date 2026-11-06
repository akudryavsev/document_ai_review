"""Фаза A2: hybrid retrieval, rerank и сборка контекста."""

from __future__ import annotations

from models.artifacts import ArtifactMeta, ContextBundle
from models.documents import CandidateBatch, CandidateRecord
from models.gateways import RerankPair, RerankRequest
from models.phases import (
    DocumentPhaseOutcome,
    PhaseContext,
    PhasePlan,
    PhasePlanItem,
    PhaseReport,
)
from models.questions import Question
from observe import Observed
from phases.base import Phase
from ports.artifacts import ArtifactStore
from ports.gateways import RerankerGateway
from ports.pipeline import ContextAssembler, HybridRetriever
from ports.trace import TraceSink
from gate import ArtifactGate
from gpu import GpuLease
from context.debug_dump import ContextDebugDump
from retrieval.debug_dump import RetrievalDebugDump
from retrieval.store_index import StoreEmbeddingIndex


class RetrievalPhase(Phase):
    """Retrieval + rerank + ContextAssembler.

    Attributes:
        retriever: HybridRetriever.
        reranker: RerankerGateway.
        assembler: ContextAssembler.
        store: ArtifactStore.
        gate: ArtifactGate.
        lease: GpuLease.
        trace_sink: TraceSink.
        debug_dump: Опциональный дамп реранка.
        vector_index: Индекс байтов хранилища или None у подделки поиска.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: RerankerGateway,
        assembler: ContextAssembler,
        store: ArtifactStore,
        gate: ArtifactGate,
        lease: GpuLease,
        trace_sink: TraceSink,
        debug_dump: RetrievalDebugDump | None = None,
        vector_index: StoreEmbeddingIndex | None = None,
    ) -> None:
        """Сохраняет порты фазы.

        Args:
            retriever: Гибридный поиск.
            reranker: Реранкер.
            assembler: Сборка ContextBundle.
            store: Хранилище.
            gate: ArtifactGate.
            lease: GpuLease.
            trace_sink: TraceSink.
            debug_dump: Дамп или None.
            vector_index: Индекс байтов хранилища. None в тестах на подделке поиска.
        """
        self.retriever = retriever
        self.reranker = reranker
        self.assembler = assembler
        self.store = store
        self.gate = gate
        self.lease = lease
        self.trace_sink = trace_sink
        self.debug_dump = debug_dump
        self.vector_index = vector_index

    def plan(self, context: PhaseContext) -> PhasePlan:
        """Решает compute/skip по meta candidates с upstream-хешами.

        Args:
            context: Контекст фазы.

        Returns:
            PhasePlan.
        """
        items: list[PhasePlanItem] = []
        for document in context.documents:
            meta = self.store.load_candidates_meta(document.doc_id)
            # Без чанков A2 не может skip-нуться как успешный артефакт.
            chunks_meta = self.store.load_chunks_meta(document.doc_id)
            if chunks_meta is None and not context.force:
                items.append(
                    PhasePlanItem(
                        document=document,
                        decision=self.gate.decide(
                            None,
                            schema_version=context.config.schema_version,
                        ),
                    )
                )
                continue
            decision = self.gate.decide(
                meta,
                schema_version=context.config.schema_version,
                force=context.force,
                chunking_hash=context.config.chunking_hash,
                retrieval_config_hash=context.config.retrieval_config_hash,
                reranker_config_hash=context.config.reranker_config_hash,
            )
            items.append(PhasePlanItem(document=document, decision=decision))
        return PhasePlan(phase="A2", items=items)

    @Observed(stage="retrieval_execute")
    def execute(self, plan: PhasePlan, context: PhaseContext) -> PhaseReport:
        """Исполняет поиск по документам и вопросам.

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
        reranker_enabled = context.config.reranker.enabled
        if needs_compute and reranker_enabled:
            self.lease.acquire(self.reranker)

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
            chunks = self.store.load_chunks(doc_id)
            if chunks is None:
                failed += 1
                outcomes.append(
                    DocumentPhaseOutcome(
                        doc_id=doc_id,
                        status="failed",
                        reason="missing_chunks",
                    )
                )
                continue
            try:
                if self.vector_index is not None:
                    chunk_matrix = self.store.load_chunk_vectors(doc_id)
                    question_matrix = self.store.load_question_vectors()
                    self.vector_index.activate(doc_id, chunk_matrix, question_matrix)
                search_candidates: list[CandidateRecord] = []
                for question in questions:
                    batch = self.retriever.search(doc_id, question, chunks)
                    # Кэш до реранка — конкатенация выходов search.
                    search_candidates.extend(batch.candidates)
                    selected = self._select_after_search(
                        batch=batch,
                        question=question,
                        context=context,
                    )
                    empty_search = not selected.candidates
                    bundle = self.assembler.assemble(
                        doc_id=doc_id,
                        question=question,
                        candidates=selected,
                        empty_search=empty_search,
                        chunks=chunks,
                    )
                    bundle.chunking_hash = context.config.chunking_hash
                    bundle.retrieval_config_hash = (
                        context.config.retrieval_config_hash
                    )
                    bundle.reranker_config_hash = (
                        context.config.reranker_config_hash
                    )
                    self.store.save_context(bundle)
                    self._write_context_dump(context, bundle)
                if context.config.retrieval.persist_candidates:
                    candidates_batch = CandidateBatch(
                        doc_id=doc_id,
                        candidates=search_candidates,
                        retrieval_config_hash=context.config.retrieval_config_hash,
                        reranker_config_hash=context.config.reranker_config_hash,
                        chunking_hash=context.config.chunking_hash,
                    )
                    meta = ArtifactMeta(
                        schema_version=context.config.schema_version,
                        chunking_hash=context.config.chunking_hash,
                        retrieval_config_hash=context.config.retrieval_config_hash,
                        reranker_config_hash=context.config.reranker_config_hash,
                    )
                    self.store.save_candidates(candidates_batch, meta)
                computed += 1
                outcomes.append(
                    DocumentPhaseOutcome(doc_id=doc_id, status="computed")
                )
            except Exception as error:  # noqa: BLE001 — устойчивость прогона
                failed += 1
                outcomes.append(
                    DocumentPhaseOutcome(
                        doc_id=doc_id,
                        status="failed",
                        reason=str(error),
                    )
                )
        return PhaseReport(
            phase="A2",
            computed=computed,
            skipped=skipped,
            failed=failed,
            outcomes=outcomes,
        )

    def _select_after_search(
        self,
        batch: CandidateBatch,
        question: Question,
        context: PhaseContext,
    ) -> CandidateBatch:
        """Применяет реранк или усечение без шлюза.

        Args:
            batch: Выход search.
            question: Вопрос с вариантами.
            context: Конфиг фазы.

        Returns:
            Пакет для сборки контекста.
        """
        if not batch.candidates:
            self._write_rerank_dump(
                doc_id=batch.doc_id,
                question_id=question.id,
                pairs=[],
                chunk_scores={},
                early_exit=False,
                selected_after=[],
            )
            return batch
        if not context.config.reranker.enabled:
            selected = self._select_without_reranker(batch, context)
            self._write_rerank_dump(
                doc_id=batch.doc_id,
                question_id=question.id,
                pairs=[],
                chunk_scores={
                    item.chunk_id: float(item.rerank_score or 0.0)
                    for item in selected.candidates
                },
                early_exit=False,
                selected_after=[
                    item.model_dump(mode="json") for item in selected.candidates
                ],
            )
            return selected
        return self._rerank(batch, question, context)

    def _select_without_reranker(
        self,
        batch: CandidateBatch,
        context: PhaseContext,
    ) -> CandidateBatch:
        """Берёт top_n по порядку поиска без early_exit.

        Args:
            batch: Выход search.
            context: Конфиг.

        Returns:
            Усечённый пакет с заполненным rerank_score.
        """
        top_n = context.config.reranker.top_n
        ordered = list(batch.candidates)
        if top_n > 0 and len(ordered) > top_n:
            # Ничья на границе по тому же ключу порядка search (уже в rank).
            boundary_rank = ordered[top_n - 1].rank
            selected = [
                item for item in ordered if item.rank <= boundary_rank
            ]
        else:
            selected = ordered
        count = len(selected)
        updated: list[CandidateRecord] = []
        for item in selected:
            score = float(count - item.rank + 1)
            updated.append(item.model_copy(update={"rerank_score": score}))
        return CandidateBatch(
            doc_id=batch.doc_id,
            candidates=updated,
            retrieval_config_hash=batch.retrieval_config_hash,
            reranker_config_hash=batch.reranker_config_hash,
            chunking_hash=batch.chunking_hash,
        )

    def _rerank(
        self,
        batch: CandidateBatch,
        question: Question,
        context: PhaseContext,
    ) -> CandidateBatch:
        """Скорирует пары вариант × кандидат и отбирает top_n.

        Args:
            batch: Кандидаты до реранка.
            question: Вопрос.
            context: Конфиг.

        Returns:
            CandidateBatch с rerank_score.
        """
        variants = self._variant_texts(question)
        pairs: list[RerankPair] = []
        for candidate in batch.candidates:
            for variant_index, variant_text in enumerate(variants):
                pairs.append(
                    RerankPair(
                        pair_id=f"{candidate.chunk_id}#{variant_index}",
                        query=variant_text,
                        passage=candidate.text,
                        passage_id=candidate.chunk_id,
                    )
                )
        response = self.reranker.infer(RerankRequest(pairs=pairs))
        score_by_pair = {item.pair_id: item.score for item in response.items}

        dump_pairs: list[dict[str, object]] = []
        chunk_scores: dict[str, float] = {}
        scored_candidates: list[CandidateRecord] = []
        for candidate in batch.candidates:
            pair_scores: list[float] = []
            for variant_index, variant_text in enumerate(variants):
                pair_id = f"{candidate.chunk_id}#{variant_index}"
                score = score_by_pair.get(pair_id)
                dump_pairs.append(
                    {
                        "pair_id": pair_id,
                        "query": variant_text,
                        "passage_id": candidate.chunk_id,
                        "score": score,
                    }
                )
                if score is None:
                    continue
                pair_scores.append(score)
            if not pair_scores:
                continue
            best = max(pair_scores)
            chunk_scores[candidate.chunk_id] = best
            scored_candidates.append(
                candidate.model_copy(update={"rerank_score": best})
            )

        early_exit_threshold = context.config.context.early_exit_threshold
        if early_exit_threshold > 0:
            if not scored_candidates:
                self._write_rerank_dump(
                    doc_id=batch.doc_id,
                    question_id=question.id,
                    pairs=dump_pairs,
                    chunk_scores=chunk_scores,
                    early_exit=True,
                    selected_after=[],
                )
                return CandidateBatch(
                    doc_id=batch.doc_id,
                    candidates=[],
                    retrieval_config_hash=batch.retrieval_config_hash,
                    reranker_config_hash=batch.reranker_config_hash,
                    chunking_hash=batch.chunking_hash,
                )
            best_overall = max(
                float(item.rerank_score or 0.0) for item in scored_candidates
            )
            if best_overall < early_exit_threshold:
                self._write_rerank_dump(
                    doc_id=batch.doc_id,
                    question_id=question.id,
                    pairs=dump_pairs,
                    chunk_scores=chunk_scores,
                    early_exit=True,
                    selected_after=[],
                )
                return CandidateBatch(
                    doc_id=batch.doc_id,
                    candidates=[],
                    retrieval_config_hash=batch.retrieval_config_hash,
                    reranker_config_hash=batch.reranker_config_hash,
                    chunking_hash=batch.chunking_hash,
                )

        score_threshold = context.config.reranker.score_threshold
        filtered = scored_candidates
        if score_threshold > 0:
            filtered = [
                item
                for item in scored_candidates
                if float(item.rerank_score or 0.0) > score_threshold
            ]

        filtered.sort(
            key=lambda item: (
                -float(item.rerank_score or 0.0),
                item.chunk_id,
            )
        )
        top_n = context.config.reranker.top_n
        if top_n > 0 and len(filtered) > top_n:
            boundary = float(filtered[top_n - 1].rerank_score or 0.0)
            selected = [
                item
                for item in filtered
                if float(item.rerank_score or 0.0) >= boundary
            ]
        else:
            selected = filtered

        self._write_rerank_dump(
            doc_id=batch.doc_id,
            question_id=question.id,
            pairs=dump_pairs,
            chunk_scores=chunk_scores,
            early_exit=False,
            selected_after=[item.model_dump(mode="json") for item in selected],
        )
        return CandidateBatch(
            doc_id=batch.doc_id,
            candidates=selected,
            retrieval_config_hash=batch.retrieval_config_hash,
            reranker_config_hash=batch.reranker_config_hash,
            chunking_hash=batch.chunking_hash,
        )

    def _variant_texts(self, question: Question) -> list[str]:
        """Тексты вариантов в порядке category, synonyms.

        Args:
            question: Вопрос.

        Returns:
            Список строк запросов.
        """
        texts = [question.category]
        texts.extend(question.synonyms)
        return texts

    def _write_context_dump(self, context: PhaseContext, bundle: ContextBundle) -> None:
        """Пишет Markdown бандла, если retrieval.debug_dump включён.

        Args:
            context: Контекст фазы с каталогом cache.
            bundle: Бандл после записи боевого JSON.
        """
        if not context.config.retrieval.debug_dump:
            return
        ContextDebugDump(
            enabled=True,
            cache_dir=context.config.runtime.cache_dir,
        ).write(bundle)

    def _write_rerank_dump(
        self,
        doc_id: str,
        question_id: str,
        pairs: list[dict[str, object]],
        chunk_scores: dict[str, float],
        early_exit: bool,
        selected_after: list[dict[str, object]],
    ) -> None:
        """Дописывает блок rerank в debug-файл.

        Args:
            doc_id: Документ.
            question_id: Вопрос.
            pairs: Пары реранка.
            chunk_scores: Max по чанкам.
            early_exit: Сработал ли ранний выход.
            selected_after: Итог для сборки.
        """
        if self.debug_dump is None:
            return
        self.debug_dump.update_rerank(
            doc_id,
            question_id,
            {
                "pairs": pairs,
                "chunk_scores": chunk_scores,
                "early_exit": early_exit,
                "selected_after": selected_after,
            },
        )
