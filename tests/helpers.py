"""Сборка DocumentAnalysisService из RunHarness."""

from __future__ import annotations

from application.service import DocumentAnalysisService
from phases.answering import AnsweringPhase
from phases.indexing import IndexingPhase
from phases.retrieval import RetrievalPhase
from tests.support import RunHarness


class ServiceFactory:
    """Собирает фасад из харнесса подделок."""

    def build(self, harness: RunHarness) -> DocumentAnalysisService:
        """Создаёт DocumentAnalysisService.

        Args:
            harness: RunHarness.

        Returns:
            DocumentAnalysisService на подделках.
        """
        indexing = IndexingPhase(
            extractor=harness.extractor,
            cleaner=harness.cleaner,
            chunker=harness.chunker,
            embedding=harness.embedding,
            store=harness.store,
            gate=harness.gate,
            lease=harness.lease,
            trace_sink=harness.trace,
        )
        retrieval = RetrievalPhase(
            retriever=harness.retriever,
            reranker=harness.reranker,
            assembler=harness.assembler,
            store=harness.store,
            gate=harness.gate,
            lease=harness.lease,
            trace_sink=harness.trace,
        )
        answering = AnsweringPhase(
            llm=harness.llm,
            verifier=harness.verifier,
            generator=harness.generator,
            result_sink=harness.result_sink,
            store=harness.store,
            gate=harness.gate,
            lease=harness.lease,
            trace_sink=harness.trace,
        )
        return DocumentAnalysisService(
            document_source=harness.document_source,
            question_source=harness.question_source,
            indexing=indexing,
            retrieval=retrieval,
            answering=answering,
            result_sink=harness.result_sink,
            lease=harness.lease,
            trace_sink=harness.trace,
        )
