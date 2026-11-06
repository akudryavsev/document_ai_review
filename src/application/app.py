"""Композиционный корень Application."""

from __future__ import annotations

from answer.generator import AnswerGenerator as DocumentAnswerGenerator
from application.result_sink import FileResultSink
from application.service import DocumentAnalysisService
from chunking.chunker import Chunker as DocumentChunker
from context.assembler import ContextAssembler as DocumentContextAssembler
from clean.text_cleaner import Cleaner as DocumentCleaner
from extract.extractors import FormatAwareExtractor, PdfExtractor
from gateways import (
    LlamaEmbeddingGateway,
    LlamaHttpClient,
    LlamaLlmGateway,
    LlamaRerankerGateway,
)
from retrieval.debug_dump import RetrievalDebugDump
from retrieval.engine import HybridSearcher
from retrieval.store_index import StoreEmbeddingIndex
from store.file_store import FileArtifactStore
from gate import ArtifactGate
from gpu import GpuLease
from jobs.inline import InlineJobExecutor
from models.config import AppConfig
from observe import LoggingTraceSink
from phases.answering import AnsweringPhase
from phases.indexing import IndexingPhase
from phases.retrieval import RetrievalPhase
from ports.file_sources import FileDocumentSource, FileQuestionSource
from ports.jobs import JobExecutor


class Application:
    """Собирает адаптеры портов и фасад прогона.

    Attributes:
        config: AppConfig.
    """

    def __init__(self, config: AppConfig) -> None:
        """Сохраняет конфиг.

        Args:
            config: AppConfig.
        """
        self.config = config

    def build(self) -> DocumentAnalysisService:
        """Создаёт DocumentAnalysisService с шлюзами llama-server.

        Извлечение, очистка, чанкинг, хранилище, поиск и ответ — боевые.
        Три шлюза ходят в один контейнер. Верификатор в граф не входит.
        Источники документов и вопросов — файловые.

        Returns:
            DocumentAnalysisService.
        """
        trace_sink = LoggingTraceSink()
        lease = GpuLease(trace_sink)
        store = FileArtifactStore(
            cache_dir=self.config.runtime.cache_dir,
            index_dir=self.config.phases.index_dir,
            contexts_dir=self.config.phases.contexts_dir,
            trace_sink=trace_sink,
        )
        gate = ArtifactGate()
        client = LlamaHttpClient(
            base_url=self.config.inference.base_url,
            timeout_sec=self.config.inference.request_timeout_sec,
            poll_interval_sec=self.config.inference.poll_interval_sec,
        )
        embedding = LlamaEmbeddingGateway(
            client=client,
            preset=self.config.embedding.model,
            timeout_sec=self.config.inference.request_timeout_sec,
            batch_size=self.config.embedding.batch_size,
            dtype=self.config.embedding.dtype,
        )
        reranker = LlamaRerankerGateway(
            client=client,
            preset=self.config.reranker.model,
            timeout_sec=self.config.inference.request_timeout_sec,
        )
        llm = LlamaLlmGateway(
            client=client,
            preset=self.config.answer.model,
            timeout_sec=self.config.inference.request_timeout_sec,
        )
        vector_index = StoreEmbeddingIndex()
        result_sink = FileResultSink(self.config.runtime.results_dir)
        indexing = IndexingPhase(
            extractor=FormatAwareExtractor(
                pdf_extractor=PdfExtractor(
                    pdf_text_layer_required=(
                        self.config.extraction.pdf_text_layer_required
                    ),
                ),
            ),
            cleaner=DocumentCleaner(),
            chunker=DocumentChunker(self.config.chunking),
            embedding=embedding,
            store=store,
            gate=gate,
            lease=lease,
            trace_sink=trace_sink,
        )
        retrieval = RetrievalPhase(
            retriever=HybridSearcher(
                config=self.config.retrieval,
                embeddings=vector_index,
                debug_dump=RetrievalDebugDump(
                    enabled=self.config.retrieval.debug_dump,
                    cache_dir=self.config.runtime.cache_dir,
                ),
                trace_sink=trace_sink,
            ),
            reranker=reranker,
            assembler=DocumentContextAssembler(
                config=self.config.context,
                embedding_model=self.config.embedding.model,
                reranker_model=self.config.reranker.model,
                top_k=self.config.retrieval.top_k,
            ),
            store=store,
            gate=gate,
            lease=lease,
            trace_sink=trace_sink,
            vector_index=vector_index,
        )
        answering = AnsweringPhase(
            llm=llm,
            generator=DocumentAnswerGenerator(
                answer_config=self.config.answer,
                runtime_config=self.config.runtime,
                llm=llm,
            ),
            result_sink=result_sink,
            store=store,
            gate=gate,
            lease=lease,
            trace_sink=trace_sink,
        )
        return DocumentAnalysisService(
            document_source=FileDocumentSource(self.config.input.documents_dir),
            question_source=FileQuestionSource(self.config.input.questions_file),
            indexing=indexing,
            retrieval=retrieval,
            answering=answering,
            result_sink=result_sink,
            lease=lease,
            trace_sink=trace_sink,
        )

    def build_job_executor(self) -> JobExecutor:
        """Собирает InlineJobExecutor над тем же фасадом.

        Returns:
            InlineJobExecutor.
        """
        return InlineJobExecutor(service=self.build())
