"""Вспомогательные типы для тестов верхнего уровня."""

from __future__ import annotations

from dataclasses import dataclass, field

from fakes import (
    FakeAnswerGenerator,
    FakeAnswerVerifier,
    FakeChunker,
    FakeCleaner,
    FakeContextAssembler,
    FakeEmbeddingGateway,
    FakeExtractor,
    FakeHybridRetriever,
    FakeLlmGateway,
    FakeRerankerGateway,
    InMemoryArtifactStore,
    RecordingResultSink,
    RecordingTraceSink,
)
from gate import ArtifactGate
from gpu import GpuLease
from models.config import AppConfig
from models.documents import DocumentRef
from models.questions import Question, QuestionSet
from models.requests import PhaseName, RunRequest
from ports.sources import DocumentSource, QuestionSource


class StaticDocumentSource(DocumentSource):
    """DocumentSource из заранее заданного списка.

    Attributes:
        documents: Список DocumentRef.
    """

    def __init__(self, documents: list[DocumentRef]) -> None:
        """Сохраняет документы.

        Args:
            documents: Список DocumentRef.
        """
        self.documents = documents

    def list_documents(self) -> list[DocumentRef]:
        """Возвращает сохранённый список.

        Returns:
            Список DocumentRef.
        """
        return list(self.documents)


class StaticQuestionSource(QuestionSource):
    """QuestionSource из заранее заданного QuestionSet.

    Attributes:
        question_set: Набор вопросов.
    """

    def __init__(self, question_set: QuestionSet) -> None:
        """Сохраняет набор.

        Args:
            question_set: QuestionSet.
        """
        self.question_set = question_set

    def load(self) -> QuestionSet:
        """Возвращает сохранённый набор.

        Returns:
            QuestionSet.
        """
        return self.question_set


@dataclass
class RunHarness:
    """Собранный граф подделок для интеграционных тестов.

    Attributes:
        config: AppConfig прогона.
        documents: Список DocumentRef.
        questions: QuestionSet.
        store: InMemoryArtifactStore.
        trace: RecordingTraceSink.
        lease: GpuLease.
        embedding: FakeEmbeddingGateway.
        reranker: FakeRerankerGateway.
        llm: FakeLlmGateway.
        extractor: FakeExtractor.
        cleaner: FakeCleaner.
        chunker: FakeChunker.
        retriever: FakeHybridRetriever.
        assembler: FakeContextAssembler.
        verifier: FakeAnswerVerifier.
        generator: FakeAnswerGenerator.
        result_sink: RecordingResultSink.
        gate: ArtifactGate.
        document_source: StaticDocumentSource.
        question_source: StaticQuestionSource.
    """

    config: AppConfig
    documents: list[DocumentRef]
    questions: QuestionSet
    store: InMemoryArtifactStore
    trace: RecordingTraceSink
    lease: GpuLease
    embedding: FakeEmbeddingGateway
    reranker: FakeRerankerGateway
    llm: FakeLlmGateway
    extractor: FakeExtractor
    cleaner: FakeCleaner
    chunker: FakeChunker
    retriever: FakeHybridRetriever
    assembler: FakeContextAssembler
    verifier: FakeAnswerVerifier
    generator: FakeAnswerGenerator
    result_sink: RecordingResultSink
    gate: ArtifactGate = field(default_factory=ArtifactGate)
    document_source: StaticDocumentSource | None = None
    question_source: StaticQuestionSource | None = None

    def __post_init__(self) -> None:
        """Собирает статические источники, если не заданы."""
        if self.document_source is None:
            self.document_source = StaticDocumentSource(self.documents)
        if self.question_source is None:
            self.question_source = StaticQuestionSource(self.questions)

    def run_request(
        self,
        phases: list[PhaseName] | None = None,
        force_phases: set[PhaseName] | None = None,
    ) -> RunRequest:
        """Строит RunRequest для харнесса.

        Args:
            phases: Список фаз.
            force_phases: Принудительные фазы.

        Returns:
            RunRequest.
        """
        return RunRequest(
            config=self.config,
            phases=phases or ["A1", "A2", "B"],
            force_phases=force_phases or set(),
        )


def build_harness(
    *,
    no_text_layer_docs: set[str] | None = None,
    empty_docs: set[str] | None = None,
    empty_pairs: set[tuple[str, str]] | None = None,
    verifier_fail: bool = False,
    fail_pairs: set[tuple[str, str]] | None = None,
    llm_fail_times: int = 0,
) -> RunHarness:
    """Собирает стандартный харнесс из двух документов и двух вопросов.

    Args:
        no_text_layer_docs: Документы без текстового слоя.
        empty_docs: Документы с нулём чанков.
        empty_pairs: Пары пустого поиска.
        verifier_fail: Падение верификации.
        fail_pairs: Пары с исчерпанием LLM retry.
        llm_fail_times: Число отказов LLM.

    Returns:
        RunHarness.
    """
    config = AppConfig()
    documents = [
        DocumentRef(doc_id="doc_a.docx", path="/tmp/doc_a.docx", suffix=".docx"),
        DocumentRef(doc_id="doc_b.pdf", path="/tmp/doc_b.pdf", suffix=".pdf"),
    ]
    questions = QuestionSet(
        questions=[
            Question(
                id="Q001",
                category="Срок хранения",
                synonyms=["retention"],
            ),
            Question(
                id="Q002",
                category="Ответственный",
                synonyms=[],
            ),
        ]
    )
    trace = RecordingTraceSink()
    llm = FakeLlmGateway(fail_times=llm_fail_times)
    generator = FakeAnswerGenerator(
        llm=llm,
        fail_pairs=fail_pairs,
        retries=config.runtime.retries,
    )
    return RunHarness(
        config=config,
        documents=documents,
        questions=questions,
        store=InMemoryArtifactStore(),
        trace=trace,
        lease=GpuLease(trace),
        embedding=FakeEmbeddingGateway(),
        reranker=FakeRerankerGateway(),
        llm=llm,
        extractor=FakeExtractor(no_text_layer_docs=no_text_layer_docs),
        cleaner=FakeCleaner(),
        chunker=FakeChunker(empty_docs=empty_docs),
        retriever=FakeHybridRetriever(empty_pairs=empty_pairs),
        assembler=FakeContextAssembler(),
        verifier=FakeAnswerVerifier(fail=verifier_fail),
        generator=generator,
        result_sink=RecordingResultSink(),
    )
