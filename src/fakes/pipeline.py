"""Подделки портов пайплайна."""

from __future__ import annotations

from models.artifacts import (
    AnswerRecord,
    Citation,
    ContextBundle,
    ContextCandidate,
    RetrievalStats,
    TokenUsage,
)
from models.documents import (
    CandidateBatch,
    CandidateRecord,
    ChunkBatch,
    ChunkRecord,
    CleanedDocument,
    DocumentRef,
    ExtractedBlock,
    ExtractedDocument,
)
from models.questions import Question, QuestionSet
from models.summary import RunSummary
from ports.pipeline import (
    AnswerGenerator,
    AnswerVerifier,
    Chunker,
    Cleaner,
    ContextAssembler,
    Extractor,
    HybridRetriever,
    ResultSink,
)


class FakeExtractor(Extractor):
    """Extractor с настраиваемыми документами без текстового слоя.

    Attributes:
        no_text_layer_docs: doc_id без текстового слоя.
        calls: Список обработанных DocumentRef.
    """

    def __init__(self, no_text_layer_docs: set[str] | None = None) -> None:
        """Сохраняет набор проблемных документов.

        Args:
            no_text_layer_docs: Документы без текстового слоя.
        """
        self.no_text_layer_docs = no_text_layer_docs or set()
        self.calls: list[DocumentRef] = []

    def extract(self, document: DocumentRef) -> ExtractedDocument:
        """Возвращает текст или failure_reason=no_text_layer.

        Args:
            document: Ссылка на файл.

        Returns:
            ExtractedDocument.
        """
        self.calls.append(document)
        if document.doc_id in self.no_text_layer_docs:
            return ExtractedDocument(
                doc_id=document.doc_id,
                has_text_layer=False,
                failure_reason="no_text_layer",
                blocks=[],
            )
        return ExtractedDocument(
            doc_id=document.doc_id,
            has_text_layer=True,
            blocks=[
                ExtractedBlock(
                    kind="paragraph",
                    text=f"text of {document.doc_id}",
                    page=1,
                    paragraph_index=1,
                )
            ],
        )


class FakeCleaner(Cleaner):
    """Прозрачная очистка.

    Attributes:
        calls: Список очищенных документов.
    """

    def __init__(self) -> None:
        """Инициализирует журнал вызовов."""
        self.calls: list[ExtractedDocument] = []

    def clean(self, document: ExtractedDocument) -> CleanedDocument:
        """Копирует текст без изменений.

        Args:
            document: Извлечённый документ.

        Returns:
            CleanedDocument.
        """
        self.calls.append(document)
        return CleanedDocument(
            doc_id=document.doc_id,
            source_sha256=document.source_sha256,
            blocks=list(document.blocks),
        )


class FakeChunker(Chunker):
    """Chunker с настраиваемыми пустыми документами.

    Attributes:
        empty_docs: doc_id с нулём чанков.
        calls: Список чанкированных документов.
    """

    def __init__(self, empty_docs: set[str] | None = None) -> None:
        """Сохраняет набор пустых документов.

        Args:
            empty_docs: Документы без чанков.
        """
        self.empty_docs = empty_docs or set()
        self.calls: list[CleanedDocument] = []

    def chunk(self, document: CleanedDocument) -> ChunkBatch:
        """Строит один чанк или пустой пакет.

        Args:
            document: Очищенный документ.

        Returns:
            ChunkBatch.
        """
        self.calls.append(document)
        if document.doc_id in self.empty_docs:
            return ChunkBatch(doc_id=document.doc_id, chunks=[])
        chunk = ChunkRecord(
            chunk_id=f"{document.doc_id}:c0",
            text=document.joined_text(),
            page_from=1,
            page_to=1,
            section_path=["1"],
            chunk_index=0,
        )
        return ChunkBatch(doc_id=document.doc_id, chunks=[chunk])


class FakeHybridRetriever(HybridRetriever):
    """Retriever с настраиваемым пустым поиском.

    Attributes:
        empty_pairs: Пары (doc_id, question_id) с пустым результатом.
        calls: Журнал вызовов.
    """

    def __init__(
        self,
        empty_pairs: set[tuple[str, str]] | None = None,
    ) -> None:
        """Сохраняет пары пустого поиска.

        Args:
            empty_pairs: Пары документ × вопрос без кандидатов.
        """
        self.empty_pairs = empty_pairs or set()
        self.calls: list[tuple[str, str]] = []

    def search(
        self,
        doc_id: str,
        question: Question,
        chunks: ChunkBatch,
    ) -> CandidateBatch:
        """Возвращает кандидатов или пустой пакет.

        Args:
            doc_id: Идентификатор документа.
            question: Вопрос.
            chunks: Чанки.

        Returns:
            CandidateBatch.
        """
        self.calls.append((doc_id, question.id))
        if (doc_id, question.id) in self.empty_pairs or not chunks.chunks:
            return CandidateBatch(doc_id=doc_id, candidates=[])
        chunk = chunks.chunks[0]
        candidate = CandidateRecord(
            question_id=question.id,
            chunk_id=chunk.chunk_id,
            vector_score=0.8,
            bm25_score=0.5,
            fused_score=0.7,
            rank=1,
            text=chunk.text,
            section_path=chunk.section_path,
            pages=[chunk.page_from],
        )
        return CandidateBatch(doc_id=doc_id, candidates=[candidate])


class FakeContextAssembler(ContextAssembler):
    """Собирает ContextBundle из кандидатов.

    Attributes:
        calls: Журнал вызовов.
    """

    def __init__(self) -> None:
        """Инициализирует журнал."""
        self.calls: list[tuple[str, str, bool]] = []

    def assemble(
        self,
        doc_id: str,
        question: Question,
        candidates: CandidateBatch,
        empty_search: bool,
        chunks: ChunkBatch,
    ) -> ContextBundle:
        """Строит бандл с инвариантами empty_search.

        Args:
            doc_id: Идентификатор документа.
            question: Вопрос.
            candidates: Кандидаты.
            empty_search: Признак пустого поиска.
            chunks: Чанки документа; подделка их не расширяет.

        Returns:
            ContextBundle.
        """
        del chunks
        self.calls.append((doc_id, question.id, empty_search))
        if empty_search:
            return ContextBundle(
                doc_id=doc_id,
                question_id=question.id,
                category=question.category,
                used_synonyms=list(question.synonyms),
                empty_search=True,
                reason="no_relevant_chunks",
                candidates=[],
            )
        context_candidates = [
            ContextCandidate(
                rank_after_rerank=index + 1,
                rerank_score=item.rerank_score or item.fused_score,
                section_path=item.section_path,
                pages=item.pages,
                text=item.text,
                chunks_included=[item.chunk_id],
            )
            for index, item in enumerate(candidates.candidates)
        ]
        return ContextBundle(
            doc_id=doc_id,
            question_id=question.id,
            category=question.category,
            used_synonyms=list(question.synonyms),
            empty_search=False,
            candidates=context_candidates,
        )


class FakeAnswerVerifier(AnswerVerifier):
    """Верификатор с настраиваемым падением.

    Attributes:
        fail: Бросать ли исключение.
        calls: Журнал вызовов.
    """

    def __init__(self, fail: bool = False) -> None:
        """Сохраняет режим отказа.

        Args:
            fail: True — verify бросает исключение.
        """
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def verify(
        self,
        bundle: ContextBundle,
        question: Question,
    ) -> ContextBundle:
        """Возвращает бандл без изменений или падает.

        Args:
            bundle: Контекст.
            question: Вопрос.

        Returns:
            Тот же ContextBundle.

        Raises:
            RuntimeError: При fail=True.
        """
        self.calls.append((bundle.doc_id, question.id))
        if self.fail:
            raise RuntimeError("verification failed")
        return bundle


class FakeAnswerGenerator(AnswerGenerator):
    """Генератор ответов через LlmGateway или локально.

    Attributes:
        llm: Опциональный LLM-шлюз.
        fail_pairs: Пары, которые помечаются failed после retry.
        retries: Число попыток до failed.
        calls: Журнал вызовов с category.
        verify_skipped_pairs: Пары, для которых verify был пропущен.
    """

    def __init__(
        self,
        llm: object | None = None,
        fail_pairs: set[tuple[str, str]] | None = None,
        retries: int = 2,
    ) -> None:
        """Настраивает поведение генератора.

        Args:
            llm: LlmGateway-подделка.
            fail_pairs: Пары с исчерпанием retry.
            retries: Сколько раз звать LLM до failed.
        """
        self.llm = llm
        self.fail_pairs = fail_pairs or set()
        self.retries = retries
        self.calls: list[dict[str, object]] = []
        self.verify_skipped_pairs: set[tuple[str, str]] = set()
        self.embedding_model = ""
        self.reranker_model = ""
        self.answer_model = ""

    def _names(self) -> dict[str, str]:
        """Имена тройки, которые AnalysisRun выставляет перед фазой B.

        Returns:
            Поля строки ответа.
        """
        return {
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "answer_model": self.answer_model,
            "model": self.answer_model,
        }

    def generate(
        self,
        bundle: ContextBundle,
        question: Question,
        questions: QuestionSet | None = None,
    ) -> AnswerRecord:
        """Вызывает LLM с category и собирает AnswerRecord.

        Args:
            bundle: Контекст.
            question: Вопрос.
            questions: Не используется в подделке.

        Returns:
            AnswerRecord; status=failed при исчерпании retry.
        """
        del questions
        call_info = {
            "doc_id": bundle.doc_id,
            "question_id": question.id,
            "category": question.category,
            "empty_search": bundle.empty_search,
            "used_synonyms": list(bundle.used_synonyms),
        }
        self.calls.append(call_info)
        pair = (bundle.doc_id, question.id)
        if pair in self.fail_pairs:
            # Исчерпываем retry только для этой пары, не отравляя следующие.
            if self.llm is not None:
                from models.gateways import GenerateRequest

                attempts = self.retries + 1
                for _ in range(attempts):
                    try:
                        self.llm.infer(
                            GenerateRequest(
                                prompt=question.category,
                                doc_id=bundle.doc_id,
                                question_id=question.id,
                            )
                        )
                    except Exception:  # noqa: BLE001 — ожидаемые отказы retry
                        continue
            return AnswerRecord(
                **self._names(),
                doc_id=bundle.doc_id,
                question_id=question.id,
                category=question.category,
                used_synonyms=list(bundle.used_synonyms),
                found=False,
                answer="",
                reason="retry_exhausted",
                status="failed",
            )
        if self.llm is not None:
            from models.gateways import GenerateRequest

            last_error: Exception | None = None
            attempts = self.retries + 1
            for _ in range(attempts):
                try:
                    response = self.llm.infer(
                        GenerateRequest(
                            prompt=question.category,
                            doc_id=bundle.doc_id,
                            question_id=question.id,
                        )
                    )
                    return AnswerRecord(
                        **self._names(),
                        doc_id=bundle.doc_id,
                        question_id=question.id,
                        category=question.category,
                        used_synonyms=list(bundle.used_synonyms),
                        found=not bundle.empty_search,
                        answer=response.text,
                        reason=(
                            "no_relevant_chunks" if bundle.empty_search else None
                        ),
                        citations=self._citations(bundle),
                        retrieval=RetrievalStats(
                            after_rerank=len(bundle.candidates),
                        ),
                        tokens=TokenUsage(
                            prompt=response.prompt_tokens,
                            completion=response.completion_tokens,
                        ),
                        latency_ms=response.latency_ms,
                        status="ok",
                    )
                except Exception as error:  # noqa: BLE001 — подделка retry
                    last_error = error
            return AnswerRecord(
                **self._names(),
                doc_id=bundle.doc_id,
                question_id=question.id,
                category=question.category,
                used_synonyms=list(bundle.used_synonyms),
                found=False,
                answer="",
                reason=str(last_error) if last_error else "retry_exhausted",
                status="failed",
            )
        return AnswerRecord(
            **self._names(),
            doc_id=bundle.doc_id,
            question_id=question.id,
            category=question.category,
            used_synonyms=list(bundle.used_synonyms),
            found=not bundle.empty_search,
            answer=(
                "not found by search"
                if bundle.empty_search
                else f"answer for {question.category}"
            ),
            reason="no_relevant_chunks" if bundle.empty_search else None,
            citations=self._citations(bundle),
            retrieval=RetrievalStats(after_rerank=len(bundle.candidates)),
            status="ok",
        )

    def _citations(self, bundle: ContextBundle) -> list[Citation]:
        """Цитата-заглушка, чтобы found=true проходил инвариант записи.

        Args:
            bundle: Контекст пары.

        Returns:
            Пустой список при пустом поиске, иначе одна цитата.
        """
        if bundle.empty_search:
            return []
        if bundle.candidates:
            first = bundle.candidates[0]
            pages = first.pages
            page_from = pages[0] if pages else 1
            page_to = pages[-1] if pages else page_from
            chunk_ids = list(first.chunks_included) or ["c0"]
            section_path = first.section_path
        else:
            page_from = 1
            page_to = 1
            chunk_ids = ["c0"]
            section_path = None
        return [
            Citation(
                chunk_ids=chunk_ids,
                page_from=page_from,
                page_to=page_to,
                section_path=section_path,
                quote="цитата",
            )
        ]


class RecordingResultSink(ResultSink):
    """Пишет ответы и сводку в память.

    Attributes:
        answers: Записанные AnswerRecord.
        summaries: Записанные RunSummary.
    """

    def __init__(self) -> None:
        """Инициализирует списки."""
        self.answers: list[AnswerRecord] = []
        self.summaries: list[RunSummary] = []

    def write_answer(self, record: AnswerRecord) -> None:
        """Заменяет строку той же тройки и вопроса.

        Args:
            record: AnswerRecord.
        """
        self.answers = [
            item
            for item in self.answers
            if not (
                item.doc_id == record.doc_id
                and item.question_id == record.question_id
                and item.embedding_model == record.embedding_model
                and item.reranker_model == record.reranker_model
                and item.answer_model == record.answer_model
            )
        ]
        self.answers.append(record)

    def write_summary(self, summary: object) -> None:
        """Сохраняет сводку.

        Args:
            summary: RunSummary.
        """
        self.summaries.append(summary)  # type: ignore[arg-type]
