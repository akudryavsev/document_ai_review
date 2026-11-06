"""Фаза A1: индексация документов и эмбеддинги вопросов."""

from __future__ import annotations

from extract.debug_dump import BlockDocumentDebugDump
from models.artifacts import ArtifactMeta
from models.documents import CleanedDocument, ExtractedDocument
from models.gateways import EmbedItem, EmbedRequest
from store.codec import ArtifactCodec
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
from ports.gateways import EmbeddingGateway
from ports.pipeline import Chunker, Cleaner, Extractor
from ports.trace import TraceSink
from gate import ArtifactGate
from gpu import GpuLease


class IndexingPhase(Phase):
    """Индексация: Extract → Clean → Chunk → Embed → vectors.f16.

    Шлюз в хранилище не пишет. Фаза забирает байты float16 и сама вызывает
    save_chunks, save_chunk_vectors и кэш вопросов. Аренда эмбеддера нужна
    только когда хотя бы один текст не найден в embed_cache.

    Attributes:
        extractor: Extractor.
        cleaner: Cleaner.
        chunker: Chunker.
        embedding: EmbeddingGateway.
        store: ArtifactStore.
        gate: ArtifactGate.
        lease: GpuLease.
        trace_sink: TraceSink.
    """

    def __init__(
        self,
        extractor: Extractor,
        cleaner: Cleaner,
        chunker: Chunker,
        embedding: EmbeddingGateway,
        store: ArtifactStore,
        gate: ArtifactGate,
        lease: GpuLease,
        trace_sink: TraceSink,
    ) -> None:
        """Сохраняет порты фазы.

        Args:
            extractor: Извлечение текста.
            cleaner: Очистка.
            chunker: Чанкинг.
            embedding: Эмбеддер.
            store: Хранилище артефактов.
            gate: Решение reuse.
            lease: Аренда GPU.
            trace_sink: Наблюдаемость.
        """
        self.extractor = extractor
        self.cleaner = cleaner
        self.chunker = chunker
        self.embedding = embedding
        self.store = store
        self.gate = gate
        self.lease = lease
        self.trace_sink = trace_sink

    def plan(self, context: PhaseContext) -> PhasePlan:
        """Решает compute/skip по meta чанков.

        Args:
            context: Контекст фазы.

        Returns:
            PhasePlan.
        """
        items: list[PhasePlanItem] = []
        for document in context.documents:
            meta = self.store.load_chunks_meta(document.doc_id)
            decision = self.gate.decide(
                meta,
                schema_version=context.config.schema_version,
                force=context.force,
                chunking_hash=context.config.chunking_hash,
            )
            items.append(PhasePlanItem(document=document, decision=decision))
        return PhasePlan(phase="A1", items=items)

    @Observed(stage="indexing_execute")
    def execute(self, plan: PhasePlan, context: PhaseContext) -> PhaseReport:
        """Исполняет индексацию последовательно по документам.

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
        zero_chunk_docs: list[str] = []
        questions_need_embed = context.questions is not None and (
            context.force or not self.store.has_questions_cache()
        )
        if questions_need_embed and context.questions is not None:
            self._embed_questions(context)

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
            try:
                extracted = self.extractor.extract(item.document)
                self._dump_debug(context, extracted, "extract")
                if extracted.failure_reason or not extracted.has_text_layer:
                    failed += 1
                    outcomes.append(
                        DocumentPhaseOutcome(
                            doc_id=doc_id,
                            status="failed",
                            reason=extracted.failure_reason or "no_text_layer",
                        )
                    )
                    continue
                cleaned = self.cleaner.clean(extracted)
                self._dump_debug(context, cleaned, "clean")
                batch = self.chunker.chunk(cleaned)
                batch.chunking_hash = context.config.chunking_hash
                batch.embedding_model = context.config.embedding.model
                if not batch.chunks:
                    failed += 1
                    zero_chunk_docs.append(doc_id)
                    outcomes.append(
                        DocumentPhaseOutcome(
                            doc_id=doc_id,
                            status="failed",
                            reason="zero_chunks",
                        )
                    )
                    continue
                ordered = sorted(batch.chunks, key=lambda chunk: chunk.chunk_index)
                rows, dim, payload = self._embed_labeled(
                    context,
                    [(chunk.chunk_id, chunk.text) for chunk in ordered],
                )
                meta = ArtifactMeta(
                    schema_version=context.config.schema_version,
                    chunking_hash=context.config.chunking_hash,
                    embedding_model=context.config.embedding.model,
                    n_chunks=len(batch.chunks),
                    source_sha256=extracted.source_sha256 or None,
                    doc_id=doc_id,
                )
                self.store.save_chunks(batch, meta)
                self.store.save_chunk_vectors(doc_id, rows, dim, payload)
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
        report = PhaseReport(
            phase="A1",
            computed=computed,
            skipped=skipped,
            failed=failed,
            outcomes=outcomes,
        )
        if zero_chunk_docs:
            # AnalysisRun заберёт причину через outcomes.
            report.outcomes = outcomes
        return report

    def _dump_debug(
        self,
        context: PhaseContext,
        document: ExtractedDocument | CleanedDocument,
        stage: str,
    ) -> None:
        """Пишет отладочный Markdown, если extraction.debug_dump включён.

        Args:
            context: Контекст фазы с конфигом.
            document: Документ после extract или clean.
            stage: extract или clean.
        """
        if not context.config.extraction.debug_dump:
            return
        dump = BlockDocumentDebugDump(
            enabled=True,
            cache_dir=context.config.runtime.cache_dir,
        )
        dump.write(document, stage)

    def _embed_questions(self, context: PhaseContext) -> None:
        """Эмбеддит category и synonyms вопросов и кэширует.

        Args:
            context: Контекст с QuestionSet.
        """
        assert context.questions is not None
        labeled: list[tuple[str, str]] = []
        for question in context.questions.questions:
            labeled.append((f"{question.id}:category", question.category))
            for index, synonym in enumerate(question.synonyms):
                labeled.append((f"{question.id}:syn:{index}", synonym))
        rows, dim, payload = self._embed_labeled(context, labeled)
        self.store.save_questions_cache(
            context.questions,
            ArtifactMeta(
                schema_version=context.config.schema_version,
                embedding_model=context.config.embedding.model,
                n_chunks=len(context.questions.questions),
            ),
        )
        self.store.save_question_vectors(rows, dim, payload)

    def _embed_labeled(
        self,
        context: PhaseContext,
        labeled: list[tuple[str, str]],
    ) -> tuple[list[tuple[str, str]], int, bytes]:
        """Собирает байты float16: кэш, затем один infer на промахи.

        Args:
            context: Конфиг с именем пресета.
            labeled: Пары (item_id, текст) в порядке рядов на диске.

        Returns:
            Пары (item_id, text_sha256), общая dim и склейка векторов.

        Raises:
            ValueError: Пустой вход, битая длина или разная dim.
            LlamaServerError: Шлюз не смог посчитать промах.
        """
        codec = ArtifactCodec()
        model = context.config.embedding.model
        dtype = "float16"
        digests: list[str] = []
        vectors: list[bytes | None] = [None] * len(labeled)
        dim: int | None = None
        miss_item: dict[str, str] = {}
        miss_text: dict[str, str] = {}
        miss_order: list[str] = []
        aliases: dict[str, list[int]] = {}
        for index, (item_id, text) in enumerate(labeled):
            digest = codec.text_sha256(text)
            digests.append(digest)
            key = self.store.embedding_cache_key(model, dtype, digest)
            cached = self.store.get_cached_embedding(key)
            if cached is not None:
                dim = self._accept_vector(dim, cached, None)
                vectors[index] = cached
                continue
            aliases.setdefault(digest, []).append(index)
            if digest in miss_item:
                continue
            miss_item[digest] = item_id
            miss_text[digest] = text
            miss_order.append(digest)
        if miss_order:
            self._ensure_embedding()
            response = self.embedding.infer(
                EmbedRequest(
                    items=[
                        EmbedItem(item_id=miss_item[digest], text=miss_text[digest])
                        for digest in miss_order
                    ]
                )
            )
            if len(response.items) != len(miss_order):
                raise ValueError("эмбеддер вернул другой набор текстов")
            for slot, digest in enumerate(miss_order):
                item = response.items[slot]
                if item.item_id != miss_item[digest] or not item.success or not item.vector:
                    raise ValueError("эмбеддер не вернул вектор")
                dim = self._accept_vector(dim, item.vector, item.dim)
                key = self.store.embedding_cache_key(model, dtype, digest)
                self.store.put_cached_embedding(
                    key,
                    item.vector,
                    digest,
                    model,
                    dtype,
                    item.dim,
                )
                for alias in aliases[digest]:
                    vectors[alias] = item.vector
        if dim is None or any(vector is None for vector in vectors):
            raise ValueError("нет вектора для текста")
        payload = b"".join(vector for vector in vectors if vector is not None)
        rows = [
            (labeled[index][0], digests[index]) for index in range(len(labeled))
        ]
        return rows, dim, payload

    def _accept_vector(
        self,
        current: int | None,
        payload: bytes,
        declared: int | None,
    ) -> int:
        """Проверяет, что байты — один вектор той же dim.

        Args:
            current: Уже принятая dim или None.
            payload: Байты одного вектора.
            declared: dim из ответа шлюза. None для строки из кэша.

        Returns:
            Размерность этого вектора.

        Raises:
            ValueError: Длина не сходится или dim отличается от остальных.
        """
        row_dim = len(payload) // 2
        if row_dim < 1 or len(payload) != row_dim * 2:
            raise ValueError("длина вектора не кратна dim")
        if declared is not None and declared != row_dim:
            raise ValueError("dim ряда отличается")
        if current is not None and current != row_dim:
            raise ValueError("dim ряда отличается")
        return row_dim

    def _ensure_embedding(self) -> None:
        """Арендует эмбеддер, если он ещё не загружен."""
        if self.embedding.loaded:
            return
        self.lease.acquire(self.embedding)
