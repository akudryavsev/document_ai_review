"""Реализация HybridRetriever: вектор + BM25, max по вариантам."""

from __future__ import annotations

from collections.abc import Sequence

from models.config import RetrievalConfig
from models.documents import CandidateBatch, CandidateRecord, ChunkBatch, ChunkRecord
from models.observe import TraceEvent
from models.questions import Question
from ports.pipeline import HybridRetriever
from ports.trace import TraceSink
from retrieval.bm25 import Bm25Index, Bm25Tokenizer
from retrieval.debug_dump import RetrievalDebugDump
from retrieval.vectors import CosineSimilarity, EmbeddingIndex


class HybridSearcher(HybridRetriever):
    """Гибридный поиск по утверждённой спеке.

    Attributes:
        config: Параметры retrieval.
        embeddings: Векторы по item_id.
        debug_dump: Опциональный писатель JSON.
        trace_sink: Опциональная трасса для отсутствия вектора.
        tokenizer: Лемматизатор BM25.
        cosine: Счётчик косинуса.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        embeddings: EmbeddingIndex,
        debug_dump: RetrievalDebugDump | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        """Сохраняет зависимости и создаёт тяжёлые помощники.

        Args:
            config: RetrievalConfig.
            embeddings: Индекс векторов.
            debug_dump: Дамп или None.
            trace_sink: TraceSink или None.
        """
        self.config = config
        self.embeddings = embeddings
        self.debug_dump = debug_dump
        self.trace_sink = trace_sink
        self.tokenizer = Bm25Tokenizer()
        self.cosine = CosineSimilarity()

    def search(
        self,
        doc_id: str,
        question: Question,
        chunks: ChunkBatch,
    ) -> CandidateBatch:
        """Ищет кандидатов до реранка.

        Args:
            doc_id: Идентификатор документа.
            question: Вопрос с synonyms.
            chunks: Чанки документа.

        Returns:
            CandidateBatch до реранка.

        Raises:
            ValueError: query_strategy не max_over_variants.
        """
        if self.config.query_strategy != "max_over_variants":
            raise ValueError(
                f"Неподдерживаемая query_strategy: {self.config.query_strategy}"
            )
        if not chunks.chunks:
            empty = CandidateBatch(doc_id=doc_id, candidates=[])
            self._write_search_dump(
                doc_id=doc_id,
                question=question,
                variants=self._variants(question),
                per_variant=[],
                after_max={"vector_top_k": [], "bm25_top_k": []},
                fused_before_threshold=[],
                selected=[],
            )
            return empty

        variants = self._variants(question)
        missing_id = self._first_missing_vector(variants, chunks.chunks)
        if missing_id is not None:
            self._emit_missing_vector(doc_id, question.id, missing_id)
            empty = CandidateBatch(doc_id=doc_id, candidates=[])
            self._write_search_dump(
                doc_id=doc_id,
                question=question,
                variants=variants,
                per_variant=[],
                after_max={"vector_top_k": [], "bm25_top_k": []},
                fused_before_threshold=[],
                selected=[],
            )
            return empty

        hybrid_enabled = self.config.hybrid.enabled
        bm25_index: Bm25Index | None = None
        if hybrid_enabled:
            texts = [chunk.text for chunk in chunks.chunks]
            bm25_index = Bm25Index(texts, self.tokenizer)

        # Скоры по каждому варианту и каждому чанку.
        per_variant_rows: list[dict[str, object]] = []
        vector_max: dict[str, float] = {
            chunk.chunk_id: float("-inf") for chunk in chunks.chunks
        }
        bm25_max: dict[str, float] = {
            chunk.chunk_id: float("-inf") for chunk in chunks.chunks
        }

        for variant in variants:
            query_vector = self.embeddings.get_vector(variant["item_id"])
            assert query_vector is not None
            bm25_scores = (
                bm25_index.score_query(variant["text"])
                if bm25_index is not None
                else [0.0] * len(chunks.chunks)
            )
            chunk_rows: list[dict[str, float | str]] = []
            for chunk_index, chunk in enumerate(chunks.chunks):
                chunk_vector = self.embeddings.get_vector(chunk.chunk_id)
                assert chunk_vector is not None
                vector_score = self.cosine.score(query_vector, chunk_vector)
                bm25_score = float(bm25_scores[chunk_index])
                chunk_rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "vector_score": vector_score,
                        "bm25_score": bm25_score,
                    }
                )
                previous_vector = vector_max[chunk.chunk_id]
                if vector_score > previous_vector:
                    vector_max[chunk.chunk_id] = vector_score
                previous_bm25 = bm25_max[chunk.chunk_id]
                if bm25_score > previous_bm25:
                    bm25_max[chunk.chunk_id] = bm25_score
            per_variant_rows.append(
                {
                    "variant_index": variant["variant_index"],
                    "chunks": chunk_rows,
                }
            )

        # После max: порог только на векторной ветви.
        threshold = self.config.score_threshold
        vector_candidates: list[tuple[str, float]] = []
        for chunk in chunks.chunks:
            score = vector_max[chunk.chunk_id]
            if threshold > 0 and not (score > threshold):
                continue
            vector_candidates.append((chunk.chunk_id, score))
        vector_top = self._top_with_ties(vector_candidates, self.config.top_k)

        bm25_top: list[tuple[str, float]] = []
        if hybrid_enabled:
            bm25_candidates = [
                (chunk.chunk_id, bm25_max[chunk.chunk_id]) for chunk in chunks.chunks
            ]
            bm25_top = self._top_with_ties(bm25_candidates, self.config.top_k)

        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks.chunks}
        selected_ids = {chunk_id for chunk_id, _ in vector_top}
        selected_ids.update(chunk_id for chunk_id, _ in bm25_top)

        vector_rank = {
            chunk_id: rank for rank, (chunk_id, _) in enumerate(vector_top, start=1)
        }
        bm25_rank = {
            chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_top, start=1)
        }

        fused_before: list[dict[str, float | str]] = []
        records: list[CandidateRecord] = []
        for chunk_id in selected_ids:
            chunk = chunk_by_id[chunk_id]
            vector_score = vector_max[chunk_id]
            if vector_score == float("-inf"):
                vector_score = 0.0
            bm25_score = bm25_max[chunk_id]
            if not hybrid_enabled or bm25_score == float("-inf"):
                bm25_score = 0.0
            fused_before.append(
                {
                    "chunk_id": chunk_id,
                    "vector_score": vector_score,
                    "bm25_score": bm25_score,
                }
            )
            records.append(
                CandidateRecord(
                    question_id=question.id,
                    chunk_id=chunk_id,
                    vector_score=vector_score,
                    bm25_score=bm25_score,
                    fused_score=vector_score,
                    rank=0,
                    rerank_score=None,
                    text=chunk.text,
                    section_path=chunk.section_path,
                    pages=self._pages(chunk),
                )
            )

        records.sort(
            key=lambda record: self._rank_sort_key(
                record,
                chunk_by_id,
                vector_rank,
                bm25_rank,
            )
        )
        for index, record in enumerate(records, start=1):
            record.rank = index

        self._write_search_dump(
            doc_id=doc_id,
            question=question,
            variants=variants,
            per_variant=per_variant_rows,
            after_max={
                "vector_top_k": [
                    {"chunk_id": chunk_id, "score": score}
                    for chunk_id, score in vector_top
                ],
                "bm25_top_k": [
                    {"chunk_id": chunk_id, "score": score}
                    for chunk_id, score in bm25_top
                ],
            },
            fused_before_threshold=fused_before,
            selected=[record.model_dump(mode="json") for record in records],
        )
        return CandidateBatch(doc_id=doc_id, candidates=records)

    def _variants(self, question: Question) -> list[dict[str, object]]:
        """Строит список вариантов category и synonyms.

        Args:
            question: Вопрос.

        Returns:
            Словари variant_index, item_id, text.
        """
        variants: list[dict[str, object]] = [
            {
                "variant_index": 0,
                "item_id": f"{question.id}:category",
                "text": question.category,
            }
        ]
        for index, synonym in enumerate(question.synonyms):
            variants.append(
                {
                    "variant_index": index + 1,
                    "item_id": f"{question.id}:syn:{index}",
                    "text": synonym,
                }
            )
        return variants

    def _first_missing_vector(
        self,
        variants: list[dict[str, object]],
        chunks: list[ChunkRecord],
    ) -> str | None:
        """Находит первый отсутствующий или нулевой вектор.

        Args:
            variants: Варианты запроса.
            chunks: Чанки.

        Returns:
            item_id или None.
        """
        for variant in variants:
            item_id = str(variant["item_id"])
            vector = self.embeddings.get_vector(item_id)
            if vector is None or self._is_zero(vector):
                return item_id
        for chunk in chunks:
            vector = self.embeddings.get_vector(chunk.chunk_id)
            if vector is None or self._is_zero(vector):
                return chunk.chunk_id
        return None

    def _is_zero(self, vector: Sequence[float]) -> bool:
        """Проверяет нулевую длину.

        Args:
            vector: Компоненты.

        Returns:
            True, если все нули или пусто.
        """
        if len(vector) == 0:
            return True
        return all(value == 0.0 for value in vector)

    def _emit_missing_vector(
        self,
        doc_id: str,
        question_id: str,
        item_id: str,
    ) -> None:
        """Пишет failed в трассу без исключения.

        Args:
            doc_id: Документ.
            question_id: Вопрос.
            item_id: Отсутствующий ключ.
        """
        if self.trace_sink is None:
            return
        self.trace_sink.emit(
            TraceEvent(
                stage="hybrid_search",
                status="failed",
                doc_id=doc_id,
                question_id=question_id,
                error=f"missing vector item_id={item_id}",
            )
        )

    def _top_with_ties(
        self,
        scored: list[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Отбирает top_k и всех с тем же скором на границе.

        Args:
            scored: Пары (chunk_id, score).
            top_k: Лимит без учёта ничьей.

        Returns:
            Список по убыванию score, затем chunk_id.
        """
        if not scored or top_k <= 0:
            return []
        ordered = sorted(scored, key=lambda item: (-item[1], item[0]))
        if len(ordered) <= top_k:
            return ordered
        boundary_score = ordered[top_k - 1][1]
        selected: list[tuple[str, float]] = []
        for chunk_id, score in ordered:
            if len(selected) < top_k or score == boundary_score:
                selected.append((chunk_id, score))
                continue
            break
        return selected

    def _rank_sort_key(
        self,
        record: CandidateRecord,
        chunk_by_id: dict[str, ChunkRecord],
        vector_rank: dict[str, int],
        bm25_rank: dict[str, int],
    ) -> tuple[int, int, str]:
        """Ключ порядка итогового пула.

        Args:
            record: Кандидат.
            chunk_by_id: Чанки по id.
            vector_rank: Ранги векторной ветви.
            bm25_rank: Ранги BM25.

        Returns:
            Кортеж для сортировки по возрастанию.
        """
        ranks: list[int] = []
        if record.chunk_id in vector_rank:
            ranks.append(vector_rank[record.chunk_id])
        if record.chunk_id in bm25_rank:
            ranks.append(bm25_rank[record.chunk_id])
        best_rank = min(ranks) if ranks else 10**9
        chunk = chunk_by_id[record.chunk_id]
        return (best_rank, chunk.chunk_index, record.chunk_id)

    def _pages(self, chunk: ChunkRecord) -> list[int]:
        """Разворачивает диапазон страниц.

        Args:
            chunk: Чанк.

        Returns:
            Список номеров или пустой список при нулях.
        """
        if chunk.page_from == 0 and chunk.page_to == 0:
            return []
        return list(range(chunk.page_from, chunk.page_to + 1))

    def _write_search_dump(
        self,
        doc_id: str,
        question: Question,
        variants: list[dict[str, object]],
        per_variant: list[dict[str, object]],
        after_max: dict[str, object],
        fused_before_threshold: list[dict[str, float | str]],
        selected: list[dict[str, object]],
    ) -> None:
        """Пишет дамп поиска, если включён.

        Args:
            doc_id: Документ.
            question: Вопрос.
            variants: Варианты.
            per_variant: Скоры по вариантам.
            after_max: Top ветвей.
            fused_before_threshold: Объединение.
            selected: Итог search.
        """
        if self.debug_dump is None:
            return
        self.debug_dump.write_search(
            {
                "doc_id": doc_id,
                "question_id": question.id,
                "variants": variants,
                "per_variant": per_variant,
                "after_max": after_max,
                "fused_before_threshold": fused_before_threshold,
                "selected": selected,
            }
        )
