"""T-RET: гибридный поиск по утверждённой спеке 4."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fakes.trace import RecordingTraceSink
from models.config import HybridConfig, RetrievalConfig
from models.documents import ChunkBatch, ChunkRecord
from models.questions import Question
from retrieval.debug_dump import RetrievalDebugDump
from retrieval.engine import HybridSearcher
from retrieval.vectors import CosineSimilarity, DictEmbeddingIndex


class TestHybridRetriever:
    """Каталог T-RET-01…21."""

    def test_t_ret_01_category_only(self) -> None:
        """Один вариант category; чанк из обеих веток — одна строка."""
        searcher, question, chunks = self._ready(
            category="модем",
            synonyms=[],
            chunk_texts=["модем в сети", "другой текст"],
        )
        batch = searcher.search("doc", question, chunks)
        assert all(item.question_id == "Q1" for item in batch.candidates)
        ids = [item.chunk_id for item in batch.candidates]
        assert len(ids) == len(set(ids))
        assert len(batch.candidates) <= 2

    def test_t_ret_02_two_synonyms(self) -> None:
        """Ключи category и syn:0/1; строк не больше числа чанков."""
        vectors = DictEmbeddingIndex()
        question = Question(
            id="Q1",
            category="срок",
            synonyms=["период", "retention"],
        )
        chunks = self._chunks(["a", "b", "c"])
        self._put_unit(vectors, "Q1:category", [1.0, 0.0])
        self._put_unit(vectors, "Q1:syn:0", [0.0, 1.0])
        self._put_unit(vectors, "Q1:syn:1", [1.0, 1.0])
        for chunk in chunks.chunks:
            self._put_unit(vectors, chunk.chunk_id, [1.0, 0.0])
        searcher = HybridSearcher(RetrievalConfig(top_k=20), vectors)
        batch = searcher.search("doc", question, chunks)
        assert len(batch.candidates) <= 3

    def test_t_ret_03_union_one_row(self) -> None:
        """Чанк из обеих ветвей — одна запись; fused равен косинусу."""
        searcher, question, chunks = self._ready(
            category="модем",
            synonyms=[],
            chunk_texts=["модем"],
            top_k=5,
        )
        batch = searcher.search("doc", question, chunks)
        assert len(batch.candidates) == 1
        item = batch.candidates[0]
        assert item.fused_score == item.vector_score
        assert item.bm25_score > 0

    def test_t_ret_04_output_fields(self) -> None:
        """rerank_score None, rank плотный, text и path с чанка."""
        searcher, question, chunks = self._ready(
            category="модем",
            synonyms=[],
            chunk_texts=["модем один", "модем два"],
        )
        chunks.chunks[0].section_path = ["1", "1.1"]
        batch = searcher.search("doc", question, chunks)
        ranks = [item.rank for item in batch.candidates]
        assert ranks == list(range(1, len(ranks) + 1))
        assert all(item.rerank_score is None for item in batch.candidates)
        by_id = {item.chunk_id: item for item in batch.candidates}
        assert by_id["c0"].text == "модем один"
        assert by_id["c0"].section_path == ["1", "1.1"]

    def test_t_ret_05_pages(self) -> None:
        """pages при page_from=12, page_to=13 — [12, 13]."""
        vectors = DictEmbeddingIndex()
        question = Question(id="Q1", category="срок", synonyms=[])
        chunk = ChunkRecord(
            chunk_id="c0",
            text="срок хранения",
            page_from=12,
            page_to=13,
            chunk_index=0,
        )
        chunks = ChunkBatch(doc_id="doc", chunks=[chunk])
        self._put_unit(vectors, "Q1:category", [1.0, 0.0])
        self._put_unit(vectors, "c0", [1.0, 0.0])
        searcher = HybridSearcher(RetrievalConfig(), vectors)
        batch = searcher.search("doc", question, chunks)
        assert batch.candidates[0].pages == [12, 13]

    def test_t_ret_06_empty_chunks(self) -> None:
        """Пустые чанки — пустой пакет без исключения."""
        vectors = DictEmbeddingIndex()
        question = Question(id="Q1", category="срок", synonyms=[])
        self._put_unit(vectors, "Q1:category", [1.0, 0.0])
        searcher = HybridSearcher(RetrievalConfig(), vectors)
        batch = searcher.search("doc", question, ChunkBatch(doc_id="doc", chunks=[]))
        assert batch.candidates == []

    def test_t_ret_07_hybrid_disabled(self) -> None:
        """Без BM25: bm25_score=0; синонимы в векторном max."""
        vectors = DictEmbeddingIndex()
        question = Question(id="Q1", category="aaa", synonyms=["bbb"])
        chunks = self._chunks(["bbb слово"])
        self._put_unit(vectors, "Q1:category", [1.0, 0.0])
        self._put_unit(vectors, "Q1:syn:0", [0.0, 1.0])
        self._put_unit(vectors, "c0", [0.0, 1.0])
        config = RetrievalConfig(hybrid=HybridConfig(enabled=False))
        searcher = HybridSearcher(config, vectors)
        batch = searcher.search("doc", question, chunks)
        assert batch.candidates[0].bm25_score == 0.0
        assert batch.candidates[0].vector_score == pytest.approx(1.0)

    def test_t_ret_08_bad_strategy(self) -> None:
        """Чужой query_strategy — ValueError."""
        vectors = DictEmbeddingIndex()
        question = Question(id="Q1", category="срок", synonyms=[])
        chunks = self._chunks(["срок"])
        self._put_unit(vectors, "Q1:category", [1.0, 0.0])
        self._put_unit(vectors, "c0", [1.0, 0.0])
        searcher = HybridSearcher(
            RetrievalConfig(query_strategy="mean"),
            vectors,
        )
        with pytest.raises(ValueError):
            searcher.search("doc", question, chunks)

    def test_t_ret_09_deterministic(self) -> None:
        """Повторный search — те же chunk_id и порядок."""
        searcher, question, chunks = self._ready(
            category="модем",
            synonyms=["роутер"],
            chunk_texts=["модем", "роутер", "принтер"],
        )
        first = searcher.search("doc", question, chunks)
        second = searcher.search("doc", question, chunks)
        assert [item.chunk_id for item in first.candidates] == [
            item.chunk_id for item in second.candidates
        ]

    def test_t_ret_10_debug_off(self, tmp_path: Path) -> None:
        """debug_dump=false — файла нет."""
        dump = RetrievalDebugDump(enabled=False, cache_dir=str(tmp_path))
        searcher, question, chunks = self._ready(
            category="модем",
            synonyms=[],
            chunk_texts=["модем"],
            dump=dump,
        )
        searcher.search("doc", question, chunks)
        assert not (tmp_path / "debug" / "doc" / "Q1.retrieval.json").exists()

    def test_t_ret_11_debug_structure(self, tmp_path: Path) -> None:
        """Дамп: два варианта, три чанка, selected совпадает с выходом."""
        dump = RetrievalDebugDump(enabled=True, cache_dir=str(tmp_path))
        searcher, question, chunks = self._ready(
            category="модем",
            synonyms=["роутер"],
            chunk_texts=["модем", "роутер", "другое"],
            dump=dump,
        )
        batch = searcher.search("doc", question, chunks)
        path = tmp_path / "debug" / "doc" / "Q1.retrieval.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["variants"]) == 2
        assert len(payload["per_variant"]) == 2
        assert all(len(row["chunks"]) == 3 for row in payload["per_variant"])
        selected_ids = [row["chunk_id"] for row in payload["selected"]]
        assert selected_ids == [item.chunk_id for item in batch.candidates]

    def test_t_ret_12_debug_hybrid_off(self, tmp_path: Path) -> None:
        """hybrid off — bm25_top_k пуст."""
        dump = RetrievalDebugDump(enabled=True, cache_dir=str(tmp_path))
        vectors = DictEmbeddingIndex()
        question = Question(id="Q1", category="модем", synonyms=[])
        chunks = self._chunks(["модем"])
        self._put_unit(vectors, "Q1:category", [1.0, 0.0])
        self._put_unit(vectors, "c0", [1.0, 0.0])
        searcher = HybridSearcher(
            RetrievalConfig(hybrid=HybridConfig(enabled=False)),
            vectors,
            debug_dump=dump,
        )
        batch = searcher.search("doc", question, chunks)
        payload = json.loads(
            (tmp_path / "debug" / "doc" / "Q1.retrieval.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["after_max"]["bm25_top_k"] == []
        assert [row["chunk_id"] for row in payload["selected"]] == [
            item.chunk_id for item in batch.candidates
        ]

    def test_t_ret_15_threshold_zero(self) -> None:
        """score_threshold=0 не режет низкие косинусы."""
        vectors = DictEmbeddingIndex()
        question = Question(id="Q1", category="q", synonyms=[])
        chunks = self._chunks(["a", "b"])
        self._put_unit(vectors, "Q1:category", [1.0, 0.0])
        # Косинусы ~0.1 и ~0.2 — оба ниже 0.30.
        vectors.put("c0", [0.1, math.sqrt(1 - 0.01)])
        vectors.put("c1", [0.2, math.sqrt(1 - 0.04)])
        searcher = HybridSearcher(RetrievalConfig(score_threshold=0.0), vectors)
        batch = searcher.search("doc", question, chunks)
        assert {item.chunk_id for item in batch.candidates} == {"c0", "c1"}

    def test_t_ret_16_strict_threshold(self) -> None:
        """Косинус ровно 0.30 при пороге 0.30 не проходит в векторную ветвь."""
        vectors = DictEmbeddingIndex()
        question = Question(id="Q1", category="q", synonyms=[])
        # Текст без общих лемм с запросом — только векторная ветвь могла бы взять.
        chunks = self._chunks(["zzzz yyyy"])
        self._put_unit(vectors, "Q1:category", [1.0, 0.0])
        vectors.put("c0", [0.3, math.sqrt(1.0 - 0.09)])
        cosine = CosineSimilarity()
        assert cosine.score(
            vectors.get_vector("Q1:category"),  # type: ignore[arg-type]
            vectors.get_vector("c0"),  # type: ignore[arg-type]
        ) == pytest.approx(0.3)
        searcher = HybridSearcher(
            RetrievalConfig(
                score_threshold=0.3,
                hybrid=HybridConfig(enabled=False),
            ),
            vectors,
        )
        batch = searcher.search("doc", question, chunks)
        assert batch.candidates == []

    def test_t_ret_17_tie_at_top_k(self) -> None:
        """top_k=1 и два одинаковых лучших косинуса — оба в ветви."""
        vectors = DictEmbeddingIndex()
        question = Question(id="Q1", category="q", synonyms=[])
        chunks = self._chunks(["aaaa", "bbbb"])
        self._put_unit(vectors, "Q1:category", [1.0, 0.0])
        self._put_unit(vectors, "c0", [1.0, 0.0])
        self._put_unit(vectors, "c1", [1.0, 0.0])
        searcher = HybridSearcher(
            RetrievalConfig(
                top_k=1,
                score_threshold=0.0,
                hybrid=HybridConfig(enabled=False),
            ),
            vectors,
        )
        batch = searcher.search("doc", question, chunks)
        assert {item.chunk_id for item in batch.candidates} == {"c0", "c1"}

    def test_t_ret_18_cosine_not_dot(self) -> None:
        """vector_score — косинус, не голое скалярное произведение."""
        vectors = DictEmbeddingIndex()
        question = Question(id="Q1", category="q", synonyms=[])
        chunks = self._chunks(["текст"])
        vectors.put("Q1:category", [3.0, 0.0])
        vectors.put("c0", [6.0, 0.0])
        searcher = HybridSearcher(
            RetrievalConfig(hybrid=HybridConfig(enabled=False)),
            vectors,
        )
        batch = searcher.search("doc", question, chunks)
        assert batch.candidates[0].vector_score == pytest.approx(1.0)
        assert batch.candidates[0].vector_score != pytest.approx(18.0)

    def test_t_ret_19_lemma_modem(self) -> None:
        """Запрос «модем», в чанке «модема» — BM25 > 0."""
        searcher, question, chunks = self._ready(
            category="модем",
            synonyms=[],
            chunk_texts=["настройка модема"],
        )
        batch = searcher.search("doc", question, chunks)
        assert batch.candidates[0].bm25_score > 0

    def test_t_ret_20_missing_vector(self) -> None:
        """Нет вектора чанка — пустой пакет и failed в трассе."""
        trace = RecordingTraceSink()
        vectors = DictEmbeddingIndex()
        question = Question(id="Q1", category="модем", synonyms=[])
        chunks = self._chunks(["модем", "другое"])
        self._put_unit(vectors, "Q1:category", [1.0, 0.0])
        self._put_unit(vectors, "c0", [1.0, 0.0])
        # c1 без вектора.
        searcher = HybridSearcher(RetrievalConfig(), vectors, trace_sink=trace)
        batch = searcher.search("doc", question, chunks)
        assert batch.candidates == []
        failed = [event for event in trace.events if event.status == "failed"]
        assert failed
        assert "c1" in (failed[0].error or "")

    def test_t_ret_21_shared_modem_lemma(self) -> None:
        """«установка модема» и «устанавливается модем» — общий терм."""
        searcher, question, chunks = self._ready(
            category="установка модема",
            synonyms=[],
            chunk_texts=["устанавливается модем на объекте"],
        )
        batch = searcher.search("doc", question, chunks)
        assert batch.candidates[0].bm25_score > 0

    def _ready(
        self,
        category: str,
        synonyms: list[str],
        chunk_texts: list[str],
        top_k: int = 20,
        dump: RetrievalDebugDump | None = None,
    ) -> tuple[HybridSearcher, Question, ChunkBatch]:
        """Собирает searcher с единичными векторами.

        Args:
            category: Текст category.
            synonyms: Синонимы.
            chunk_texts: Тексты чанков.
            top_k: Лимит ветви.
            dump: Дамп или None.

        Returns:
            searcher, question, chunks.
        """
        vectors = DictEmbeddingIndex()
        question = Question(id="Q1", category=category, synonyms=synonyms)
        chunks = self._chunks(chunk_texts)
        self._put_unit(vectors, "Q1:category", [1.0, 0.0])
        for index, _synonym in enumerate(synonyms):
            angle = (index + 1) * 0.2
            self._put_unit(
                vectors,
                f"Q1:syn:{index}",
                [math.cos(angle), math.sin(angle)],
            )
        for chunk in chunks.chunks:
            self._put_unit(vectors, chunk.chunk_id, [1.0, 0.0])
        config = RetrievalConfig(top_k=top_k, score_threshold=0.0)
        searcher = HybridSearcher(config, vectors, debug_dump=dump)
        return searcher, question, chunks

    def _chunks(self, texts: list[str]) -> ChunkBatch:
        """Строит ChunkBatch с id c0, c1, …

        Args:
            texts: Тексты.

        Returns:
            ChunkBatch.
        """
        records = [
            ChunkRecord(
                chunk_id=f"c{index}",
                text=text,
                chunk_index=index,
            )
            for index, text in enumerate(texts)
        ]
        return ChunkBatch(doc_id="doc", chunks=records)

    def _put_unit(
        self,
        index: DictEmbeddingIndex,
        item_id: str,
        values: list[float],
    ) -> None:
        """Кладёт вектор как есть.

        Args:
            index: Индекс.
            item_id: Ключ.
            values: Компоненты.
        """
        index.put(item_id, values)
