"""T-RNK и persist candidates по утверждённой спеке 4."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakes.gateways import FakeRerankerGateway
from fakes.pipeline import FakeContextAssembler, FakeHybridRetriever
from fakes.trace import RecordingTraceSink
from gate import ArtifactGate
from gpu import GpuLease
from models.artifacts import ArtifactMeta
from models.config import AppConfig, RerankerConfig
from models.documents import CandidateBatch, CandidateRecord, ChunkBatch, ChunkRecord
from models.phases import PhaseContext, PhasePlan, PhasePlanItem, ReuseDecision
from models.questions import Question, QuestionSet
from models.documents import DocumentRef
from phases.retrieval import RetrievalPhase
from retrieval.debug_dump import RetrievalDebugDump
from store.file_store import FileArtifactStore
from store.memory_store import InMemoryArtifactStore


class TestRetrievalRerank:
    """Каталог T-RNK и T-RET-13/14, T-RNK-12/13."""

    def test_t_ret_13_persist_before_rerank(self, tmp_path: Path) -> None:
        """candidates.jsonl — выходы search без rerank_score."""
        store = FileArtifactStore(
            cache_dir=tmp_path / "cache",
            index_dir=tmp_path / "index",
            contexts_dir=tmp_path / "contexts",
        )
        chunks = ChunkBatch(
            doc_id="doc",
            chunks=[
                ChunkRecord(chunk_id="c0", text="текст", chunk_index=0),
            ],
        )
        store.save_chunks(
            chunks,
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        config.retrieval.persist_candidates = True
        config.reranker.enabled = True
        phase, reranker = self._phase(store, config)
        reranker.load()
        context = self._context(
            config,
            questions=[
                Question(id="Q1", category="a", synonyms=[]),
                Question(id="Q2", category="b", synonyms=[]),
            ],
        )
        plan = self._compute_plan("doc")
        phase.execute(plan, context)
        path = tmp_path / "index" / "doc" / "candidates.jsonl"
        assert path.exists()
        lines = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        question_ids = {row["question_id"] for row in lines}
        assert question_ids == {"Q1", "Q2"}
        assert all(row.get("rerank_score") is None for row in lines)

    def test_t_ret_14_persist_off(self, tmp_path: Path) -> None:
        """persist_candidates=false — файла нет, прогон идёт."""
        store = FileArtifactStore(
            cache_dir=tmp_path / "cache",
            index_dir=tmp_path / "index",
            contexts_dir=tmp_path / "contexts",
        )
        chunks = ChunkBatch(
            doc_id="doc",
            chunks=[ChunkRecord(chunk_id="c0", text="текст", chunk_index=0)],
        )
        store.save_chunks(
            chunks,
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        config.retrieval.persist_candidates = False
        phase, reranker = self._phase(store, config)
        reranker.load()
        context = self._context(
            config,
            questions=[Question(id="Q1", category="a", synonyms=[])],
        )
        report = phase.execute(self._compute_plan("doc"), context)
        assert report.computed == 1
        assert not (tmp_path / "index" / "doc" / "candidates.jsonl").exists()

    def test_t_rnk_01_empty_search(self) -> None:
        """Пустой search — infer не вызывается, empty_search=true."""
        store = InMemoryArtifactStore()
        store.save_chunks(
            ChunkBatch(doc_id="doc", chunks=[]),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        retriever = FakeHybridRetriever(empty_pairs={("doc", "Q1")})
        assembler = FakeContextAssembler()
        phase, reranker = self._phase(store, config, retriever=retriever, assembler=assembler)
        # Пустые чанки: FakeHybridRetriever тоже вернёт пусто.
        store.save_chunks(
            ChunkBatch(
                doc_id="doc",
                chunks=[ChunkRecord(chunk_id="c0", text="x", chunk_index=0)],
            ),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        context = self._context(
            config,
            questions=[Question(id="Q1", category="a", synonyms=[])],
        )
        phase.execute(self._compute_plan("doc"), context)
        assert reranker.infer_calls == []
        assert assembler.calls[0][2] is True

    def test_t_rnk_02_four_pairs(self) -> None:
        """synonyms_rerank: два варианта × два чанка — четыре пары."""
        store = InMemoryArtifactStore()
        chunks = ChunkBatch(
            doc_id="doc",
            chunks=[
                ChunkRecord(chunk_id="c0", text="t0", chunk_index=0),
                ChunkRecord(chunk_id="c1", text="t1", chunk_index=1),
            ],
        )
        store.save_chunks(chunks, ArtifactMeta(schema_version="1.0", chunking_hash="ch"))
        config = AppConfig()
        config.reranker.query_strategy = "synonyms_rerank"
        phase, reranker = self._phase(
            store,
            config,
            retriever=self._fixed_retriever(
                [
                    self._candidate("Q1", "c0", "t0", rank=1),
                    self._candidate("Q1", "c1", "t1", rank=2),
                ]
            ),
        )
        reranker.load()
        context = self._context(
            config,
            questions=[
                Question(id="Q1", category="cat", synonyms=["syn"]),
            ],
        )
        phase.execute(self._compute_plan("doc"), context)
        assert len(reranker.infer_calls) == 1
        pair_ids = [pair.pair_id for pair in reranker.infer_calls[0].pairs]
        assert pair_ids == ["c0#0", "c0#1", "c1#0", "c1#1"]
        assert all(
            pair.query in {"cat", "syn"} for pair in reranker.infer_calls[0].pairs
        )

    def test_t_rnk_03_max_over_pairs(self) -> None:
        """У чанка два скора — в отбор идёт максимум."""
        store = InMemoryArtifactStore()
        store.save_chunks(
            ChunkBatch(
                doc_id="doc",
                chunks=[ChunkRecord(chunk_id="c0", text="t0", chunk_index=0)],
            ),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        config.reranker.query_strategy = "synonyms_rerank"
        config.reranker.top_n = 5
        config.reranker.score_threshold = 0.0
        config.context.early_exit_threshold = 0.0
        reranker = FakeRerankerGateway(
            scores_by_pair={"c0#0": 0.2, "c0#1": 0.8},
        )
        phase, _ = self._phase(
            store,
            config,
            retriever=self._fixed_retriever(
                [self._candidate("Q1", "c0", "t0", rank=1)]
            ),
            reranker=reranker,
        )
        reranker.load()
        assembler = phase.assembler
        assert isinstance(assembler, FakeContextAssembler)
        context = self._context(
            config,
            questions=[Question(id="Q1", category="cat", synonyms=["syn"])],
        )
        phase.execute(self._compute_plan("doc"), context)
        bundle = store.load_context("doc", "Q1")
        assert bundle is not None
        assert bundle.candidates[0].chunks_included == ["c0"]
        assert bundle.candidates[0].rerank_score == pytest.approx(0.8)

    def test_t_rnk_04_early_exit(self) -> None:
        """Лучший скор строго меньше early_exit — пустой пакет."""
        store = InMemoryArtifactStore()
        store.save_chunks(
            ChunkBatch(
                doc_id="doc",
                chunks=[
                    ChunkRecord(chunk_id="c0", text="t0", chunk_index=0),
                    ChunkRecord(chunk_id="c1", text="t1", chunk_index=1),
                ],
            ),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        config.context.early_exit_threshold = 0.5
        config.reranker.score_threshold = 0.1
        reranker = FakeRerankerGateway(default_score=0.4)
        phase, _ = self._phase(
            store,
            config,
            retriever=self._fixed_retriever(
                [
                    self._candidate("Q1", "c0", "t0", rank=1),
                    self._candidate("Q1", "c1", "t1", rank=2),
                ]
            ),
            reranker=reranker,
        )
        reranker.load()
        assembler = FakeContextAssembler()
        phase.assembler = assembler
        context = self._context(
            config,
            questions=[Question(id="Q1", category="cat", synonyms=[])],
        )
        phase.execute(self._compute_plan("doc"), context)
        assert assembler.calls[0][2] is True

    def test_t_rnk_05_missing_pair_score(self) -> None:
        """Нет скора пары — чанк не проходит."""
        store = InMemoryArtifactStore()
        store.save_chunks(
            ChunkBatch(
                doc_id="doc",
                chunks=[
                    ChunkRecord(chunk_id="c0", text="t0", chunk_index=0),
                    ChunkRecord(chunk_id="c1", text="t1", chunk_index=1),
                ],
            ),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        config.context.early_exit_threshold = 0.0
        config.reranker.score_threshold = 0.0
        reranker = FakeRerankerGateway(
            scores_by_pair={"c1#0": 0.9},
            omit_pair_ids={"c0#0"},
        )
        phase, _ = self._phase(
            store,
            config,
            retriever=self._fixed_retriever(
                [
                    self._candidate("Q1", "c0", "t0", rank=1),
                    self._candidate("Q1", "c1", "t1", rank=2),
                ]
            ),
            reranker=reranker,
        )
        reranker.load()
        context = self._context(
            config,
            questions=[Question(id="Q1", category="cat", synonyms=[])],
        )
        phase.execute(self._compute_plan("doc"), context)
        bundle = store.load_context("doc", "Q1")
        assert bundle is not None
        assert [item.chunks_included[0] for item in bundle.candidates] == ["c1"]

    def test_t_rnk_06_preserves_search_fields(self) -> None:
        """После отбора поля поиска те же, rerank_score заполнен."""
        store = InMemoryArtifactStore()
        store.save_chunks(
            ChunkBatch(
                doc_id="doc",
                chunks=[ChunkRecord(chunk_id="c0", text="hello", chunk_index=0)],
            ),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        candidate = self._candidate(
            "Q1",
            "c0",
            "hello",
            rank=1,
            vector_score=0.55,
            bm25_score=1.2,
            fused_score=0.55,
            section_path=["A"],
            pages=[3],
        )
        phase, reranker = self._phase(
            store,
            config,
            retriever=self._fixed_retriever([candidate]),
        )
        reranker.load()
        context = self._context(
            config,
            questions=[Question(id="Q1", category="cat", synonyms=[])],
        )
        phase.execute(self._compute_plan("doc"), context)
        bundle = store.load_context("doc", "Q1")
        assert bundle is not None
        item = bundle.candidates[0]
        assert item.text == "hello"
        assert item.section_path == ["A"]
        assert item.pages == [3]
        assert item.rerank_score is not None

    def test_t_rnk_07_infer_exception(self) -> None:
        """Исключение infer не превращается в пустой бандл внутри поиска."""
        store = InMemoryArtifactStore()
        store.save_chunks(
            ChunkBatch(
                doc_id="doc",
                chunks=[ChunkRecord(chunk_id="c0", text="t", chunk_index=0)],
            ),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        reranker = FakeRerankerGateway(raise_error=RuntimeError("boom"))
        phase, _ = self._phase(
            store,
            config,
            retriever=self._fixed_retriever(
                [self._candidate("Q1", "c0", "t", rank=1)]
            ),
            reranker=reranker,
        )
        reranker.load()
        context = self._context(
            config,
            questions=[Question(id="Q1", category="cat", synonyms=[])],
        )
        report = phase.execute(self._compute_plan("doc"), context)
        assert report.failed == 1
        assert store.load_context("doc", "Q1") is None

    def test_t_rnk_08_debug_rerank(self, tmp_path: Path) -> None:
        """Дамп содержит четыре пары и selected_after."""
        dump = RetrievalDebugDump(enabled=True, cache_dir=str(tmp_path))
        store = InMemoryArtifactStore()
        store.save_chunks(
            ChunkBatch(
                doc_id="doc",
                chunks=[
                    ChunkRecord(chunk_id="c0", text="t0", chunk_index=0),
                    ChunkRecord(chunk_id="c1", text="t1", chunk_index=1),
                ],
            ),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        config.reranker.query_strategy = "synonyms_rerank"
        phase, reranker = self._phase(
            store,
            config,
            retriever=self._fixed_retriever(
                [
                    self._candidate("Q1", "c0", "t0", rank=1),
                    self._candidate("Q1", "c1", "t1", rank=2),
                ]
            ),
            dump=dump,
        )
        reranker.load()
        # Имитация dump после search.
        dump.write_search(
            {
                "doc_id": "doc",
                "question_id": "Q1",
                "variants": [],
                "per_variant": [],
                "after_max": {"vector_top_k": [], "bm25_top_k": []},
                "fused_before_threshold": [],
                "selected": [],
            }
        )
        context = self._context(
            config,
            questions=[Question(id="Q1", category="cat", synonyms=["syn"])],
        )
        phase.execute(self._compute_plan("doc"), context)
        payload = json.loads(
            (tmp_path / "debug" / "doc" / "Q1.retrieval.json").read_text(
                encoding="utf-8"
            )
        )
        assert len(payload["rerank"]["pairs"]) == 4
        selected_ids = [
            row["chunk_id"] for row in payload["rerank"]["selected_after"]
        ]
        bundle = store.load_context("doc", "Q1")
        assert bundle is not None
        assert selected_ids == [
            item.chunks_included[0] for item in bundle.candidates
        ]

    def test_t_rnk_09_zero_thresholds_keep_top_n(self) -> None:
        """Нулевые пороги — top_n лучших даже при скоре < 0.4."""
        store = InMemoryArtifactStore()
        store.save_chunks(
            ChunkBatch(
                doc_id="doc",
                chunks=[
                    ChunkRecord(chunk_id="c0", text="t0", chunk_index=0),
                    ChunkRecord(chunk_id="c1", text="t1", chunk_index=1),
                    ChunkRecord(chunk_id="c2", text="t2", chunk_index=2),
                ],
            ),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        config.reranker.top_n = 2
        config.reranker.score_threshold = 0.0
        config.context.early_exit_threshold = 0.0
        reranker = FakeRerankerGateway(
            scores_by_pair={"c0#0": 0.35, "c1#0": 0.25, "c2#0": 0.15},
        )
        phase, _ = self._phase(
            store,
            config,
            retriever=self._fixed_retriever(
                [
                    self._candidate("Q1", "c0", "t0", rank=1),
                    self._candidate("Q1", "c1", "t1", rank=2),
                    self._candidate("Q1", "c2", "t2", rank=3),
                ]
            ),
            reranker=reranker,
        )
        reranker.load()
        assembler = FakeContextAssembler()
        phase.assembler = assembler
        context = self._context(
            config,
            questions=[Question(id="Q1", category="cat", synonyms=[])],
        )
        phase.execute(self._compute_plan("doc"), context)
        assert assembler.calls[0][2] is False
        bundle = store.load_context("doc", "Q1")
        assert bundle is not None
        assert len(bundle.candidates) == 2

    def test_t_rnk_10_equal_threshold_rejected(self) -> None:
        """Скор равный порогу не проходит."""
        store = InMemoryArtifactStore()
        store.save_chunks(
            ChunkBatch(
                doc_id="doc",
                chunks=[
                    ChunkRecord(chunk_id="c0", text="t0", chunk_index=0),
                    ChunkRecord(chunk_id="c1", text="t1", chunk_index=1),
                ],
            ),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        config.reranker.score_threshold = 0.5
        config.context.early_exit_threshold = 0.0
        reranker = FakeRerankerGateway(
            scores_by_pair={"c0#0": 0.5, "c1#0": 0.9},
        )
        phase, _ = self._phase(
            store,
            config,
            retriever=self._fixed_retriever(
                [
                    self._candidate("Q1", "c0", "t0", rank=1),
                    self._candidate("Q1", "c1", "t1", rank=2),
                ]
            ),
            reranker=reranker,
        )
        reranker.load()
        context = self._context(
            config,
            questions=[Question(id="Q1", category="cat", synonyms=[])],
        )
        phase.execute(self._compute_plan("doc"), context)
        bundle = store.load_context("doc", "Q1")
        assert bundle is not None
        assert [item.chunks_included[0] for item in bundle.candidates] == ["c1"]

    def test_t_rnk_11_reranker_disabled(self) -> None:
        """Реранкер выключен: top_n по порядку search, early_exit не действует."""
        store = InMemoryArtifactStore()
        store.save_chunks(
            ChunkBatch(
                doc_id="doc",
                chunks=[
                    ChunkRecord(chunk_id="c0", text="t0", chunk_index=0),
                    ChunkRecord(chunk_id="c1", text="t1", chunk_index=1),
                    ChunkRecord(chunk_id="c2", text="t2", chunk_index=2),
                ],
            ),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        config.reranker.enabled = False
        config.reranker.top_n = 2
        config.context.early_exit_threshold = 0.9
        phase, reranker = self._phase(
            store,
            config,
            retriever=self._fixed_retriever(
                [
                    self._candidate("Q1", "c0", "t0", rank=1),
                    self._candidate("Q1", "c1", "t1", rank=2),
                    self._candidate("Q1", "c2", "t2", rank=3),
                ]
            ),
        )
        assembler = FakeContextAssembler()
        phase.assembler = assembler
        context = self._context(
            config,
            questions=[Question(id="Q1", category="cat", synonyms=[])],
        )
        phase.execute(self._compute_plan("doc"), context)
        assert reranker.infer_calls == []
        assert assembler.calls[0][2] is False
        bundle = store.load_context("doc", "Q1")
        assert bundle is not None
        assert [item.chunks_included[0] for item in bundle.candidates] == [
            "c0",
            "c1",
        ]
        assert bundle.candidates[0].rerank_score == pytest.approx(2.0)
        assert bundle.candidates[1].rerank_score == pytest.approx(1.0)

    def test_t_rnk_12_category_default_two_pairs(self) -> None:
        """Умолчание category: два синонима × два чанка — две пары по category."""
        store = InMemoryArtifactStore()
        store.save_chunks(
            ChunkBatch(
                doc_id="doc",
                chunks=[
                    ChunkRecord(chunk_id="c0", text="t0", chunk_index=0),
                    ChunkRecord(chunk_id="c1", text="t1", chunk_index=1),
                ],
            ),
            ArtifactMeta(schema_version="1.0", chunking_hash="ch"),
        )
        config = AppConfig()
        assert config.reranker.query_strategy == "category"
        default_hash = config.reranker_config_hash
        phase, reranker = self._phase(
            store,
            config,
            retriever=self._fixed_retriever(
                [
                    self._candidate("Q1", "c0", "t0", rank=1),
                    self._candidate("Q1", "c1", "t1", rank=2),
                ]
            ),
        )
        reranker.load()
        context = self._context(
            config,
            questions=[Question(id="Q1", category="cat", synonyms=["syn"])],
        )
        phase.execute(self._compute_plan("doc"), context)
        assert len(reranker.infer_calls) == 1
        pairs = reranker.infer_calls[0].pairs
        assert [pair.pair_id for pair in pairs] == ["c0#0", "c1#0"]
        assert all(pair.query == "cat" for pair in pairs)
        other = AppConfig()
        other.reranker.query_strategy = "synonyms_rerank"
        assert other.reranker_config_hash != default_hash

    def test_t_rnk_13_unknown_query_strategy(self) -> None:
        """Чужой reranker.query_strategy — ValueError при разборе конфига."""
        with pytest.raises(ValueError, match="query_strategy"):
            RerankerConfig(query_strategy="mean")

    def _phase(
        self,
        store: InMemoryArtifactStore | FileArtifactStore,
        config: AppConfig,
        retriever: FakeHybridRetriever | None = None,
        reranker: FakeRerankerGateway | None = None,
        assembler: FakeContextAssembler | None = None,
        dump: RetrievalDebugDump | None = None,
    ) -> tuple[RetrievalPhase, FakeRerankerGateway]:
        """Собирает фазу A2.

        Args:
            store: Хранилище.
            config: Конфиг.
            retriever: Подделка поиска.
            reranker: Подделка реранкера.
            assembler: Сборщик.
            dump: Дамп.

        Returns:
            Фаза и реранкер.
        """
        trace = RecordingTraceSink()
        lease = GpuLease(trace)
        active_reranker = reranker or FakeRerankerGateway()
        phase = RetrievalPhase(
            retriever=retriever or FakeHybridRetriever(),
            reranker=active_reranker,
            assembler=assembler or FakeContextAssembler(),
            store=store,
            gate=ArtifactGate(),
            lease=lease,
            trace_sink=trace,
            debug_dump=dump,
        )
        return phase, active_reranker

    def _context(
        self,
        config: AppConfig,
        questions: list[Question],
    ) -> PhaseContext:
        """Контекст с одним документом.

        Args:
            config: Конфиг.
            questions: Вопросы.

        Returns:
            PhaseContext.
        """
        return PhaseContext(
            config=config,
            documents=[DocumentRef(doc_id="doc", path="doc.txt", suffix=".txt")],
            questions=QuestionSet(questions=questions),
            force=True,
        )

    def _compute_plan(self, doc_id: str) -> PhasePlan:
        """План с одним compute.

        Args:
            doc_id: Документ.

        Returns:
            PhasePlan.
        """
        return PhasePlan(
            phase="A2",
            items=[
                PhasePlanItem(
                    document=DocumentRef(
                        doc_id=doc_id,
                        path=f"{doc_id}.txt",
                        suffix=".txt",
                    ),
                    decision=ReuseDecision(action="compute", reason="force"),
                )
            ],
        )

    def _candidate(
        self,
        question_id: str,
        chunk_id: str,
        text: str,
        rank: int,
        vector_score: float = 0.8,
        bm25_score: float = 0.5,
        fused_score: float = 0.8,
        section_path: list[str] | None = None,
        pages: list[int] | None = None,
    ) -> CandidateRecord:
        """Собирает CandidateRecord.

        Args:
            question_id: Вопрос.
            chunk_id: Чанк.
            text: Текст.
            rank: Ранг.
            vector_score: Косинус.
            bm25_score: BM25.
            fused_score: fused.
            section_path: Путь.
            pages: Страницы.

        Returns:
            CandidateRecord.
        """
        return CandidateRecord(
            question_id=question_id,
            chunk_id=chunk_id,
            vector_score=vector_score,
            bm25_score=bm25_score,
            fused_score=fused_score,
            rank=rank,
            text=text,
            section_path=section_path if section_path is not None else [],
            pages=pages or [],
        )

    def _fixed_retriever(
        self,
        candidates: list[CandidateRecord],
    ) -> FakeHybridRetriever:
        """Retriever с фиксированным списком кандидатов.

        Args:
            candidates: Кандидаты.

        Returns:
            Подделка с переопределённым search.
        """
        retriever = FakeHybridRetriever()

        def search(
            doc_id: str,
            question: Question,
            chunks: ChunkBatch,
        ) -> CandidateBatch:
            retriever.calls.append((doc_id, question.id))
            return CandidateBatch(doc_id=doc_id, candidates=list(candidates))

        retriever.search = search  # type: ignore[method-assign]
        return retriever
