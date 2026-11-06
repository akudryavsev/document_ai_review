"""T-CTX: сборка контекста и отладочный дамп."""

from __future__ import annotations

from pathlib import Path

import pytest

from context.assembler import ContextAssembler
from context.debug_dump import ContextDebugDump
from fakes import FakeHybridRetriever, FakeRerankerGateway, RecordingTraceSink
from gate import ArtifactGate
from gpu import GpuLease
from models.artifacts import ArtifactMeta, ContextBundle
from models.config import AppConfig, ContextConfig, RerankerConfig, RetrievalConfig, RuntimeConfig
from models.documents import (
    CandidateBatch,
    CandidateRecord,
    ChunkBatch,
    ChunkRecord,
    DocumentRef,
)
from models.phases import PhaseContext, PhasePlan, PhasePlanItem, ReuseDecision
from models.questions import Question, QuestionSet
from phases.retrieval import RetrievalPhase
from store.file_store import FileArtifactStore


class TestContextAssembler:
    """Каталог T-CTX-01…18."""

    def test_t_ctx_01_empty_search_copies_synonyms(self) -> None:
        """Пустой поиск копирует category и синонимы, бюджет нулевой."""
        question = self._question()
        bundle = self._assembler().assemble(
            doc_id="doc",
            question=question,
            candidates=self._batch([]),
            empty_search=True,
            chunks=self._chunks([]),
        )
        assert bundle.empty_search is True
        assert bundle.reason == "no_relevant_chunks"
        assert bundle.candidates == []
        assert bundle.category == question.category
        assert bundle.used_synonyms == ["синоним-а", "синоним-б"]
        assert bundle.budget_used_tokens == 0
        assert bundle.retrieval_meta["after_rerank"] == 0
        assert bundle.retrieval_meta["top_k"] == 20

    def test_t_ctx_02_empty_flag_with_hits_raises(self) -> None:
        """Непустой пакет при empty_search=true — ошибка."""
        chunk = self._chunk("c_0000", "текст", 0, ["А"])
        with pytest.raises(ValueError):
            self._assembler().assemble(
                doc_id="doc",
                question=self._question(),
                candidates=self._batch([self._hit(chunk, 0.5)]),
                empty_search=True,
                chunks=self._chunks([chunk]),
            )

    def test_t_ctx_03_nonempty_flag_requires_score(self) -> None:
        """Пустой пакет или скор None при непустом поиске — ошибка."""
        assembler = self._assembler()
        question = self._question()
        with pytest.raises(ValueError):
            assembler.assemble(
                doc_id="doc",
                question=question,
                candidates=self._batch([]),
                empty_search=False,
                chunks=self._chunks([]),
            )
        chunk = self._chunk("c_0000", "текст", 0, ["А"])
        hit = self._hit(chunk, None)
        with pytest.raises(ValueError):
            assembler.assemble(
                doc_id="doc",
                question=question,
                candidates=self._batch([hit]),
                empty_search=False,
                chunks=self._chunks([chunk]),
            )

    def test_t_ctx_04_short_section_includes_every_chunk(self) -> None:
        """Два хита и соседний чанк короткой секции склеиваются в один фрагмент."""
        chunks = self._linked(
            [
                ("c_0000", "первый абзац", ["Сроки"]),
                ("c_0001", "второй абзац", ["Сроки"]),
                ("c_0002", "третий абзац", ["Сроки"]),
            ]
        )
        hits = [self._hit(chunks[0], 0.4), self._hit(chunks[2], 0.9)]
        bundle = self._assemble(chunks, hits)
        assert len(bundle.candidates) == 1
        candidate = bundle.candidates[0]
        assert candidate.chunks_included == ["c_0000", "c_0001", "c_0002"]
        assert candidate.truncated is False
        assert candidate.text == "первый абзац\nвторой абзац\nтретий абзац"
        assert "[1]" not in candidate.text
        assert candidate.rerank_score == 0.9

    def test_t_ctx_05_different_paths_stay_apart(self) -> None:
        """Две секции не смешивают текст, выше та, у которой больше скор."""
        chunks = self._linked(
            [
                ("c_0000", "текст альфа", ["Альфа"]),
                ("c_0001", "текст бета", ["Бета"]),
            ]
        )
        hits = [self._hit(chunks[0], 0.2), self._hit(chunks[1], 0.8)]
        bundle = self._assemble(chunks, hits)
        assert [item.section_path for item in bundle.candidates] == [
            ["Бета"],
            ["Альфа"],
        ]
        assert bundle.candidates[0].rank_after_rerank == 1
        assert "текст бета" in bundle.candidates[0].text
        assert "текст альфа" not in bundle.candidates[0].text
        assert "текст альфа" in bundle.candidates[1].text
        assert "текст бета" not in bundle.candidates[1].text

    def test_t_ctx_06_window_stops_at_fallback_width(self) -> None:
        """Окно без пути берёт по одному соседу и не берёт чанк на расстоянии 2."""
        chunks = self._linked(
            [
                ("c_0000", "далеко-слева", None),
                ("c_0001", "слева", None),
                ("c_0002", "хит", None),
                ("c_0003", "справа", None),
                ("c_0004", "далеко-справа", None),
            ]
        )
        bundle = self._assemble(
            chunks,
            [self._hit(chunks[2], 0.7)],
            neighbors_fallback=1,
        )
        included = bundle.candidates[0].chunks_included
        assert included == ["c_0001", "c_0002", "c_0003"]
        assert "далеко-слева" not in bundle.candidates[0].text
        assert "далеко-справа" not in bundle.candidates[0].text

    def test_t_ctx_07_budget_keeps_second_hit_and_cuts_neighbor(self) -> None:
        """Бюджет оставляет хит второй секции и снимает её соседа."""
        chunks = self._linked(
            [
                ("c_0000", "альфа один два три", ["Альфа"]),
                ("c_0001", "бета хит", ["Бета"]),
                ("c_0002", "бета сосед лишний", ["Бета"]),
            ]
        )
        hits = [self._hit(chunks[0], 0.9), self._hit(chunks[1], 0.4)]
        bundle = self._assemble(chunks, hits, max_total_tokens=6, max_section_tokens=100)
        assert len(bundle.candidates) == 2
        second = bundle.candidates[1]
        assert second.truncated is True
        assert "бета хит" in second.text
        assert "бета сосед лишний" not in second.text
        assert bundle.candidates[0].truncated is False
        assert "альфа один два три" in bundle.candidates[0].text

    def test_t_ctx_08_sections_used_skips_pathless(self) -> None:
        """В sections_used попадает последний заголовок, окно без пути молчит."""
        chunks = self._linked(
            [
                ("c_0000", "текст секции", ["Глава", "3.2.1. Сроки"]),
                ("c_0001", "без пути", None),
            ]
        )
        hits = [self._hit(chunks[0], 0.6), self._hit(chunks[1], 0.2)]
        bundle = self._assemble(chunks, hits)
        assert bundle.retrieval_meta["sections_used"] == ["3.2.1. Сроки"]

    def test_t_ctx_09_assembler_has_no_gateways(self) -> None:
        """Сборщик не держит реранкер, LLM и хранилище."""
        assembler = self._assembler()
        assert not hasattr(assembler, "reranker")
        assert not hasattr(assembler, "llm")
        assert not hasattr(assembler, "store")
        chunk = self._chunk("c_0000", "текст", 0, ["А"])
        self._assemble([chunk], [self._hit(chunk, 0.5)])
        assembler.assemble(
            doc_id="doc",
            question=self._question(),
            candidates=self._batch([]),
            empty_search=True,
            chunks=self._chunks([]),
        )

    def test_t_ctx_10_window_enters_named_section(self) -> None:
        """Окно без пути забирает соседний именованный чанк, путь фрагмента пуст."""
        chunks = self._linked(
            [
                ("c_0000", "хит без пути", None),
                ("c_0001", "текст соседа", ["Соседний раздел"]),
            ]
        )
        bundle = self._assemble(
            chunks,
            [self._hit(chunks[0], 0.5)],
            neighbors_fallback=1,
        )
        candidate = bundle.candidates[0]
        assert "текст соседа" in candidate.text
        assert candidate.section_path is None
        assert bundle.retrieval_meta["sections_used"] == []

    def test_t_ctx_11_overlapping_windows_stay_duplicated(self) -> None:
        """Два окна без пути не сливаются, общий чанк входит в оба."""
        chunks = self._linked(
            [
                ("c_0000", "лево", None),
                ("c_0001", "хит-старший", None),
                ("c_0002", "общий", None),
                ("c_0003", "хит-младший", None),
                ("c_0004", "право", None),
            ]
        )
        hits = [self._hit(chunks[1], 0.9), self._hit(chunks[3], 0.3)]
        bundle = self._assemble(chunks, hits, neighbors_fallback=1)
        assert len(bundle.candidates) == 2
        assert bundle.candidates[0].rerank_score == 0.9
        assert bundle.candidates[1].rerank_score == 0.3
        assert "c_0002" in bundle.candidates[0].chunks_included
        assert "c_0002" in bundle.candidates[1].chunks_included

    def test_t_ctx_12_total_budget_cuts_outer_neighbors(self) -> None:
        """Соседи снимаются с краёв, хит остаётся, фрагмент помечен truncated."""
        chunks = self._linked(
            [
                ("c_0000", "alpha", ["Секция"]),
                ("c_0001", "beta", ["Секция"]),
                ("c_0002", "gamma", ["Секция"]),
                ("c_0003", "delta", ["Секция"]),
                ("c_0004", "epsilon", ["Секция"]),
            ]
        )
        bundle = self._assemble(
            chunks,
            [self._hit(chunks[2], 0.8)],
            max_section_tokens=100,
            max_total_tokens=3,
        )
        candidate = bundle.candidates[0]
        assert candidate.truncated is True
        assert candidate.chunks_included == ["c_0001", "c_0002", "c_0003"]
        assert "gamma" in candidate.text
        assert "alpha" not in candidate.text
        assert "epsilon" not in candidate.text

    def test_t_ctx_13_config_rejects_small_total_budget(self) -> None:
        """Бюджет меньше top_n * размер чанка не загружается."""
        with pytest.raises(ValueError):
            AppConfig(
                context=ContextConfig(max_total_tokens=100),
                reranker=RerankerConfig(top_n=5),
            )
        AppConfig()

    def test_t_ctx_14_other_expansion_raises(self) -> None:
        """neighbors и none не подменяются режимом section."""
        chunk = self._chunk("c_0000", "текст", 0, ["А"])
        for expansion in ("neighbors", "none"):
            assembler = self._assembler(expansion=expansion)
            with pytest.raises(ValueError):
                assembler.assemble(
                    doc_id="doc",
                    question=self._question(),
                    candidates=self._batch([self._hit(chunk, 0.5)]),
                    empty_search=False,
                    chunks=self._chunks([chunk]),
                )

    def test_t_ctx_15_missing_hit_chunk_raises(self) -> None:
        """Хит, которого нет в пакете чанков, — ошибка."""
        chunk = self._chunk("c_0000", "текст", 0, ["А"])
        missing = self._chunk("c_9999", "нет", 1, ["А"])
        with pytest.raises(ValueError):
            self._assemble([chunk], [self._hit(missing, 0.5)])

    def test_t_ctx_16_debug_dump_disabled_keeps_json(
        self,
        tmp_path: Path,
    ) -> None:
        """Без флага Markdown нет, боевой JSON пары записан."""
        self._run_phase(tmp_path, debug_dump=False, empty=False)
        assert not (
            tmp_path / "cache" / "debug" / "doc.docx" / "Q1.context.md"
        ).exists()
        assert (tmp_path / "contexts" / "doc.docx" / "Q1.json").is_file()

    def test_t_ctx_17_debug_dump_empty_search(self, tmp_path: Path) -> None:
        """Дамп пустого поиска содержит причину, синонимы и пустые фрагменты."""
        self._run_phase(tmp_path, debug_dump=True, empty=True)
        text = (
            tmp_path / "cache" / "debug" / "doc.docx" / "Q1.context.md"
        ).read_text(encoding="utf-8")
        assert "no_relevant_chunks" in text
        assert "Срок хранения" in text
        assert text.index("синоним-а") < text.index("синоним-б")
        assert "Фрагментов нет." in text

    def test_t_ctx_18_debug_dump_fragment_text(self, tmp_path: Path) -> None:
        """Дамп содержит текст, путь и chunk_id без префикса промпта."""
        self._run_phase(tmp_path, debug_dump=True, empty=False)
        text = (
            tmp_path / "cache" / "debug" / "doc.docx" / "Q1.context.md"
        ).read_text(encoding="utf-8")
        assert "хранится пять лет" in text
        assert "3.2.1. Сроки" in text
        assert "c_0000" in text
        assert "[1] стр." not in text

    def test_debug_dump_disabled_write_returns_none(self, tmp_path: Path) -> None:
        """Выключенный ContextDebugDump файл не создаёт."""
        bundle = ContextBundle(
            doc_id="doc",
            question_id="Q1",
            category="вопрос",
            empty_search=True,
            reason="no_relevant_chunks",
        )
        dump = ContextDebugDump(enabled=False, cache_dir=str(tmp_path))
        assert dump.write(bundle) is None
        assert not dump.path_for("doc", "Q1").exists()

    def _assemble(
        self,
        chunks: list[ChunkRecord],
        hits: list[CandidateRecord],
        neighbors_fallback: int = 2,
        max_total_tokens: int = 6000,
        max_section_tokens: int = 3000,
    ) -> ContextBundle:
        """Собирает непустой бандл из готовых чанков и хитов.

        Args:
            chunks: Чанки документа.
            hits: Хиты реранка.
            neighbors_fallback: Ширина окна.
            max_total_tokens: Общий бюджет.
            max_section_tokens: Лимит секции.

        Returns:
            ContextBundle.
        """
        assembler = self._assembler(
            neighbors_fallback=neighbors_fallback,
            max_total_tokens=max_total_tokens,
            max_section_tokens=max_section_tokens,
        )
        return assembler.assemble(
            doc_id="doc",
            question=self._question(),
            candidates=self._batch(hits),
            empty_search=False,
            chunks=self._chunks(chunks),
        )

    def _assembler(self, expansion: str = "section", **kwargs: int) -> ContextAssembler:
        """Собирает сборщик с тестовыми именами моделей.

        Args:
            expansion: Режим контекста.
            **kwargs: Поля ContextConfig числом.

        Returns:
            ContextAssembler.
        """
        config = ContextConfig(
            expansion=expansion,  # type: ignore[arg-type]
            neighbors_fallback=int(kwargs.get("neighbors_fallback", 2)),
            max_total_tokens=int(kwargs.get("max_total_tokens", 6000)),
            max_section_tokens=int(kwargs.get("max_section_tokens", 3000)),
        )
        return ContextAssembler(
            config=config,
            embedding_model="emb",
            reranker_model="rerank",
            top_k=20,
        )

    def _question(self) -> Question:
        """Вопрос с двумя синонимами.

        Returns:
            Question.
        """
        return Question(
            id="Q1",
            category="Срок хранения",
            synonyms=["синоним-а", "синоним-б"],
        )

    def _batch(self, hits: list[CandidateRecord]) -> CandidateBatch:
        """Пакет хитов одного документа.

        Args:
            hits: Кандидаты.

        Returns:
            CandidateBatch.
        """
        return CandidateBatch(doc_id="doc", candidates=hits)

    def _chunks(self, chunks: list[ChunkRecord]) -> ChunkBatch:
        """Пакет чанков.

        Args:
            chunks: Чанки.

        Returns:
            ChunkBatch.
        """
        return ChunkBatch(doc_id="doc", chunks=chunks)

    def _chunk(
        self,
        chunk_id: str,
        text: str,
        index: int,
        section_path: list[str] | None,
    ) -> ChunkRecord:
        """Один чанк без соседей.

        Args:
            chunk_id: Идентификатор.
            text: Текст.
            index: chunk_index.
            section_path: Путь или None.

        Returns:
            ChunkRecord.
        """
        return ChunkRecord(
            chunk_id=chunk_id,
            text=text,
            section_path=section_path,
            chunk_index=index,
            page_from=index + 1,
            page_to=index + 1,
        )

    def _linked(
        self,
        rows: list[tuple[str, str, list[str] | None]],
    ) -> list[ChunkRecord]:
        """Цепочка чанков с prev_id и next_id.

        Args:
            rows: Тройки id, текст, путь.

        Returns:
            Список ChunkRecord по порядку.
        """
        records: list[ChunkRecord] = []
        for index, (chunk_id, text, path) in enumerate(rows):
            previous = rows[index - 1][0] if index else None
            following = rows[index + 1][0] if index + 1 < len(rows) else None
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=text,
                    section_path=path,
                    chunk_index=index,
                    prev_id=previous,
                    next_id=following,
                    page_from=12 if index == 0 else 0,
                    page_to=13 if index == 0 else 0,
                )
            )
        return records

    def _hit(self, chunk: ChunkRecord, score: float | None) -> CandidateRecord:
        """Хит реранка по чанку.

        Args:
            chunk: Чанк документа.
            score: rerank_score или None.

        Returns:
            CandidateRecord.
        """
        return CandidateRecord(
            question_id="Q1",
            chunk_id=chunk.chunk_id,
            rerank_score=score,
            text=chunk.text,
            section_path=chunk.section_path,
            rank=chunk.chunk_index + 1,
        )

    def _run_phase(self, tmp_path: Path, debug_dump: bool, empty: bool) -> None:
        """Прогоняет A2 на одном документе и одном вопросе.

        Args:
            tmp_path: Временный корень файлов.
            debug_dump: Писать ли context.md.
            empty: Пустой ли поиск.
        """
        cache_dir = tmp_path / "cache"
        contexts_dir = tmp_path / "contexts"
        index_dir = tmp_path / "index"
        store = FileArtifactStore(cache_dir, index_dir, contexts_dir)
        chunk = ChunkRecord(
            chunk_id="c_0000",
            text="хранится пять лет",
            section_path=["3. Регламент", "3.2.1. Сроки"],
            chunk_index=0,
            page_from=12,
            page_to=13,
        )
        store.save_chunks(
            ChunkBatch(doc_id="doc.docx", chunks=[chunk]),
            ArtifactMeta(schema_version="1.0", doc_id="doc.docx", n_chunks=1),
        )
        trace = RecordingTraceSink()
        retriever = FakeHybridRetriever()
        if empty:
            retriever.empty_pairs.add(("doc.docx", "Q1"))
        config = AppConfig(
            retrieval=RetrievalConfig(debug_dump=debug_dump, persist_candidates=False),
            reranker=RerankerConfig(enabled=False),
            runtime=RuntimeConfig(cache_dir=str(cache_dir)),
            phases=self._phases(contexts_dir, index_dir),
        )
        phase = RetrievalPhase(
            retriever=retriever,
            reranker=FakeRerankerGateway(),
            assembler=ContextAssembler(
                config=config.context,
                embedding_model=config.embedding.model,
                reranker_model=config.reranker.model,
                top_k=config.retrieval.top_k,
            ),
            store=store,
            gate=ArtifactGate(),
            lease=GpuLease(trace),
            trace_sink=trace,
        )
        question = Question(
            id="Q1",
            category="Срок хранения",
            synonyms=["синоним-а", "синоним-б"],
        )
        context = PhaseContext(
            config=config,
            documents=[
                DocumentRef(doc_id="doc.docx", path="doc.docx", suffix=".docx")
            ],
            questions=QuestionSet(questions=[question]),
        )
        plan = PhasePlan(
            phase="A2",
            items=[
                PhasePlanItem(
                    document=context.documents[0],
                    decision=ReuseDecision(action="compute", reason="force"),
                )
            ],
        )
        report = phase.execute(plan, context)
        assert report.failed == 0

    def _phases(self, contexts_dir: Path, index_dir: Path):
        """Каталоги фаз для тестового конфига.

        Args:
            contexts_dir: Каталог бандлов.
            index_dir: Каталог кандидатов.

        Returns:
            PhasesConfig.
        """
        from models.config import PhasesConfig

        return PhasesConfig(
            contexts_dir=str(contexts_dir),
            index_dir=str(index_dir),
        )
