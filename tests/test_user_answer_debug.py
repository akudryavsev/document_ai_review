"""T-USR и T-DBG: короткий ответ и отладочная папка."""

from __future__ import annotations

import json
from pathlib import Path

from answer.debug_dump import AnswerDebugDump
from answer.generator import AnswerGenerator
from application.result_sink import FileResultSink
from chunking.debug_dump import ChunkDebugDump
from fakes import (
    FakeCleaner,
    FakeEmbeddingGateway,
    FakeExtractor,
    InMemoryArtifactStore,
    RecordingTraceSink,
)
from gate import ArtifactGate
from gpu import GpuLease
from models.artifacts import AnswerRecord, ContextBundle, ContextCandidate
from models.config import AppConfig, RuntimeConfig
from models.documents import ChunkBatch, ChunkRecord, DocumentRef
from models.gateways import GenerateResponse
from models.phases import PhaseContext
from models.questions import Question, QuestionSet
from phases.answering import AnsweringPhase
from phases.indexing import IndexingPhase
from ports.gateways import LlmGateway
from ports.pipeline import Chunker
from tests.test_answer_generator import ScriptedLlm


class FixedChunker(Chunker):
    """Отдаёт заранее заданные чанки.

    Attributes:
        chunks_by_doc: doc_id → список чанков.
    """

    def __init__(self, chunks_by_doc: dict[str, list[ChunkRecord]]) -> None:
        """Сохраняет чанки.

        Args:
            chunks_by_doc: Пакеты по документам.
        """
        self.chunks_by_doc = chunks_by_doc

    def chunk(self, document):  # noqa: ANN001 — порт CleanedDocument
        """Возвращает пакет документа.

        Args:
            document: Очищенный документ.

        Returns:
            ChunkBatch.
        """
        chunks = self.chunks_by_doc.get(document.doc_id, [])
        return ChunkBatch(doc_id=document.doc_id, chunks=list(chunks))


class TestUserAnswerReport:
    """Каталог T-USR."""

    def test_t_usr_01_question_and_answer_only(self, tmp_path: Path) -> None:
        """В md вопрос, ответ и пояснение. Цитата и промпт не попадают."""
        sink = FileResultSink(tmp_path)
        sink.write_answer(
            self._record(
                question_id="Q001",
                category="Первый вопрос",
                answer="Короткий ответ",
                explanation="секрет пояснения",
                reason="not_present",
                prepared_context="секрет промпта",
            )
        )
        sink.write_answer(
            self._record(
                question_id="Q002",
                category="Второй вопрос",
                answer="",
                reason="missing_context",
                status="failed",
            )
        )
        text = (tmp_path / "doc.docx.md").read_text(encoding="utf-8")
        assert text == (
            "## Первый вопрос\n"
            "\n"
            "Короткий ответ\n"
            "\n"
            "секрет пояснения\n"
            "\n"
            "## Второй вопрос\n"
            "\n"
            "Контекст для ответа не собран.\n"
        )
        assert "секрет промпта" not in text
        assert "цитата" not in text

    def test_t_usr_02_repeated_category_shows_model(self, tmp_path: Path) -> None:
        """Две тройки одного вопроса подписывают имя модели ответа."""
        sink = FileResultSink(tmp_path)
        sink.write_answer(
            self._record(
                question_id="Q001",
                category="Общий вопрос",
                answer="Ответ А",
                answer_model="a.gguf",
            )
        )
        sink.write_answer(
            self._record(
                question_id="Q001",
                category="Общий вопрос",
                answer="Ответ Б",
                answer_model="b.gguf",
            )
        )
        text = (tmp_path / "doc.docx.md").read_text(encoding="utf-8")
        assert text.index("a.gguf") < text.index("Ответ А")
        assert text.index("b.gguf") < text.index("Ответ Б")

    def test_t_usr_03_single_category_hides_model(self, tmp_path: Path) -> None:
        """Одна строка на вопрос не печатает имя модели."""
        sink = FileResultSink(tmp_path)
        sink.write_answer(
            self._record(
                question_id="Q001",
                category="Один вопрос",
                answer="Только ответ",
                answer_model="gemma-secret.gguf",
            )
        )
        text = (tmp_path / "doc.docx.md").read_text(encoding="utf-8")
        assert "Только ответ" in text
        assert "gemma-secret.gguf" not in text

    def _record(
        self,
        *,
        question_id: str,
        category: str,
        answer: str,
        explanation: str = "",
        reason: str | None = None,
        status: str = "ok",
        answer_model: str = "answer.gguf",
        prepared_context: str | None = None,
    ) -> AnswerRecord:
        """Строка ответа для md.

        Args:
            question_id: Вопрос.
            category: Текст вопроса.
            answer: Поле answer.
            explanation: Пояснение для md.
            reason: Причина.
            status: ok или failed.
            answer_model: Имя файла модели ответа.
            prepared_context: Промпт, которого не должно быть в md.

        Returns:
            AnswerRecord.
        """
        return AnswerRecord(
            doc_id="doc.docx",
            question_id=question_id,
            category=category,
            answer=answer,
            explanation=explanation,
            reason=reason,
            found=False,
            status=status,
            model=answer_model,
            answer_model=answer_model,
            prepared_context=prepared_context,
        )


class TestDebugDump:
    """Каталог T-DBG."""

    def test_t_dbg_01_warning_skips_new_debug_files(self, tmp_path: Path) -> None:
        """Без DEBUG новые файлы не создаются, md ответа есть."""
        results = tmp_path / "results"
        cache = tmp_path / "cache"
        sink = FileResultSink(results)
        record = AnswerRecord(
            doc_id="doc.docx",
            question_id="Q001",
            category="Вопрос",
            answer="Ответ",
            found=False,
            status="ok",
        )
        sink.write_answer(record)
        AnswerDebugDump(enabled=False, cache_dir=str(cache)).write_pair(record)
        AnswerDebugDump(enabled=False, cache_dir=str(cache)).write_prompt(
            "doc.docx",
            "Q001",
            "промпт",
        )
        AnswerDebugDump(enabled=False, cache_dir=str(cache)).write_attempts(
            "doc.docx",
            "Q001",
            [],
        )
        ChunkDebugDump(enabled=False, cache_dir=str(cache)).write(
            "doc.docx",
            [ChunkRecord(chunk_id="c_0000", text="текст", chunk_index=0)],
        )
        assert (results / "doc.docx.md").is_file()
        debug = cache / "debug"
        assert not (debug / "doc.docx.chunks.md").exists()
        assert not (debug / "doc.docx" / "Q001.prompt.txt").exists()
        assert not (debug / "doc.docx" / "Q001.attempts.jsonl").exists()
        assert not (debug / "doc.docx" / "Q001.pair.json").exists()

    def test_t_dbg_02_attempts_keep_raw_text(self, tmp_path: Path) -> None:
        """Первая попытка не JSON, вторая валидна. Промпт не в jsonl."""
        cache = tmp_path / "cache"
        results = tmp_path / "results"
        valid = json.dumps(
            {
                "found": True,
                "answer": "Годный ответ",
                "explanation": "пояснение",
                "reason": None,
                "citations": [{"n": 1, "quote": "цитата"}],
            },
            ensure_ascii=False,
        )
        llm = ScriptedLlm(
            [
                GenerateResponse(text="не json"),
                GenerateResponse(text=valid),
            ]
        )
        self._answer(
            tmp_path,
            llm,
            cache,
            results,
            bundle=self._bundle(empty=False),
        )
        prompt_path = cache / "debug" / "doc.docx" / "Q001.prompt.txt"
        attempts_path = cache / "debug" / "doc.docx" / "Q001.attempts.jsonl"
        prompt = prompt_path.read_text(encoding="utf-8")
        assert prompt == llm.calls[0].prompt
        rows = [
            json.loads(line)
            for line in attempts_path.read_text(encoding="utf-8").splitlines()
        ]
        assert rows[0]["outcome"] == "invalid_json"
        assert rows[0]["text"] == "не json"
        assert rows[1]["outcome"] == "ok"
        stored = AnswerRecord.model_validate_json(
            (results / "doc.docx.jsonl").read_text(encoding="utf-8").strip()
        )
        assert stored.prepared_context is None

    def test_t_dbg_03_exceptions_are_attempts(self, tmp_path: Path) -> None:
        """Каждое исключение — строка attempts, в md короткая фраза."""
        cache = tmp_path / "cache"
        results = tmp_path / "results"
        llm = ScriptedLlm([RuntimeError("down"), RuntimeError("down")])
        self._answer(
            tmp_path,
            llm,
            cache,
            results,
            bundle=self._bundle(empty=False),
        )
        rows = [
            json.loads(line)
            for line in (
                cache / "debug" / "doc.docx" / "Q001.attempts.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 2
        assert rows[0]["outcome"] == "exception"
        assert rows[0]["text"] == ""
        assert rows[1]["outcome"] == "exception"
        text = (results / "doc.docx.md").read_text(encoding="utf-8")
        assert "Модель недоступна." in text

    def test_t_dbg_04_empty_search_has_pair_only(self, tmp_path: Path) -> None:
        """Пустой поиск не пишет промпт и попытки."""
        cache = tmp_path / "cache"
        results = tmp_path / "results"
        llm = ScriptedLlm([])
        self._answer(
            tmp_path,
            llm,
            cache,
            results,
            bundle=self._bundle(empty=True),
        )
        folder = cache / "debug" / "doc.docx"
        assert not (folder / "Q001.prompt.txt").exists()
        assert not (folder / "Q001.attempts.jsonl").exists()
        pair = json.loads((folder / "Q001.pair.json").read_text(encoding="utf-8"))
        assert pair["reason"] == "no_relevant_chunks"
        assert llm.calls == []

    def test_t_dbg_05_missing_context(self, tmp_path: Path) -> None:
        """Нет бандла — только pair.json и короткая фраза в md."""
        cache = tmp_path / "cache"
        results = tmp_path / "results"
        llm = ScriptedLlm([])
        self._answer(tmp_path, llm, cache, results, bundle=None)
        folder = cache / "debug" / "doc.docx"
        names = {path.name for path in folder.iterdir()}
        assert names == {"Q001.pair.json"}
        pair = json.loads((folder / "Q001.pair.json").read_text(encoding="utf-8"))
        assert pair["reason"] == "missing_context"
        text = (results / "doc.docx.md").read_text(encoding="utf-8")
        assert "Контекст для ответа не собран." in text

    def test_t_dbg_06_two_chunks(self, tmp_path: Path) -> None:
        """chunks.md содержит оба идентификатора и оба текста."""
        cache = tmp_path / "cache"
        doc_id = "two.docx"
        chunks = [
            ChunkRecord(chunk_id="c_0001", text="бета", chunk_index=1, page_from=2, page_to=2),
            ChunkRecord(
                chunk_id="c_0000",
                text="альфа",
                chunk_index=0,
                page_from=1,
                page_to=1,
                section_path=["Глава", "Пункт"],
            ),
        ]
        phase = self._indexing(
            cache,
            doc_id,
            FixedChunker({doc_id: chunks}),
            chunk_dump=ChunkDebugDump(enabled=True, cache_dir=str(cache)),
        )
        context = self._index_context(cache, doc_id)
        phase.execute(phase.plan(context), context)
        text = (cache / "debug" / "two.docx.chunks.md").read_text(encoding="utf-8")
        assert text.index("c_0000") < text.index("c_0001")
        assert "альфа" in text
        assert "бета" in text
        assert "Глава / Пункт" in text

    def test_t_dbg_07_debug_writes_extract_without_flag(self, tmp_path: Path) -> None:
        """DEBUG пишет extract.md при выключенном extraction.debug_dump."""
        cache = tmp_path / "cache"
        doc_id = "scan.pdf"
        phase = self._indexing(
            cache,
            doc_id,
            FixedChunker({}),
            no_text_layer={doc_id},
        )
        context = self._index_context(cache, doc_id)
        assert context.config.extraction.debug_dump is False
        phase.execute(phase.plan(context), context)
        assert (cache / "debug" / "scan.pdf.extract.md").is_file()

    def _answer(
        self,
        tmp_path: Path,
        llm: LlmGateway,
        cache: Path,
        results: Path,
        bundle: ContextBundle | None,
    ) -> None:
        """Прогоняет фазу B для одного документа и одного вопроса.

        Args:
            tmp_path: Корень теста.
            llm: Шлюз с очередью ответов.
            cache: Каталог debug.
            results: Каталог md и jsonl.
            bundle: Бандл пары или None, если контекста нет.
        """
        if isinstance(llm, ScriptedLlm):
            llm.load()
        trace = RecordingTraceSink()
        store = InMemoryArtifactStore(trace)
        if bundle is not None:
            store.save_context(bundle)
        config = AppConfig(
            runtime=RuntimeConfig(
                log_level="DEBUG",
                cache_dir=str(cache),
                results_dir=str(results),
                retries=2,
            )
        )
        question = Question(id="Q001", category="Срок хранения")
        context = PhaseContext(
            config=config,
            documents=[DocumentRef(doc_id="doc.docx", path="/tmp/doc.docx", suffix=".docx")],
            questions=QuestionSet(version="1.0", questions=[question]),
            requested_phases=["B"],
        )
        phase = AnsweringPhase(
            llm=llm,
            generator=AnswerGenerator(
                answer_config=config.answer,
                runtime_config=config.runtime,
                llm=llm,
            ),
            result_sink=FileResultSink(results),
            store=store,
            gate=ArtifactGate(),
            lease=GpuLease(trace),
            trace_sink=trace,
            answer_dump=AnswerDebugDump(enabled=True, cache_dir=str(cache)),
        )
        phase.execute(phase.plan(context), context)

    def _bundle(self, *, empty: bool) -> ContextBundle:
        """Бандл одного фрагмента или пустого поиска.

        Args:
            empty: Пустой поиск.

        Returns:
            ContextBundle.
        """
        candidates: list[ContextCandidate] = []
        if not empty:
            candidates.append(
                ContextCandidate(
                    rank_after_rerank=1,
                    rerank_score=0.5,
                    section_path=["Раздел"],
                    pages=[1],
                    text="текст фрагмента",
                    chunks_included=["c_0000"],
                )
            )
        return ContextBundle(
            doc_id="doc.docx",
            question_id="Q001",
            category="Срок хранения",
            empty_search=empty,
            reason="no_relevant_chunks" if empty else None,
            candidates=candidates,
        )

    def _indexing(
        self,
        cache: Path,
        doc_id: str,
        chunker: Chunker,
        chunk_dump: ChunkDebugDump | None = None,
        no_text_layer: set[str] | None = None,
    ) -> IndexingPhase:
        """Собирает фазу A1 с дампом в cache.

        Args:
            cache: runtime.cache_dir.
            doc_id: Документ. Нужен сигнатуре вызова, пакет задаёт chunker.
            chunker: Нарезка.
            chunk_dump: Дамп чанков или None.
            no_text_layer: Документы без текстового слоя.

        Returns:
            IndexingPhase.
        """
        del doc_id
        trace = RecordingTraceSink()
        return IndexingPhase(
            extractor=FakeExtractor(no_text_layer_docs=no_text_layer),
            cleaner=FakeCleaner(),
            chunker=chunker,
            embedding=FakeEmbeddingGateway(),
            store=InMemoryArtifactStore(trace),
            gate=ArtifactGate(),
            lease=GpuLease(trace),
            trace_sink=trace,
            chunk_dump=chunk_dump,
        )

    def _index_context(self, cache: Path, doc_id: str) -> PhaseContext:
        """Контекст A1 с уровнем DEBUG и выключенным флагом extract.

        Args:
            cache: runtime.cache_dir.
            doc_id: Документ.

        Returns:
            PhaseContext.
        """
        return PhaseContext(
            config=AppConfig(
                runtime=RuntimeConfig(log_level="DEBUG", cache_dir=str(cache))
            ),
            documents=[
                DocumentRef(doc_id=doc_id, path=f"/tmp/{doc_id}", suffix=".docx")
            ],
            questions=None,
            requested_phases=["A1"],
        )
