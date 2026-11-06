"""T-STO: файловое и in-memory хранилище артефактов."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fakes.trace import RecordingTraceSink
from models.artifacts import ArtifactMeta, ContextBundle
from models.documents import CandidateBatch, CandidateRecord, ChunkBatch, ChunkRecord
from models.questions import Question, QuestionSet
from store.errors import EmbeddingBindingError
from store.file_store import FileArtifactStore
from store.memory_store import InMemoryArtifactStore


class StoreSamples:
    """Собирает пакеты для сценариев хранилища."""

    def batch(
        self,
        doc_id: str = "reglament.docx",
        texts: list[str] | None = None,
    ) -> tuple[ChunkBatch, ArtifactMeta]:
        """Пакет чанков и meta с source_sha256.

        Args:
            doc_id: Идентификатор документа.
            texts: Тексты чанков.

        Returns:
            Пара batch и meta.
        """
        chosen = texts or ["Срок хранения составляет 5 лет."]
        chunks: list[ChunkRecord] = []
        last = len(chosen) - 1
        for index, text in enumerate(chosen):
            prev_id = None if index == 0 else f"c_{index - 1:04d}"
            next_id = None if index == last else f"c_{index + 1:04d}"
            chunks.append(
                ChunkRecord(
                    chunk_id=f"c_{index:04d}",
                    text=text,
                    page_from=index + 1,
                    page_to=index + 1,
                    section_path=["Сроки"],
                    chunk_index=index,
                    prev_id=prev_id,
                    next_id=next_id,
                )
            )
        batch = ChunkBatch(doc_id=doc_id, chunks=chunks, chunking_hash="hash-a")
        meta = ArtifactMeta(
            schema_version="1.0",
            chunking_hash="hash-a",
            embedding_model="fake-model",
            n_chunks=len(chunks),
            source_sha256="ab" * 32,
            doc_id=doc_id,
        )
        return batch, meta

    def digest(self, text: str) -> str:
        """SHA-256 текста.

        Args:
            text: Текст чанка.

        Returns:
            Hex.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def matrix(self, rows: int, dim: int) -> bytes:
        """Байты матрицы нужного размера.

        Args:
            rows: Число чанков.
            dim: Размерность.

        Returns:
            payload длиной rows * dim * 2.
        """
        return b"\x00\x3c" * (rows * dim)


class TestFileArtifactStore:
    """Каталог T-STO для FileArtifactStore."""

    def setup_method(self) -> None:
        """Готовит образцы."""
        self.samples = StoreSamples()

    def test_t_sto_01_roundtrip_chunks(self, tmp_path: Path) -> None:
        """save_chunks + load_chunks сохраняют поля чанков и doc_id."""
        store = self._store(tmp_path)
        batch, meta = self.samples.batch(texts=["Первый", "Второй"])
        store.save_chunks(batch, meta)
        loaded = store.load_chunks(batch.doc_id)
        assert loaded is not None
        assert loaded.doc_id == batch.doc_id
        assert [chunk.chunk_id for chunk in loaded.chunks] == ["c_0000", "c_0001"]
        assert [chunk.text for chunk in loaded.chunks] == ["Первый", "Второй"]
        assert loaded.chunks[0].section_path == ["Сроки"]
        assert loaded.chunks[0].page_from == 1
        assert loaded.chunks[1].prev_id == "c_0000"
        assert loaded.chunks[0].text_sha256 == self.samples.digest("Первый")

    def test_t_sto_02_missing_directory(self, tmp_path: Path) -> None:
        """Нет каталога — load_chunks и load_chunks_meta возвращают None."""
        store = self._store(tmp_path / "absent")
        assert store.load_chunks("reglament.docx") is None
        assert store.load_chunks_meta("reglament.docx") is None

    def test_t_sto_03_source_sha256_in_meta(self, tmp_path: Path) -> None:
        """После save в meta.json есть source_sha256."""
        store = self._store(tmp_path)
        batch, meta = self.samples.batch()
        store.save_chunks(batch, meta)
        raw = json.loads(
            (tmp_path / "cache" / "emb" / batch.doc_id / "meta.json").read_text(
                encoding="utf-8"
            )
        )
        assert raw["source_sha256"] == "ab" * 32
        loaded = store.load_chunks_meta(batch.doc_id)
        assert loaded is not None
        assert loaded.source_sha256 == "ab" * 32

    def test_t_sto_04_vectors_roundtrip(self, tmp_path: Path) -> None:
        """Верный размер и хеши: load и verify ok, индекс на диске."""
        store = self._store(tmp_path)
        batch, meta = self.samples.batch()
        store.save_chunks(batch, meta)
        saved = store.load_chunks(batch.doc_id)
        assert saved is not None
        rows = [(chunk.chunk_id, chunk.text_sha256) for chunk in saved.chunks]
        dim = 4
        payload = self.samples.matrix(len(rows), dim)
        store.save_chunk_vectors(batch.doc_id, rows, dim, payload)
        loaded = store.load_chunk_vectors(batch.doc_id)
        assert loaded is not None
        assert loaded.chunk_ids == [rows[0][0]]
        assert loaded.text_sha256s == [rows[0][1]]
        assert loaded.dim == dim
        assert loaded.payload == payload
        report = store.verify_chunk_embeddings(batch.doc_id)
        assert report.ok is True
        index_path = (
            tmp_path / "cache" / "emb" / batch.doc_id / "vectors.index.jsonl"
        )
        assert index_path.is_file()
        meta_loaded = store.load_chunks_meta(batch.doc_id)
        assert meta_loaded is not None
        assert meta_loaded.vectors_index_sha256 is not None
        assert len(meta_loaded.vectors_index_sha256) == 64

    def test_t_sto_05_bad_payload_size(self, tmp_path: Path) -> None:
        """Неверный размер payload — исключение, файл векторов не публикуется."""
        store = self._store(tmp_path)
        batch, meta = self.samples.batch()
        store.save_chunks(batch, meta)
        saved = store.load_chunks(batch.doc_id)
        assert saved is not None
        rows = [(chunk.chunk_id, chunk.text_sha256) for chunk in saved.chunks]
        with pytest.raises(ValueError):
            store.save_chunk_vectors(batch.doc_id, rows, 4, b"\x00\x00")
        vector_path = tmp_path / "cache" / "emb" / batch.doc_id / "vectors.f16"
        assert not vector_path.exists()
        assert store.load_chunk_vectors(batch.doc_id) is None

    def test_t_sto_06_replace_chunks(self, tmp_path: Path) -> None:
        """Повторный save_chunks полностью заменяет старые чанки."""
        store = self._store(tmp_path)
        first, meta = self.samples.batch(texts=["Старый", "Ещё"])
        store.save_chunks(first, meta)
        second, second_meta = self.samples.batch(texts=["Новый"])
        store.save_chunks(second, second_meta)
        loaded = store.load_chunks(first.doc_id)
        assert loaded is not None
        assert [chunk.text for chunk in loaded.chunks] == ["Новый"]

    def test_t_sto_07_context_roundtrip(self, tmp_path: Path) -> None:
        """Бандл с empty_search=true проходит круг save/load."""
        store = self._store(tmp_path)
        bundle = ContextBundle(
            doc_id="reglament.docx",
            question_id="Q001",
            category="Срок хранения",
            used_synonyms=["retention"],
            empty_search=True,
            reason="no_relevant_chunks",
            candidates=[],
        )
        store.save_context(bundle)
        loaded = store.load_context(bundle.doc_id, bundle.question_id)
        assert loaded == bundle

    def test_t_sto_08_candidates_jsonl(self, tmp_path: Path) -> None:
        """Число строк candidates.jsonl равно числу кандидатов, скоры на месте."""
        store = self._store(tmp_path)
        batch = CandidateBatch(
            doc_id="reglament.docx",
            candidates=[
                CandidateRecord(
                    question_id="Q001",
                    chunk_id="c_0000",
                    vector_score=0.8,
                    bm25_score=0.2,
                    fused_score=0.6,
                    rank=1,
                    text="Срок",
                ),
                CandidateRecord(
                    question_id="Q001",
                    chunk_id="c_0001",
                    vector_score=0.4,
                    bm25_score=0.1,
                    fused_score=0.3,
                    rank=2,
                    text="Другое",
                ),
            ],
        )
        meta = ArtifactMeta(
            schema_version="1.0",
            chunking_hash="h",
            retrieval_config_hash="r",
            reranker_config_hash="k",
            source_sha256="ab" * 32,
        )
        store.save_candidates(batch, meta)
        path = tmp_path / "index" / batch.doc_id / "candidates.jsonl"
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["vector_score"] == 0.8
        assert first["bm25_score"] == 0.2
        assert first["fused_score"] == 0.6

    def test_t_sto_10_unsafe_doc_id(self, tmp_path: Path) -> None:
        """doc_id с / или .. — ValueError, файлы вне корня не создаются."""
        store = self._store(tmp_path)
        batch, meta = self.samples.batch(doc_id="../outside")
        with pytest.raises(ValueError):
            store.save_chunks(batch, meta)
        batch.doc_id = "nested/name.docx"
        with pytest.raises(ValueError):
            store.save_chunks(batch, meta)
        assert not (tmp_path / "outside").exists()
        assert not (tmp_path.parent / "outside").exists()
        assert not (tmp_path / "cache" / "emb").exists()

    def test_t_sto_11_tampered_text(self, tmp_path: Path) -> None:
        """Подмена text при том же chunk_id ломает verify и не зовёт поиск."""
        store = self._store(tmp_path)
        batch, meta = self.samples.batch()
        store.save_chunks(batch, meta)
        saved = store.load_chunks(batch.doc_id)
        assert saved is not None
        rows = [(chunk.chunk_id, chunk.text_sha256) for chunk in saved.chunks]
        store.save_chunk_vectors(batch.doc_id, rows, 2, self.samples.matrix(1, 2))
        chunks_path = tmp_path / "cache" / "emb" / batch.doc_id / "chunks.jsonl"
        row = json.loads(chunks_path.read_text(encoding="utf-8").splitlines()[0])
        row["text"] = "подменённый текст"
        chunks_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        with pytest.raises(EmbeddingBindingError) as caught:
            store.verify_chunk_embeddings(batch.doc_id)
        assert caught.value.report.ok is False
        assert caught.value.report.mismatched_hashes

    def test_t_sto_12_foreign_chunk_id(self, tmp_path: Path) -> None:
        """Чужой chunk_id в индексе даёт mismatched_ids."""
        store = self._store(tmp_path)
        batch, meta = self.samples.batch()
        store.save_chunks(batch, meta)
        saved = store.load_chunks(batch.doc_id)
        assert saved is not None
        rows = [(chunk.chunk_id, chunk.text_sha256) for chunk in saved.chunks]
        store.save_chunk_vectors(batch.doc_id, rows, 2, self.samples.matrix(1, 2))
        index_path = tmp_path / "cache" / "emb" / batch.doc_id / "vectors.index.jsonl"
        row = json.loads(index_path.read_text(encoding="utf-8").splitlines()[0])
        row["chunk_id"] = "c_foreign"
        index_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        with pytest.raises(EmbeddingBindingError) as caught:
            store.verify_chunk_embeddings(batch.doc_id)
        assert "c_foreign" in caught.value.report.mismatched_ids

    def test_t_sto_13_embed_cache_hit(self, tmp_path: Path) -> None:
        """Повторный embed того же текста даёт embed_cache_hit и файл кэша."""
        sink = RecordingTraceSink()
        store = self._store(tmp_path, sink)
        digest = self.samples.digest("один и тот же текст")
        key = store.embedding_cache_key("fake-model", "float16", digest)
        assert store.get_cached_embedding(key) is None
        payload = b"\x00\x3c\x00\x3c"
        store.put_cached_embedding(
            key,
            payload,
            text_sha256=digest,
            model="fake-model",
            dtype="float16",
            dim=2,
        )
        found = store.get_cached_embedding(key)
        assert found == payload
        hits = [
            event
            for event in sink.events
            if event.counters.get("embed_cache_hit", 0) >= 1
        ]
        assert hits
        assert (tmp_path / "cache" / "embed_cache" / f"{key}.f16").is_file()
        assert (tmp_path / "cache" / "embed_cache" / f"{key}.json").is_file()

    def test_t_sto_14_text_change_is_cache_miss_then_ok(self, tmp_path: Path) -> None:
        """Смена текста даёт новый хеш, промах кэша и успешный verify после save."""
        sink = RecordingTraceSink()
        store = self._store(tmp_path, sink)
        batch, meta = self.samples.batch(texts=["старый текст"])
        store.save_chunks(batch, meta)
        saved = store.load_chunks(batch.doc_id)
        assert saved is not None
        old_hash = saved.chunks[0].text_sha256
        rows = [(saved.chunks[0].chunk_id, old_hash)]
        store.save_chunk_vectors(batch.doc_id, rows, 2, self.samples.matrix(1, 2))
        old_key = store.embedding_cache_key("fake-model", "float16", old_hash)
        store.put_cached_embedding(
            old_key,
            self.samples.matrix(1, 2),
            text_sha256=old_hash,
            model="fake-model",
            dtype="float16",
            dim=2,
        )
        changed, changed_meta = self.samples.batch(texts=["новый текст"])
        new_hash = self.samples.digest("новый текст")
        assert new_hash != old_hash
        new_key = store.embedding_cache_key("fake-model", "float16", new_hash)
        assert store.get_cached_embedding(new_key) is None
        assert any(event.counters.get("embed_cache_miss", 0) >= 1 for event in sink.events)
        store.save_chunks(changed, changed_meta)
        reloaded = store.load_chunks(batch.doc_id)
        assert reloaded is not None
        assert reloaded.chunks[0].text_sha256 == new_hash
        new_rows = [(reloaded.chunks[0].chunk_id, new_hash)]
        store.save_chunk_vectors(batch.doc_id, new_rows, 2, self.samples.matrix(1, 2))
        report = store.verify_chunk_embeddings(batch.doc_id)
        assert report.ok is True

    def test_t_sto_15_verify_trace(self, tmp_path: Path) -> None:
        """Успешный verify пишет artifact_binding_verified / ok и n_chunks."""
        sink = RecordingTraceSink()
        store = self._store(tmp_path, sink)
        batch, meta = self.samples.batch()
        store.save_chunks(batch, meta)
        saved = store.load_chunks(batch.doc_id)
        assert saved is not None
        rows = [(chunk.chunk_id, chunk.text_sha256) for chunk in saved.chunks]
        store.save_chunk_vectors(batch.doc_id, rows, 2, self.samples.matrix(1, 2))
        matched = [
            event
            for event in sink.events
            if event.stage == "artifact_binding_verified" and event.status == "ok"
        ]
        assert matched
        assert matched[-1].counters.get("n_chunks") == 1

    def test_question_vectors_roundtrip(self, tmp_path: Path) -> None:
        """Кэш вопросов пишется и проходит ту же сверку, что чанки."""
        store = self._store(tmp_path)
        questions = QuestionSet(
            questions=[
                Question(id="Q001", category="Срок", synonyms=["retention"]),
            ]
        )
        store.save_questions_cache(
            questions,
            ArtifactMeta(schema_version="1.0", embedding_model="fake-model"),
        )
        assert store.has_questions_cache() is True
        raw = (tmp_path / "cache" / "questions.jsonl").read_text(encoding="utf-8")
        rows_json = [json.loads(line) for line in raw.splitlines() if line.strip()]
        pairs = [(row["item_id"], row["text_sha256"]) for row in rows_json]
        dim = 2
        store.save_question_vectors(pairs, dim, self.samples.matrix(len(pairs), dim))
        report = store.verify_question_embeddings()
        assert report.ok is True
        loaded = store.load_question_vectors()
        assert loaded is not None
        assert loaded.item_ids == [pair[0] for pair in pairs]

    def test_wrong_text_sha256_rejected(self, tmp_path: Path) -> None:
        """Переданный хеш, не совпадающий с текстом, — ValueError до записи."""
        store = self._store(tmp_path)
        batch, meta = self.samples.batch()
        batch.chunks[0].text_sha256 = "ab" * 32
        with pytest.raises(ValueError):
            store.save_chunks(batch, meta)
        assert store.load_chunks(batch.doc_id) is None

    def _store(
        self,
        tmp_path: Path,
        sink: RecordingTraceSink | None = None,
    ) -> FileArtifactStore:
        """Файловое хранилище во временном каталоге.

        Args:
            tmp_path: Корень теста.
            sink: Необязательный журнал событий.

        Returns:
            FileArtifactStore.
        """
        return FileArtifactStore(
            cache_dir=tmp_path / "cache",
            index_dir=tmp_path / "index",
            contexts_dir=tmp_path / "contexts",
            trace_sink=sink,
        )


class TestInMemoryArtifactStore:
    """T-STO-09: те же сценарии 01–03 и 06–07, что у файла."""

    def setup_method(self) -> None:
        """Готовит образцы и память."""
        self.samples = StoreSamples()
        self.store = InMemoryArtifactStore()

    def test_t_sto_09_chunks_meta_and_replace(self) -> None:
        """Память повторяет roundtrip, пустое чтение, source_sha256 и замену."""
        assert self.store.load_chunks("reglament.docx") is None
        assert self.store.load_chunks_meta("reglament.docx") is None
        batch, meta = self.samples.batch(texts=["Первый"])
        self.store.save_chunks(batch, meta)
        loaded = self.store.load_chunks(batch.doc_id)
        assert loaded is not None
        assert loaded.doc_id == batch.doc_id
        assert loaded.chunks[0].text == "Первый"
        assert loaded.chunks[0].text_sha256 == self.samples.digest("Первый")
        stored_meta = self.store.load_chunks_meta(batch.doc_id)
        assert stored_meta is not None
        assert stored_meta.source_sha256 == "ab" * 32
        replacement, replacement_meta = self.samples.batch(texts=["Только новый"])
        self.store.save_chunks(replacement, replacement_meta)
        replaced = self.store.load_chunks(batch.doc_id)
        assert replaced is not None
        assert [chunk.text for chunk in replaced.chunks] == ["Только новый"]

    def test_t_sto_09_context_roundtrip(self) -> None:
        """In-memory save/load бандла с empty_search совпадает с файлом."""
        bundle = ContextBundle(
            doc_id="reglament.docx",
            question_id="Q001",
            category="Срок",
            empty_search=True,
            reason="no_relevant_chunks",
            candidates=[],
        )
        self.store.save_context(bundle)
        assert self.store.load_context(bundle.doc_id, bundle.question_id) == bundle
