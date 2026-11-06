"""Общая логика ArtifactStore: ключи, связка, trace. Хранение байтов — у наследника."""

from __future__ import annotations

from abc import abstractmethod

from models.artifacts import (
    ArtifactMeta,
    ChunkVectorMatrix,
    ContextBundle,
    EmbeddingBindingReport,
    QuestionVectorMatrix,
)
from models.documents import CandidateBatch, ChunkBatch
from models.observe import TraceEvent
from models.questions import QuestionSet
from ports.artifacts import ArtifactStore
from ports.trace import TraceSink
from store.codec import ArtifactCodec
from store.errors import EmbeddingBindingError


class ArtifactTrace:
    """Пишет TraceEvent, если к хранилищу подключён TraceSink.

    Attributes:
        sink: Приёмник или None.
    """

    def __init__(self, sink: TraceSink | None) -> None:
        """Сохраняет приёмник.

        Args:
            sink: TraceSink или None, если события не нужны.
        """
        self.sink = sink

    def event(
        self,
        stage: str,
        status: str,
        doc_id: str | None = None,
        counters: dict[str, int | float] | None = None,
        error: str | None = None,
        debug_payload: str | None = None,
    ) -> None:
        """Отправляет одно событие.

        Args:
            stage: Имя стадии.
            status: started, ok или failed.
            doc_id: Документ, если есть.
            counters: Числовые счётчики.
            error: Текст ошибки.
            debug_payload: Короткий список id, не матрица.
        """
        if self.sink is None:
            return
        if status not in {"started", "ok", "failed"}:
            status = "failed"
        self.sink.emit(
            TraceEvent(
                stage=stage,
                status=status,  # type: ignore[arg-type]
                doc_id=doc_id,
                counters=counters or {},
                error=error,
                debug_payload=debug_payload,
            )
        )


class CheckingArtifactStore(ArtifactStore):
    """Правила записи и verify. Наследник хранит байты по логическим ключам.

    Attributes:
        codec: Сериализация и хеши.
        trace: События связки и кэша.
    """

    def __init__(self, trace_sink: TraceSink | None = None) -> None:
        """Создаёт кодек и приёмник событий.

        Args:
            trace_sink: TraceSink или None.
        """
        self.codec = ArtifactCodec()
        self.trace = ArtifactTrace(trace_sink)

    @abstractmethod
    def _read(self, key: str) -> bytes | None:
        """Читает байты ключа.

        Args:
            key: Логический ключ файла.

        Returns:
            Байты или None, если записи нет.
        """

    @abstractmethod
    def _write_many(self, items: dict[str, bytes]) -> None:
        """Атомарно заменяет набор ключей.

        Args:
            items: Ключ → байты.
        """

    def embedding_cache_key(self, model: str, dtype: str, text_sha256: str) -> str:
        """Ключ кэша эмбеддинга.

        Args:
            model: Имя модели.
            dtype: Тип данных.
            text_sha256: Хеш текста.

        Returns:
            Hex SHA-256.
        """
        return self.codec.cache_key(model, dtype, text_sha256)

    def save_chunks(self, batch: ChunkBatch, meta: ArtifactMeta) -> None:
        """Пишет chunks.jsonl и meta.json. Векторы не трогает.

        Args:
            batch: Пакет чанков.
            meta: Meta A1.

        Raises:
            ValueError: Небезопасный doc_id или чужой text_sha256.
        """
        doc_id = self.codec.safe_name(batch.doc_id)
        stamped = self.codec.stamp_batch(batch)
        stored = stamped.model_copy(update={"doc_id": doc_id})
        self._write_many(
            {
                self._chunks_key(doc_id): self.codec.chunks_jsonl(stored),
                self._chunks_meta_key(doc_id): self.codec.meta_json(meta),
            }
        )

    def load_chunks(self, doc_id: str) -> ChunkBatch | None:
        """Читает чанки.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            ChunkBatch или None.
        """
        safe_id = self.codec.safe_name(doc_id)
        payload = self._read(self._chunks_key(safe_id))
        if payload is None:
            return None
        meta = self._read_chunk_meta(safe_id)
        return self.codec.chunks_from_jsonl(safe_id, payload, meta)

    def load_chunks_meta(self, doc_id: str) -> ArtifactMeta | None:
        """Читает meta чанков.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            ArtifactMeta или None.
        """
        safe_id = self.codec.safe_name(doc_id)
        return self._read_chunk_meta(safe_id)

    def save_chunk_vectors(
        self,
        doc_id: str,
        rows: list[tuple[str, str]],
        dim: int,
        payload: bytes,
    ) -> None:
        """Пишет vectors.f16 и индекс, затем verify.

        Args:
            doc_id: Идентификатор документа.
            rows: Пары (chunk_id, text_sha256) в порядке рядов.
            dim: Размерность.
            payload: Little-endian float16.

        Raises:
            ValueError: Нет чанков, пары или размер не сходятся.
            EmbeddingBindingError: Запись не прошла verify.
        """
        safe_id = self.codec.safe_name(doc_id)
        batch = self.load_chunks(safe_id)
        if batch is None:
            self._emit_vectors(
                "artifact_vectors_saved",
                safe_id,
                len(rows),
                dim,
                "нет chunks.jsonl",
            )
            raise ValueError("нет chunks.jsonl для записи векторов")
        expected = self.codec.vector_rows(batch)
        if list(rows) != expected:
            self._emit_vectors(
                "artifact_vectors_saved",
                safe_id,
                len(rows),
                dim,
                "пары не совпадают с чанками",
            )
            raise ValueError(
                "пары (chunk_id, text_sha256) не совпадают с chunks.jsonl"
            )
        expected_size = len(rows) * dim * 2
        if dim < 1 or len(payload) != expected_size:
            self._emit_vectors(
                "artifact_vectors_saved",
                safe_id,
                len(rows),
                dim,
                "неверный размер payload",
            )
            raise ValueError("неверный размер payload векторов")
        index_bytes = self.codec.index_jsonl(list(rows), "chunk_id")
        index_hash = self.codec.file_sha256(index_bytes)
        current_meta = self._read_chunk_meta(safe_id) or ArtifactMeta(doc_id=safe_id)
        updated = current_meta.model_copy(
            update={
                "doc_id": safe_id,
                "n_chunks": len(batch.chunks),
                "embedding_dim": dim,
                "embedding_dtype": "float16",
                "vectors_index_sha256": index_hash,
            }
        )
        self._write_many(
            {
                self._vectors_key(safe_id): payload,
                self._vectors_index_key(safe_id): index_bytes,
                self._chunks_meta_key(safe_id): self.codec.meta_json(updated),
            }
        )
        self._emit_vectors("artifact_vectors_saved", safe_id, len(rows), dim, None)
        self.verify_chunk_embeddings(safe_id)

    def verify_chunk_embeddings(self, doc_id: str) -> EmbeddingBindingReport:
        """Сверяет чанки, индекс и размер f16.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            Отчёт с ok=true.

        Raises:
            EmbeddingBindingError: Связка не сошлась.
        """
        safe_id = self.codec.safe_name(doc_id)
        report = self._chunk_report(safe_id)
        self._emit_binding("artifact_binding_verified", safe_id, report)
        if not report.ok:
            raise EmbeddingBindingError(report)
        return report

    def load_chunk_vectors(self, doc_id: str) -> ChunkVectorMatrix | None:
        """Читает матрицу чанков после verify.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            Оболочка с ids и payload или None, если векторов нет.

        Raises:
            EmbeddingBindingError: Файлы есть, но связка сломана.
        """
        safe_id = self.codec.safe_name(doc_id)
        payload = self._read(self._vectors_key(safe_id))
        index_payload = self._read(self._vectors_index_key(safe_id))
        if payload is None and index_payload is None:
            return None
        self.verify_chunk_embeddings(safe_id)
        rows = self.codec.parse_index(index_payload or b"")
        meta = self._read_chunk_meta(safe_id)
        dim = meta.embedding_dim if meta is not None and meta.embedding_dim else 0
        return ChunkVectorMatrix(
            chunk_ids=[item_id for item_id, _digest in rows],
            text_sha256s=[digest for _item_id, digest in rows],
            dim=dim,
            payload=payload or b"",
        )

    def get_cached_embedding(self, key: str) -> bytes | None:
        """Ищет один вектор в embed_cache и пишет hit/miss.

        Args:
            key: Ключ из embedding_cache_key.

        Returns:
            Байты вектора или None.
        """
        safe_key = self._safe_cache_key(key)
        payload = self._read(self._embed_vector_key(safe_key))
        if payload is None:
            self.trace.event(
                "embedding",
                "ok",
                counters={"embed_cache_miss": 1},
            )
            return None
        self.trace.event(
            "embedding",
            "ok",
            counters={"embed_cache_hit": 1},
        )
        return payload

    def put_cached_embedding(
        self,
        key: str,
        payload: bytes,
        text_sha256: str,
        model: str,
        dtype: str,
        dim: int,
    ) -> None:
        """Кладёт вектор и его паспорт в embed_cache.

        Args:
            key: Ключ кэша.
            payload: Один вектор float16.
            text_sha256: Хеш текста.
            model: Имя модели.
            dtype: Тип данных.
            dim: Размерность.
        """
        safe_key = self._safe_cache_key(key)
        record = self.codec.new_cache_record(text_sha256, model, dtype, dim)
        self._write_many(
            {
                self._embed_vector_key(safe_key): payload,
                self._embed_meta_key(safe_key): self.codec.cache_meta_json(record),
            }
        )

    def save_candidates(self, batch: CandidateBatch, meta: ArtifactMeta) -> None:
        """Пишет candidates.jsonl и meta A2.

        Args:
            batch: Кандидаты.
            meta: Meta retrieval.
        """
        doc_id = self.codec.safe_name(batch.doc_id)
        stored = batch.model_copy(update={"doc_id": doc_id})
        self._write_many(
            {
                self._candidates_key(doc_id): self.codec.candidates_jsonl(stored),
                self._candidates_meta_key(doc_id): self.codec.meta_json(meta),
            }
        )

    def load_candidates(self, doc_id: str) -> CandidateBatch | None:
        """Читает кандидатов.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            CandidateBatch или None.
        """
        safe_id = self.codec.safe_name(doc_id)
        payload = self._read(self._candidates_key(safe_id))
        if payload is None:
            return None
        meta = self.load_candidates_meta(safe_id)
        return self.codec.candidates_from_jsonl(safe_id, payload, meta)

    def load_candidates_meta(self, doc_id: str) -> ArtifactMeta | None:
        """Читает meta кандидатов.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            ArtifactMeta или None.
        """
        safe_id = self.codec.safe_name(doc_id)
        payload = self._read(self._candidates_meta_key(safe_id))
        if payload is None:
            return None
        return self.codec.meta_from_json(payload)

    def save_context(self, bundle: ContextBundle) -> None:
        """Пишет JSON бандла.

        Args:
            bundle: ContextBundle.
        """
        doc_id = self.codec.safe_name(bundle.doc_id)
        question_id = self.codec.safe_name(bundle.question_id)
        stored = bundle.model_copy(
            update={"doc_id": doc_id, "question_id": question_id}
        )
        self._write_many(
            {
                self._context_key(doc_id, question_id): self.codec.context_json(
                    stored
                )
            }
        )

    def load_context(self, doc_id: str, question_id: str) -> ContextBundle | None:
        """Читает бандл.

        Args:
            doc_id: Идентификатор документа.
            question_id: Идентификатор вопроса.

        Returns:
            ContextBundle или None.
        """
        safe_doc = self.codec.safe_name(doc_id)
        safe_question = self.codec.safe_name(question_id)
        payload = self._read(self._context_key(safe_doc, safe_question))
        if payload is None:
            return None
        return self.codec.context_from_json(payload)

    def save_questions_cache(self, questions: QuestionSet, meta: ArtifactMeta) -> None:
        """Пишет questions.jsonl и questions.meta.json.

        Args:
            questions: Набор вопросов.
            meta: Meta кэша.
        """
        records = self.codec.question_records(questions)
        self._write_many(
            {
                self._questions_key(): self.codec.questions_jsonl(records),
                self._questions_meta_key(): self.codec.meta_json(meta),
            }
        )

    def load_questions_meta(self) -> ArtifactMeta | None:
        """Читает meta кэша вопросов.

        Returns:
            ArtifactMeta или None.
        """
        payload = self._read(self._questions_meta_key())
        if payload is None:
            return None
        return self.codec.meta_from_json(payload)

    def has_questions_cache(self) -> bool:
        """Проверяет наличие questions.jsonl.

        Returns:
            True, если файл кэша записан.
        """
        return self._read(self._questions_key()) is not None

    def save_question_vectors(
        self,
        rows: list[tuple[str, str]],
        dim: int,
        payload: bytes,
    ) -> None:
        """Пишет questions.f16 и индекс, затем verify.

        Args:
            rows: Пары (item_id, text_sha256).
            dim: Размерность.
            payload: Little-endian float16.

        Raises:
            ValueError: Нет кэша вопросов или размер неверен.
            EmbeddingBindingError: Связка не сошлась.
        """
        raw_questions = self._read(self._questions_key())
        if raw_questions is None:
            self._emit_vectors(
                "question_vectors_saved",
                None,
                len(rows),
                dim,
                "нет questions.jsonl",
            )
            raise ValueError("нет questions.jsonl для записи векторов")
        records = self.codec.questions_from_jsonl(raw_questions)
        expected = self.codec.question_rows(records)
        if list(rows) != expected:
            self._emit_vectors(
                "question_vectors_saved",
                None,
                len(rows),
                dim,
                "пары не совпадают с вопросами",
            )
            raise ValueError("пары вопросов не совпадают с questions.jsonl")
        expected_size = len(rows) * dim * 2
        if dim < 1 or len(payload) != expected_size:
            self._emit_vectors(
                "question_vectors_saved",
                None,
                len(rows),
                dim,
                "неверный размер payload",
            )
            raise ValueError("неверный размер payload векторов вопросов")
        index_bytes = self.codec.index_jsonl(list(rows), "item_id")
        index_hash = self.codec.file_sha256(index_bytes)
        current = self.load_questions_meta() or ArtifactMeta()
        updated = current.model_copy(
            update={
                "n_chunks": len(records),
                "embedding_dim": dim,
                "embedding_dtype": "float16",
                "vectors_index_sha256": index_hash,
            }
        )
        self._write_many(
            {
                self._question_vectors_key(): payload,
                self._question_index_key(): index_bytes,
                self._questions_meta_key(): self.codec.meta_json(updated),
            }
        )
        self._emit_vectors("question_vectors_saved", None, len(rows), dim, None)
        self.verify_question_embeddings()

    def verify_question_embeddings(self) -> EmbeddingBindingReport:
        """Сверяет кэш вопросов с матрицей.

        Returns:
            Отчёт с ok=true.

        Raises:
            EmbeddingBindingError: Связка не сошлась.
        """
        report = self._question_report()
        self._emit_binding("question_binding_verified", None, report)
        if not report.ok:
            raise EmbeddingBindingError(report)
        return report

    def load_question_vectors(self) -> QuestionVectorMatrix | None:
        """Читает матрицу вопросов после verify.

        Returns:
            Оболочка или None, если векторов нет.

        Raises:
            EmbeddingBindingError: Файлы есть, но связка сломана.
        """
        payload = self._read(self._question_vectors_key())
        index_payload = self._read(self._question_index_key())
        if payload is None and index_payload is None:
            return None
        self.verify_question_embeddings()
        rows = self.codec.parse_index(index_payload or b"")
        meta = self.load_questions_meta()
        dim = meta.embedding_dim if meta is not None and meta.embedding_dim else 0
        return QuestionVectorMatrix(
            item_ids=[item_id for item_id, _digest in rows],
            text_sha256s=[digest for _item_id, digest in rows],
            dim=dim,
            payload=payload or b"",
        )

    def save_answers_meta(self, doc_id: str, meta: ArtifactMeta) -> None:
        """Пишет meta ответов документа.

        Args:
            doc_id: Идентификатор документа.
            meta: Upstream-хеши фазы B.
        """
        safe_id = self.codec.safe_name(doc_id)
        self._write_many(
            {self._answers_meta_key(safe_id): self.codec.meta_json(meta)}
        )

    def load_answers_meta(self, doc_id: str) -> ArtifactMeta | None:
        """Читает meta ответов.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            ArtifactMeta или None.
        """
        safe_id = self.codec.safe_name(doc_id)
        payload = self._read(self._answers_meta_key(safe_id))
        if payload is None:
            return None
        return self.codec.meta_from_json(payload)

    def _chunk_report(self, doc_id: str) -> EmbeddingBindingReport:
        """Собирает отчёт по файлам одного документа.

        Args:
            doc_id: Безопасный идентификатор.

        Returns:
            EmbeddingBindingReport, в том числе с ok=false.
        """
        raw_chunks = self._read(self._chunks_key(doc_id))
        raw_index = self._read(self._vectors_index_key(doc_id))
        payload = self._read(self._vectors_key(doc_id))
        meta = self._read_chunk_meta(doc_id)
        if raw_chunks is None:
            return EmbeddingBindingReport(
                ok=False,
                n_chunks=0,
                n_vector_rows=0,
                size_error="нет chunks.jsonl",
            )
        batch = self.codec.chunks_from_jsonl(doc_id, raw_chunks, meta)
        records = [
            (chunk.chunk_id, chunk.text_sha256, chunk.text)
            for chunk in sorted(batch.chunks, key=lambda item: item.chunk_index)
        ]
        index_rows = [] if raw_index is None else self.codec.parse_index(raw_index)
        dim = self._dim_from_meta(meta, len(records), payload)
        return self.codec.build_report(records, index_rows, payload, dim)

    def _question_report(self) -> EmbeddingBindingReport:
        """Собирает отчёт по кэшу вопросов.

        Returns:
            EmbeddingBindingReport.
        """
        raw_questions = self._read(self._questions_key())
        raw_index = self._read(self._question_index_key())
        payload = self._read(self._question_vectors_key())
        meta = self.load_questions_meta()
        if raw_questions is None:
            return EmbeddingBindingReport(
                ok=False,
                n_chunks=0,
                n_vector_rows=0,
                size_error="нет questions.jsonl",
            )
        records_model = self.codec.questions_from_jsonl(raw_questions)
        records = [
            (record.item_id, record.text_sha256, record.text)
            for record in records_model
        ]
        index_rows = [] if raw_index is None else self.codec.parse_index(raw_index)
        dim = self._dim_from_meta(meta, len(records), payload)
        return self.codec.build_report(records, index_rows, payload, dim)

    def _dim_from_meta(
        self,
        meta: ArtifactMeta | None,
        record_count: int,
        payload: bytes | None,
    ) -> int:
        """Берёт размерность из meta или из длины матрицы.

        Args:
            meta: Meta с embedding_dim.
            record_count: Число строк.
            payload: Байты матрицы.

        Returns:
            Положительная размерность или 0, если восстановить нельзя.
        """
        if meta is not None and meta.embedding_dim:
            return meta.embedding_dim
        if payload is None or record_count < 1:
            return 0
        width = record_count * 2
        if len(payload) % width != 0:
            return 0
        return len(payload) // width

    def _read_chunk_meta(self, doc_id: str) -> ArtifactMeta | None:
        """Читает meta.json чанков.

        Args:
            doc_id: Безопасный идентификатор.

        Returns:
            ArtifactMeta или None.
        """
        payload = self._read(self._chunks_meta_key(doc_id))
        if payload is None:
            return None
        return self.codec.meta_from_json(payload)

    def _emit_vectors(
        self,
        stage: str,
        doc_id: str | None,
        n_chunks: int,
        dim: int,
        error: str | None,
    ) -> None:
        """Пишет событие сохранения векторов.

        Args:
            stage: artifact_vectors_saved или question_vectors_saved.
            doc_id: Документ или None для вопросов.
            n_chunks: Число рядов.
            dim: Размерность.
            error: Текст при неудаче; None — успех.
        """
        status = "failed" if error else "ok"
        self.trace.event(
            stage,
            status,
            doc_id=doc_id,
            counters={"n_chunks": n_chunks, "dim": dim},
            error=error,
        )

    def _emit_binding(
        self,
        stage: str,
        doc_id: str | None,
        report: EmbeddingBindingReport,
    ) -> None:
        """Пишет событие verify.

        Args:
            stage: Имя стадии связки.
            doc_id: Документ или None.
            report: Отчёт сверки.
        """
        if report.ok:
            self.trace.event(
                stage,
                "ok",
                doc_id=doc_id,
                counters={
                    "n_chunks": report.n_chunks,
                    "matched": report.n_chunks,
                },
            )
            return
        preview_ids = (report.mismatched_hashes + report.mismatched_ids)[:5]
        preview = ",".join(preview_ids) if preview_ids else None
        self.trace.event(
            stage,
            "failed",
            doc_id=doc_id,
            counters={
                "n_chunks": report.n_chunks,
                "n_mismatched_hashes": len(report.mismatched_hashes),
                "n_mismatched_ids": len(report.mismatched_ids),
            },
            error=report.size_error or "рассинхрон текста и вектора",
            debug_payload=preview,
        )

    def _safe_cache_key(self, key: str) -> str:
        """Не даёт ключу кэша выйти из каталога embed_cache.

        Args:
            key: Предложенный ключ.

        Returns:
            Тот же ключ.

        Raises:
            ValueError: В ключе есть разделитель пути.
        """
        if key == "" or "/" in key or "\\" in key or ".." in key:
            raise ValueError("небезопасный ключ кэша эмбеддингов")
        return key

    def _chunks_key(self, doc_id: str) -> str:
        """Ключ chunks.jsonl.

        Args:
            doc_id: Безопасный идентификатор.

        Returns:
            Логический ключ.
        """
        return f"cache/emb/{doc_id}/chunks.jsonl"

    def _chunks_meta_key(self, doc_id: str) -> str:
        """Ключ meta.json чанков.

        Args:
            doc_id: Безопасный идентификатор.

        Returns:
            Логический ключ.
        """
        return f"cache/emb/{doc_id}/meta.json"

    def _vectors_key(self, doc_id: str) -> str:
        """Ключ vectors.f16.

        Args:
            doc_id: Безопасный идентификатор.

        Returns:
            Логический ключ.
        """
        return f"cache/emb/{doc_id}/vectors.f16"

    def _vectors_index_key(self, doc_id: str) -> str:
        """Ключ vectors.index.jsonl.

        Args:
            doc_id: Безопасный идентификатор.

        Returns:
            Логический ключ.
        """
        return f"cache/emb/{doc_id}/vectors.index.jsonl"

    def _candidates_key(self, doc_id: str) -> str:
        """Ключ candidates.jsonl.

        Args:
            doc_id: Безопасный идентификатор.

        Returns:
            Логический ключ.
        """
        return f"index/{doc_id}/candidates.jsonl"

    def _candidates_meta_key(self, doc_id: str) -> str:
        """Ключ meta кандидатов.

        Args:
            doc_id: Безопасный идентификатор.

        Returns:
            Логический ключ.
        """
        return f"index/{doc_id}/meta.json"

    def _context_key(self, doc_id: str, question_id: str) -> str:
        """Ключ JSON бандла.

        Args:
            doc_id: Безопасный идентификатор документа.
            question_id: Безопасный идентификатор вопроса.

        Returns:
            Логический ключ.
        """
        return f"contexts/{doc_id}/{question_id}.json"

    def _questions_key(self) -> str:
        """Ключ questions.jsonl.

        Returns:
            Логический ключ.
        """
        return "cache/questions.jsonl"

    def _questions_meta_key(self) -> str:
        """Ключ questions.meta.json.

        Returns:
            Логический ключ.
        """
        return "cache/questions.meta.json"

    def _question_vectors_key(self) -> str:
        """Ключ questions.f16.

        Returns:
            Логический ключ.
        """
        return "cache/questions.f16"

    def _question_index_key(self) -> str:
        """Ключ questions.index.jsonl.

        Returns:
            Логический ключ.
        """
        return "cache/questions.index.jsonl"

    def _embed_vector_key(self, key: str) -> str:
        """Ключ файла вектора в embed_cache.

        Args:
            key: Безопасный ключ кэша.

        Returns:
            Логический ключ.
        """
        return f"cache/embed_cache/{key}.f16"

    def _embed_meta_key(self, key: str) -> str:
        """Ключ паспорта вектора в embed_cache.

        Args:
            key: Безопасный ключ кэша.

        Returns:
            Логический ключ.
        """
        return f"cache/embed_cache/{key}.json"

    def _answers_meta_key(self, doc_id: str) -> str:
        """Ключ meta ответов.

        Args:
            doc_id: Безопасный идентификатор.

        Returns:
            Логический ключ.
        """
        return f"cache/answers/{doc_id}/meta.json"
