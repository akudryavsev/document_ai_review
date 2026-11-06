"""Сериализация артефактов и сверка text_sha256."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from models.artifacts import (
    ArtifactMeta,
    CachedEmbeddingRecord,
    ContextBundle,
    EmbeddingBindingReport,
    QuestionVariantRecord,
)
from models.documents import CandidateBatch, CandidateRecord, ChunkBatch, ChunkRecord
from models.questions import QuestionSet


class ArtifactCodec:
    """Кодирует pydantic-модели артефактов в байты и проверяет связку."""

    def text_sha256(self, text: str) -> str:
        """Считает SHA-256 UTF-8 текста без дополнительной нормализации.

        Args:
            text: Текст, как он ушёл в эмбеддер.

        Returns:
            Нижний регистр hex длиной 64.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return digest

    def file_sha256(self, payload: bytes) -> str:
        """Считает SHA-256 сырых байтов файла.

        Args:
            payload: Содержимое файла.

        Returns:
            Нижний регистр hex.
        """
        return hashlib.sha256(payload).hexdigest()

    def cache_key(self, model: str, dtype: str, text_sha256: str) -> str:
        """Строит ключ embed_cache из модели, типа и хеша текста.

        Args:
            model: Имя модели эмбеддера.
            dtype: Тип данных, как в конфиге.
            text_sha256: Хеш уже финального текста.

        Returns:
            SHA-256 строки model|dtype|text_sha256.
        """
        material = f"{model}|{dtype}|{text_sha256}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def safe_name(self, value: str) -> str:
        """Проверяет, что идентификатор можно положить в имя файла.

        Args:
            value: doc_id или question_id.

        Returns:
            То же имя, если оно безопасно.

        Raises:
            ValueError: В имени есть путь, `..` или оно пустое.
        """
        if value == "" or value in {".", ".."}:
            raise ValueError(f"небезопасный идентификатор: {value!r}")
        if "/" in value or "\\" in value or ".." in value:
            raise ValueError(f"небезопасный идентификатор: {value!r}")
        return value

    def stamp_batch(self, batch: ChunkBatch) -> ChunkBatch:
        """Проставляет text_sha256 чанкам, не переписывая чужой хеш.

        Args:
            batch: Пакет чанков.

        Returns:
            Копия пакета с заполненными хешами.

        Raises:
            ValueError: Переданный хеш не совпадает с текстом.
        """
        stamped: list[ChunkRecord] = []
        for chunk in batch.chunks:
            digest = self.text_sha256(chunk.text)
            if chunk.text_sha256 != "" and chunk.text_sha256 != digest:
                raise ValueError(
                    f"text_sha256 не совпадает с текстом чанка {chunk.chunk_id}"
                )
            stamped.append(chunk.model_copy(update={"text_sha256": digest}))
        return batch.model_copy(update={"chunks": stamped})

    def vector_rows(self, batch: ChunkBatch) -> list[tuple[str, str]]:
        """Возвращает пары (chunk_id, text_sha256) в порядке chunk_index.

        Args:
            batch: Пакет с уже посчитанными хешами.

        Returns:
            Список пар для save_chunk_vectors.
        """
        ordered = sorted(batch.chunks, key=lambda chunk: chunk.chunk_index)
        rows = [(chunk.chunk_id, chunk.text_sha256) for chunk in ordered]
        return rows

    def chunks_jsonl(self, batch: ChunkBatch) -> bytes:
        """Сериализует чанки в JSONL без векторов.

        Args:
            batch: Пакет чанков.

        Returns:
            UTF-8 байты, по объекту на строку, по возрастанию chunk_index.
        """
        ordered = sorted(batch.chunks, key=lambda chunk: chunk.chunk_index)
        lines: list[str] = []
        for chunk in ordered:
            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "text_sha256": chunk.text_sha256,
                "page_from": chunk.page_from,
                "page_to": chunk.page_to,
                "section_path": chunk.section_path,
                "chunk_index": chunk.chunk_index,
                "prev_id": chunk.prev_id,
                "next_id": chunk.next_id,
            }
            lines.append(json.dumps(payload, ensure_ascii=False))
        return self._lines_bytes(lines)

    def chunks_from_jsonl(
        self,
        doc_id: str,
        payload: bytes,
        meta: ArtifactMeta | None,
    ) -> ChunkBatch:
        """Читает ChunkBatch из JSONL.

        Args:
            doc_id: Идентификатор документа.
            payload: Содержимое chunks.jsonl.
            meta: Meta A1, если она есть.

        Returns:
            ChunkBatch. Хеши конфигурации берутся из meta.
        """
        chunks: list[ChunkRecord] = []
        for row in self._json_lines(payload):
            chunks.append(ChunkRecord.model_validate(row))
        chunks.sort(key=lambda chunk: chunk.chunk_index)
        chunking_hash = ""
        embedding_model = None
        if meta is not None:
            chunking_hash = meta.chunking_hash or ""
            embedding_model = meta.embedding_model
        return ChunkBatch(
            doc_id=doc_id,
            chunks=chunks,
            chunking_hash=chunking_hash,
            embedding_model=embedding_model,
        )

    def meta_json(self, meta: ArtifactMeta) -> bytes:
        """Сериализует meta.json.

        Args:
            meta: Метаданные артефакта.

        Returns:
            UTF-8 JSON с отступом 2.
        """
        text = json.dumps(
            meta.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        return (text + "\n").encode("utf-8")

    def meta_from_json(self, payload: bytes) -> ArtifactMeta:
        """Читает ArtifactMeta.

        Args:
            payload: Содержимое meta.json.

        Returns:
            ArtifactMeta.
        """
        return ArtifactMeta.model_validate_json(payload)

    def index_jsonl(self, rows: list[tuple[str, str]], id_field: str) -> bytes:
        """Сериализует индекс рядов векторов.

        Args:
            rows: Пары (id, text_sha256) в порядке матрицы.
            id_field: Имя поля id: chunk_id или item_id.

        Returns:
            UTF-8 JSONL.
        """
        lines: list[str] = []
        for row_index, (item_id, digest) in enumerate(rows):
            payload = {
                "row": row_index,
                id_field: item_id,
                "text_sha256": digest,
            }
            lines.append(json.dumps(payload, ensure_ascii=False))
        return self._lines_bytes(lines)

    def parse_index(self, payload: bytes) -> list[tuple[str, str]]:
        """Читает пары (id, text_sha256) из индекса.

        Args:
            payload: Содержимое *.index.jsonl.

        Returns:
            Пары в порядке строк файла.
        """
        rows: list[tuple[str, str]] = []
        for row in self._json_lines(payload):
            item_id = row.get("chunk_id")
            if item_id is None:
                item_id = row.get("item_id")
            digest = row.get("text_sha256")
            if not isinstance(item_id, str) or not isinstance(digest, str):
                raise ValueError("строка индекса без id или text_sha256")
            rows.append((item_id, digest))
        return rows

    def question_records(self, questions: QuestionSet) -> list[QuestionVariantRecord]:
        """Разворачивает вопросы в варианты category и synonyms.

        Args:
            questions: Набор вопросов прогона.

        Returns:
            Записи questions.jsonl с посчитанным text_sha256.
        """
        records: list[QuestionVariantRecord] = []
        for question in questions.questions:
            category_record = QuestionVariantRecord(
                question_id=question.id,
                variant="category",
                text=question.category,
                text_sha256=self.text_sha256(question.category),
                item_id=f"{question.id}:category",
            )
            records.append(category_record)
            for index, synonym in enumerate(question.synonyms):
                synonym_record = QuestionVariantRecord(
                    question_id=question.id,
                    variant=f"synonym:{index}",
                    text=synonym,
                    text_sha256=self.text_sha256(synonym),
                    item_id=f"{question.id}:syn:{index}",
                )
                records.append(synonym_record)
        return records

    def questions_jsonl(self, records: list[QuestionVariantRecord]) -> bytes:
        """Сериализует кэш вопросов.

        Args:
            records: Варианты вопросов.

        Returns:
            UTF-8 JSONL.
        """
        lines: list[str] = []
        for record in records:
            lines.append(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
            )
        return self._lines_bytes(lines)

    def questions_from_jsonl(self, payload: bytes) -> list[QuestionVariantRecord]:
        """Читает варианты вопросов.

        Args:
            payload: Содержимое questions.jsonl.

        Returns:
            Список QuestionVariantRecord.
        """
        records: list[QuestionVariantRecord] = []
        for row in self._json_lines(payload):
            records.append(QuestionVariantRecord.model_validate(row))
        return records

    def question_rows(
        self,
        records: list[QuestionVariantRecord],
    ) -> list[tuple[str, str]]:
        """Возвращает пары (item_id, text_sha256) вариантов.

        Args:
            records: Записи кэша вопросов.

        Returns:
            Пары в порядке файла.
        """
        return [(record.item_id, record.text_sha256) for record in records]

    def candidates_jsonl(self, batch: CandidateBatch) -> bytes:
        """Сериализует кандидатов.

        Args:
            batch: Пакет кандидатов.

        Returns:
            UTF-8 JSONL.
        """
        lines: list[str] = []
        for candidate in batch.candidates:
            payload = candidate.model_dump(mode="json")
            lines.append(json.dumps(payload, ensure_ascii=False))
        return self._lines_bytes(lines)

    def candidates_from_jsonl(
        self,
        doc_id: str,
        payload: bytes,
        meta: ArtifactMeta | None,
    ) -> CandidateBatch:
        """Читает CandidateBatch.

        Args:
            doc_id: Идентификатор документа.
            payload: Содержимое candidates.jsonl.
            meta: Meta A2, если есть.

        Returns:
            CandidateBatch.
        """
        candidates: list[CandidateRecord] = []
        for row in self._json_lines(payload):
            candidates.append(CandidateRecord.model_validate(row))
        retrieval_hash = ""
        reranker_hash = ""
        chunking_hash = ""
        if meta is not None:
            retrieval_hash = meta.retrieval_config_hash or ""
            reranker_hash = meta.reranker_config_hash or ""
            chunking_hash = meta.chunking_hash or ""
        return CandidateBatch(
            doc_id=doc_id,
            candidates=candidates,
            retrieval_config_hash=retrieval_hash,
            reranker_config_hash=reranker_hash,
            chunking_hash=chunking_hash,
        )

    def context_json(self, bundle: ContextBundle) -> bytes:
        """Сериализует ContextBundle.

        Args:
            bundle: Бандл пары документ × вопрос.

        Returns:
            UTF-8 JSON с отступом 2.
        """
        text = json.dumps(
            bundle.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        return (text + "\n").encode("utf-8")

    def context_from_json(self, payload: bytes) -> ContextBundle:
        """Читает ContextBundle.

        Args:
            payload: Содержимое JSON-файла.

        Returns:
            ContextBundle.
        """
        return ContextBundle.model_validate_json(payload)

    def cache_meta_json(self, record: CachedEmbeddingRecord) -> bytes:
        """Сериализует паспорт вектора в embed_cache.

        Args:
            record: Паспорт без самих байтов вектора.

        Returns:
            UTF-8 JSON.
        """
        text = json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        return (text + "\n").encode("utf-8")

    def new_cache_record(
        self,
        text_sha256: str,
        model: str,
        dtype: str,
        dim: int,
    ) -> CachedEmbeddingRecord:
        """Собирает паспорт кэша с текущим временем.

        Args:
            text_sha256: Хеш текста.
            model: Имя модели.
            dtype: Тип данных.
            dim: Размерность.

        Returns:
            CachedEmbeddingRecord.
        """
        created_at = datetime.now(timezone.utc).isoformat()
        return CachedEmbeddingRecord(
            text_sha256=text_sha256,
            model=model,
            dtype=dtype,
            dim=dim,
            created_at=created_at,
        )

    def build_report(
        self,
        records: list[tuple[str, str, str]],
        index_rows: list[tuple[str, str]],
        payload: bytes | None,
        dim: int,
    ) -> EmbeddingBindingReport:
        """Сверяет тексты, индекс и размер матрицы.

        Args:
            records: Тройки (id, сохранённый хеш, текст).
            index_rows: Пары (id, text_sha256) из индекса векторов.
            payload: Байты float16 или None, если файла нет.
            dim: Размерность из meta.

        Returns:
            Отчёт. ok=false, если есть расхождение или неверный размер.
        """
        record_ids = [item_id for item_id, _stored, _text in records]
        index_ids = [item_id for item_id, _digest in index_rows]
        mismatched_ids = self._id_mismatches(record_ids, index_ids)
        mismatched_hashes = self._hash_mismatches(records, index_rows)
        size_error = self._size_error(
            record_count=len(records),
            index_count=len(index_rows),
            payload=payload,
            dim=dim,
        )
        ok = (
            size_error is None
            and not mismatched_ids
            and not mismatched_hashes
        )
        return EmbeddingBindingReport(
            ok=ok,
            n_chunks=len(records),
            n_vector_rows=len(index_rows),
            mismatched_ids=mismatched_ids,
            mismatched_hashes=mismatched_hashes,
            size_error=size_error,
        )

    def _id_mismatches(
        self,
        record_ids: list[str],
        index_ids: list[str],
    ) -> list[str]:
        """Ищет идентификаторы, которые есть только в одной из сторон.

        Args:
            record_ids: Id текстовых записей.
            index_ids: Id строк индекса.

        Returns:
            Список id без повторов в порядке обнаружения.
        """
        record_set = set(record_ids)
        index_set = set(index_ids)
        mismatched: list[str] = []
        for item_id in index_ids:
            if item_id not in record_set and item_id not in mismatched:
                mismatched.append(item_id)
        for item_id in record_ids:
            if item_id not in index_set and item_id not in mismatched:
                mismatched.append(item_id)
        return mismatched

    def _hash_mismatches(
        self,
        records: list[tuple[str, str, str]],
        index_rows: list[tuple[str, str]],
    ) -> list[str]:
        """Сравнивает хеш текста с хешем, на котором лежит вектор.

        Args:
            records: Тройки (id, сохранённый хеш, текст).
            index_rows: Пары индекса.

        Returns:
            Id, у которых text_sha256 не сходится.
        """
        index_by_id = {item_id: digest for item_id, digest in index_rows}
        mismatched: list[str] = []
        for item_id, stored_hash, text in records:
            if item_id not in index_by_id:
                continue
            # Истина — хеш текущего текста, а не только поле в JSONL.
            actual = self.text_sha256(text)
            bound = index_by_id[item_id]
            stored = stored_hash or actual
            if bound != actual or stored != bound:
                mismatched.append(item_id)
        return mismatched

    def _size_error(
        self,
        record_count: int,
        index_count: int,
        payload: bytes | None,
        dim: int,
    ) -> str | None:
        """Проверяет размер float16 и число строк индекса.

        Args:
            record_count: Число текстовых записей.
            index_count: Число строк индекса.
            payload: Байты матрицы или None.
            dim: Размерность.

        Returns:
            Текст ошибки или None.
        """
        if payload is None:
            return "нет файла векторов"
        if dim < 1:
            return "некорректная размерность"
        expected = record_count * dim * 2
        if len(payload) != expected:
            return f"размер матрицы {len(payload)} != {expected}"
        if index_count != record_count:
            return "число строк индекса не совпадает с числом записей"
        return None

    def _json_lines(self, payload: bytes) -> list[dict]:
        """Разбирает JSONL. Пустой файл даёт пустой список.

        Args:
            payload: Байты файла.

        Returns:
            Список JSON-объектов.

        Raises:
            json.JSONDecodeError: Строка не является JSON.
            ValueError: Строка не объект.
        """
        text = payload.decode("utf-8")
        rows: list[dict] = []
        for line in text.splitlines():
            if line.strip() == "":
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError("строка JSONL должна быть объектом")
            rows.append(parsed)
        return rows

    def _lines_bytes(self, lines: list[str]) -> bytes:
        """Склеивает JSONL и добавляет перевод строки в конце непустого файла.

        Args:
            lines: Уже сериализованные объекты.

        Returns:
            UTF-8 байты.
        """
        if not lines:
            return b""
        text = "\n".join(lines) + "\n"
        return text.encode("utf-8")
