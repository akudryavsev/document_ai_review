"""Модели документов на границах фаз."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DocumentRef(BaseModel):
    """Ссылка на входной документ.

    Attributes:
        doc_id: Идентификатор документа (обычно имя файла).
        path: Путь к файлу на диске.
        suffix: Расширение (.docx / .pdf).
    """

    doc_id: str
    path: str
    suffix: str


class ExtractedBlock(BaseModel):
    """Один блок извлечённого или очищенного документа.

    Attributes:
        kind: Роль блока.
        text: Текст этого блока.
        page: Номер страницы с 1 или None.
        section_path: Путь заголовков; None — структура не извлечена.
        heading_level: Уровень заголовка с 1, только у heading.
        paragraph_index: Номер абзаца с 1 среди paragraph и table.
    """

    kind: Literal["heading", "paragraph", "table", "header", "footer"]
    text: str
    page: int | None = None
    section_path: list[str] | None = None
    heading_level: int | None = None
    paragraph_index: int | None = None

    @field_validator("section_path")
    @classmethod
    def reject_empty_section_path(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        """Запрещает пустой путь: это не то же самое, что отсутствие структуры.

        Args:
            value: Путь заголовков или None.

        Returns:
            Исходный путь, если он None или непустой.

        Raises:
            ValueError: Передан пустой список.
        """
        if value is not None and len(value) == 0:
            raise ValueError("Пустой section_path недопустим")
        return value


class BlockDocument(BaseModel):
    """Документ, у которого текст живёт только в списке блоков.

    Attributes:
        doc_id: Идентификатор документа.
        source_sha256: Отпечаток сырых байтов исходного файла.
        blocks: Блоки в порядке чтения.
    """

    doc_id: str
    source_sha256: str = ""
    blocks: list[ExtractedBlock] = Field(default_factory=list)

    def joined_text(self) -> str:
        """Склеивает тексты всех блоков через перевод строки.

        Returns:
            Строка из blocks[].text. Пустой список даёт пустую строку.
        """
        parts = [block.text for block in self.blocks]
        return "\n".join(parts)

    def page_texts(self) -> list[str]:
        """Собирает тексты по страницам, не сохраняя копию в модели.

        Returns:
            Список строк: индекс 0 — страница 1. Блок без page не входит.
            Если у основного текста нет номеров — список из одного joined_text().
        """
        main_kinds = {"heading", "paragraph", "table"}
        main_blocks = [
            block for block in self.blocks if block.kind in main_kinds
        ]
        numbered = [block for block in main_blocks if block.page is not None]
        if not numbered:
            return [self.joined_text()]
        last_page = max(block.page for block in numbered if block.page is not None)
        pages = [""] * last_page
        grouped: dict[int, list[str]] = {}
        for block in self.blocks:
            if block.page is None:
                continue
            grouped.setdefault(block.page, []).append(block.text)
        for page_number, texts in grouped.items():
            if page_number < 1 or page_number > last_page:
                continue
            pages[page_number - 1] = "\n".join(texts)
        return pages


class ExtractedDocument(BlockDocument):
    """Документ после извлечения текста.

    Attributes:
        doc_id: Идентификатор документа.
        source_sha256: SHA-256 сырых байтов файла.
        blocks: Единственное хранилище текста.
        has_text_layer: Есть ли текстовый слой (для PDF).
        failure_reason: Причина отказа извлечения, если есть.
    """

    has_text_layer: bool = True
    failure_reason: str | None = None


class CleanedDocument(BlockDocument):
    """Документ после очистки.

    Attributes:
        doc_id: Идентификатор документа.
        source_sha256: Копия отпечатка исходного файла.
        blocks: Оставшиеся блоки без перенумерации.
    """


class ChunkRecord(BaseModel):
    """Один чанк документа.

    Attributes:
        chunk_id: Идентификатор чанка.
        text: Текст чанка.
        text_sha256: SHA-256 текста; пустая строка, пока store не посчитал.
        page_from: Начальная страница; 0, если номера нет.
        page_to: Конечная страница; 0, если номера нет.
        section_path: Иерархия заголовков или None.
        chunk_index: Порядковый номер с 0.
        prev_id: Идентификатор предыдущего чанка.
        next_id: Идентификатор следующего чанка.
    """

    chunk_id: str
    text: str
    text_sha256: str = ""
    page_from: int = 0
    page_to: int = 0
    section_path: list[str] | None = None
    chunk_index: int = 0
    prev_id: str | None = None
    next_id: str | None = None

    @field_validator("section_path")
    @classmethod
    def reject_empty_section_path(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        """Запрещает пустой путь чанка.

        Args:
            value: Путь раздела или None.

        Returns:
            Исходный путь.

        Raises:
            ValueError: Передан пустой список.
        """
        if value is not None and len(value) == 0:
            raise ValueError("Пустой section_path недопустим")
        return value


class ChunkBatch(BaseModel):
    """Пакет чанков одного документа.

    Attributes:
        doc_id: Идентификатор документа.
        chunks: Список чанков.
        chunking_hash: Хеш параметров чанкинга.
        embedding_model: Модель эмбеддера, если уже известна.
    """

    doc_id: str
    chunks: list[ChunkRecord] = Field(default_factory=list)
    chunking_hash: str = ""
    embedding_model: str | None = None


class CandidateRecord(BaseModel):
    """Кандидат retrieval до или после реранка.

    Attributes:
        question_id: Идентификатор вопроса.
        chunk_id: Идентификатор чанка.
        vector_score: Скор векторного поиска.
        bm25_score: Скор BM25.
        fused_score: Слитый скор.
        rank: Ранг до реранка.
        rerank_score: Скор реранкера.
        text: Текст пассажа.
        section_path: Иерархия заголовков; None, если путь не извлечён.
        pages: Страницы пассажа.
    """

    question_id: str
    chunk_id: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rank: int = 0
    rerank_score: float | None = None
    text: str = ""
    section_path: list[str] | None = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)


class CandidateBatch(BaseModel):
    """Пакет кандидатов по документу или паре.

    Attributes:
        doc_id: Идентификатор документа.
        candidates: Список кандидатов.
        retrieval_config_hash: Хеш retrieval-конфига.
        reranker_config_hash: Хеш реранкера.
        chunking_hash: Хеш чанкинга upstream-артефакта.
    """

    doc_id: str
    candidates: list[CandidateRecord] = Field(default_factory=list)
    retrieval_config_hash: str = ""
    reranker_config_hash: str = ""
    chunking_hash: str = ""
