"""T-CHK: нарезка очищенного документа."""

from __future__ import annotations

import pytest

from chunking.chunker import Chunker, WhitespaceTokenCounter
from models.config import ChunkingConfig
from models.documents import CleanedDocument, ExtractedBlock


class TestChunker:
    """Каталог T-CHK-01…15."""

    def test_t_chk_01_sections_do_not_mix(self) -> None:
        """Два коротких раздела не попадают в один чанк, путь — свой."""
        document = self._document(
            [
                ExtractedBlock(
                    kind="paragraph",
                    text="Текст раздела А",
                    section_path=["А"],
                ),
                ExtractedBlock(
                    kind="paragraph",
                    text="Текст раздела Б",
                    section_path=["Б"],
                ),
            ]
        )
        batch = self._chunker().chunk(document)
        assert len(batch.chunks) == 2
        assert batch.chunks[0].section_path == ["А"]
        assert batch.chunks[1].section_path == ["Б"]
        assert "Текст раздела А" in batch.chunks[0].text
        assert "Текст раздела Б" not in batch.chunks[0].text
        assert "Текст раздела Б" in batch.chunks[1].text
        assert "Текст раздела А" not in batch.chunks[1].text

    def test_t_chk_02_short_paragraph_is_one_chunk(self) -> None:
        """Короткий абзац при лимите 512 — один чанк без искусственного перекрытия."""
        document = self._document(
            [ExtractedBlock(kind="paragraph", text="Короткий абзац")]
        )
        batch = self._chunker(max_tokens=512, overlap_tokens=64).chunk(document)
        assert len(batch.chunks) == 1
        assert batch.chunks[0].text == "Короткий абзац"
        assert batch.chunking_hash == ""
        assert batch.embedding_model is None

    def test_t_chk_03_none_path_not_mixed(self) -> None:
        """Абзац до заголовка остаётся в разделе с section_path is None."""
        document = self._document(
            [
                ExtractedBlock(kind="paragraph", text="До", section_path=None),
                ExtractedBlock(
                    kind="heading",
                    text="Глава",
                    section_path=["Глава"],
                    heading_level=1,
                ),
                ExtractedBlock(
                    kind="paragraph",
                    text="После",
                    section_path=["Глава"],
                ),
            ]
        )
        batch = self._chunker().chunk(document)
        assert batch.chunks[0].section_path is None
        assert batch.chunks[0].text == "До"
        assert all(
            "До" not in chunk.text or chunk.section_path is None
            for chunk in batch.chunks
        )
        assert any(chunk.section_path == ["Глава"] for chunk in batch.chunks)
        assert all("До" not in chunk.text or "После" not in chunk.text for chunk in batch.chunks)

    def test_t_chk_04_heading_text_covered(self) -> None:
        """Текст заголовка входит в текст раздела и покрыт чанками."""
        document = self._document(
            [
                ExtractedBlock(
                    kind="heading",
                    text="3.2.1. Сроки",
                    section_path=["3.2.1. Сроки"],
                    heading_level=3,
                ),
                ExtractedBlock(
                    kind="paragraph",
                    text="Срок хранения составляет 5 лет.",
                    section_path=["3.2.1. Сроки"],
                ),
            ]
        )
        batch = self._chunker().chunk(document)
        covered = "\n".join(chunk.text for chunk in batch.chunks)
        assert "3.2.1. Сроки" in covered
        assert "Срок хранения составляет 5 лет." in covered
        assert batch.chunks[0].section_path == ["3.2.1. Сроки"]

    def test_t_chk_05_duplicate_text_keeps_first(self) -> None:
        """Из двух одинаковых текстов остаётся первое вхождение, цепочка без дыр."""
        document = self._document(
            [
                ExtractedBlock(
                    kind="paragraph",
                    text="Повтор",
                    page=1,
                    section_path=["А"],
                ),
                ExtractedBlock(
                    kind="paragraph",
                    text="Повтор",
                    page=5,
                    section_path=["Б"],
                ),
            ]
        )
        batch = self._chunker().chunk(document)
        assert len(batch.chunks) == 1
        chunk = batch.chunks[0]
        assert chunk.text == "Повтор"
        assert chunk.chunk_index == 0
        assert chunk.chunk_id == "c_0000"
        assert chunk.prev_id is None
        assert chunk.next_id is None
        assert chunk.page_from == 1
        assert chunk.page_to == 1

    def test_t_chk_06_no_main_blocks(self) -> None:
        """Только колонтитул — пустой пакет без исключения."""
        document = self._document(
            [ExtractedBlock(kind="header", text="Шапка")]
        )
        batch = self._chunker().chunk(document)
        assert batch.chunks == []

    def test_t_chk_07_repeat_is_stable(self) -> None:
        """Два вызова на одном входе дают те же id и тот же порядок."""
        document = self._document(
            [
                ExtractedBlock(kind="paragraph", text="Один", section_path=["А"]),
                ExtractedBlock(kind="paragraph", text="Два", section_path=["Б"]),
            ]
        )
        chunker = self._chunker()
        first = chunker.chunk(document)
        second = chunker.chunk(document)
        assert [chunk.chunk_id for chunk in first.chunks] == [
            chunk.chunk_id for chunk in second.chunks
        ]
        assert [chunk.text for chunk in first.chunks] == [
            chunk.text for chunk in second.chunks
        ]

    def test_t_chk_08_neighbor_chain(self) -> None:
        """Три чанка: индексы 0…2, у первого нет prev, у последнего нет next."""
        document = self._document(
            [
                ExtractedBlock(kind="paragraph", text="Альфа", section_path=["1"]),
                ExtractedBlock(kind="paragraph", text="Бета", section_path=["2"]),
                ExtractedBlock(kind="paragraph", text="Гамма", section_path=["3"]),
            ]
        )
        batch = self._chunker().chunk(document)
        assert [chunk.chunk_index for chunk in batch.chunks] == [0, 1, 2]
        assert batch.chunks[0].prev_id is None
        assert batch.chunks[0].next_id == "c_0001"
        assert batch.chunks[1].prev_id == "c_0000"
        assert batch.chunks[1].next_id == "c_0002"
        assert batch.chunks[2].prev_id == "c_0001"
        assert batch.chunks[2].next_id is None

    def test_t_chk_09_overlap_not_smaller_than_max(self) -> None:
        """overlap_tokens >= max_tokens — ValueError до построения чанков."""
        with pytest.raises(ValueError):
            Chunker(
                ChunkingConfig(
                    max_tokens=8,
                    overlap_tokens=8,
                    min_chunk_tokens=0,
                )
            )

    def test_t_chk_10_header_text_absent(self) -> None:
        """Текст header не попадает ни в один чанк."""
        document = self._document(
            [
                ExtractedBlock(kind="header", text="КОЛОНТИТУЛ"),
                ExtractedBlock(kind="paragraph", text="Основной"),
            ]
        )
        batch = self._chunker().chunk(document)
        assert batch.chunks
        assert all("КОЛОНТИТУЛ" not in chunk.text for chunk in batch.chunks)
        assert any("Основной" in chunk.text for chunk in batch.chunks)

    def test_t_chk_11_prefers_paragraph_breaks(self) -> None:
        """Длинный раздел из трёх абзацев режется по переводам строк и покрывается целиком."""
        blocks = [
            ExtractedBlock(
                kind="paragraph",
                text="a1 a2 a3",
                section_path=["Раздел"],
            ),
            ExtractedBlock(
                kind="paragraph",
                text="b1 b2 b3",
                section_path=["Раздел"],
            ),
            ExtractedBlock(
                kind="paragraph",
                text="c1 c2 c3",
                section_path=["Раздел"],
            ),
        ]
        document = self._document(blocks)
        batch = self._chunker(max_tokens=4, overlap_tokens=0, min_chunk_tokens=0).chunk(
            document
        )
        bodies = [chunk.text.strip() for chunk in batch.chunks]
        assert bodies == ["a1 a2 a3", "b1 b2 b3", "c1 c2 c3"]
        section = "a1 a2 a3\nb1 b2 b3\nc1 c2 c3"
        for token in section.split():
            assert any(token in chunk.text for chunk in batch.chunks)

    def test_t_chk_12_short_tail_merged(self) -> None:
        """Короткий хвост присоединяется к предыдущему чанку раздела."""
        document = self._document(
            [ExtractedBlock(kind="paragraph", text="w1 w2 w3 w4 w5")]
        )
        batch = self._chunker(max_tokens=3, overlap_tokens=0, min_chunk_tokens=3).chunk(
            document
        )
        assert len(batch.chunks) == 1
        assert "w4" in batch.chunks[0].text
        assert "w5" in batch.chunks[0].text
        assert "w1" in batch.chunks[0].text

    def test_t_chk_13_single_short_chunk_kept(self) -> None:
        """Единственный короткий чанк раздела не выбрасывается."""
        document = self._document(
            [ExtractedBlock(kind="paragraph", text="короткий")]
        )
        batch = self._chunker(max_tokens=10, overlap_tokens=0, min_chunk_tokens=80).chunk(
            document
        )
        assert len(batch.chunks) == 1
        assert batch.chunks[0].text == "короткий"

    def test_t_chk_14_min_zero_does_not_merge(self) -> None:
        """При min_chunk_tokens=0 короткий хвост остаётся отдельным чанком."""
        document = self._document(
            [ExtractedBlock(kind="paragraph", text="w1 w2 w3 w4 w5")]
        )
        batch = self._chunker(max_tokens=3, overlap_tokens=0, min_chunk_tokens=0).chunk(
            document
        )
        assert len(batch.chunks) == 2
        assert "w4" in batch.chunks[1].text
        assert "w1" not in batch.chunks[1].text

    def test_t_chk_15_unsplittable_word(self) -> None:
        """Одно слово длиннее окна по символам остаётся одним чанком без исключения."""
        word = "https://example.com/" + ("x" * 80)
        document = self._document([ExtractedBlock(kind="paragraph", text=word)])
        batch = self._chunker(max_tokens=1, overlap_tokens=0, min_chunk_tokens=0).chunk(
            document
        )
        assert len(batch.chunks) == 1
        assert batch.chunks[0].text == word
        assert WhitespaceTokenCounter().count(batch.chunks[0].text) == 1

    def test_preserve_section_path_false(self) -> None:
        """При preserve_section_path=false у чанков путь None."""
        document = self._document(
            [
                ExtractedBlock(
                    kind="paragraph",
                    text="Текст",
                    section_path=["Раздел"],
                )
            ]
        )
        batch = self._chunker(preserve_section_path=False).chunk(document)
        assert batch.chunks[0].section_path is None

    def _chunker(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        min_chunk_tokens: int = 0,
        preserve_section_path: bool = True,
    ) -> Chunker:
        """Собирает чанкер с явными лимитами.

        Args:
            max_tokens: Лимит окна.
            overlap_tokens: Перекрытие.
            min_chunk_tokens: Порог слияния хвоста.
            preserve_section_path: Копировать ли путь.

        Returns:
            Chunker.
        """
        return Chunker(
            ChunkingConfig(
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                min_chunk_tokens=min_chunk_tokens,
                preserve_section_path=preserve_section_path,
            )
        )

    def _document(self, blocks: list[ExtractedBlock]) -> CleanedDocument:
        """Собирает очищенный документ.

        Args:
            blocks: Блоки.

        Returns:
            CleanedDocument.
        """
        return CleanedDocument(
            doc_id="reglament.docx",
            source_sha256="ab" * 32,
            blocks=blocks,
        )
