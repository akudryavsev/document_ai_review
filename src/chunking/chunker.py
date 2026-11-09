"""Нарезка очищенного документа на чанки."""

from __future__ import annotations

from models.config import ChunkingConfig
from models.documents import (
    ChunkBatch,
    ChunkRecord,
    CleanedDocument,
    ExtractedBlock,
)
from ports.pipeline import Chunker as ChunkerPort


class WhitespaceTokenCounter:
    """Считает токены как фрагменты str.split() без модели эмбеддинга."""

    def count(self, text: str) -> int:
        """Возвращает число whitespace-токенов.

        Args:
            text: Фрагмент текста.

        Returns:
            0 для пустой строки и строки из одних пробелов.
        """
        return len(self.spans(text))

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Находит границы токенов в исходной строке.

        Args:
            text: Фрагмент текста.

        Returns:
            Пары (начало, конец) не включая пробелы между словами.
        """
        found: list[tuple[int, int]] = []
        index = 0
        length = len(text)
        while index < length:
            while index < length and text[index].isspace():
                index += 1
            if index >= length:
                break
            start = index
            while index < length and not text[index].isspace():
                index += 1
            found.append((start, index))
        return found


class ChunkDraft:
    """Черновик чанка до перенумерации.

    Attributes:
        text: Подстрока раздела.
        section_path: Путь раздела или None.
        page_from: Минимальная страница или 0.
        page_to: Максимальная страница или 0.
    """

    def __init__(
        self,
        text: str,
        section_path: list[str] | None,
        page_from: int,
        page_to: int,
    ) -> None:
        """Сохраняет поля черновика.

        Args:
            text: Текст чанка.
            section_path: Путь раздела.
            page_from: Начальная страница.
            page_to: Конечная страница.
        """
        self.text = text
        self.section_path = section_path
        self.page_from = page_from
        self.page_to = page_to


class Chunker(ChunkerPort):
    """Режет документ по разделам, затем по токенам с перекрытием.

    Attributes:
        config: Параметры нарезки.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        """Проверяет конфигурацию и сохраняет её.

        Args:
            config: ChunkingConfig.

        Raises:
            ValueError: Лимиты не позволяют сдвигать окно.
        """
        self._validate(config)
        self.config = config
        self._counter = WhitespaceTokenCounter()

    def chunk(self, document: CleanedDocument) -> ChunkBatch:
        """Строит ChunkBatch в памяти, без записи на диск.

        Args:
            document: Очищенный документ.

        Returns:
            Пакет с chunking_hash="" и embedding_model=None.
        """
        drafts = self._drafts(document.blocks)
        records = self._records(drafts)
        return ChunkBatch(
            doc_id=document.doc_id,
            chunks=records,
            chunking_hash="",
            embedding_model=None,
        )

    def _validate(self, config: ChunkingConfig) -> None:
        """Отклоняет конфигурацию, на которой окно не двигается вперёд.

        Args:
            config: Параметры нарезки.

        Raises:
            ValueError: Нарушено правило раздела 4.1 спецификации чанкинга.
        """
        if config.max_tokens < 1:
            raise ValueError("max_tokens должен быть не меньше 1")
        if config.overlap_tokens < 0:
            raise ValueError("overlap_tokens не может быть отрицательным")
        if config.min_chunk_tokens < 0:
            raise ValueError("min_chunk_tokens не может быть отрицательным")
        if config.overlap_tokens >= config.max_tokens:
            raise ValueError("overlap_tokens должен быть строго меньше max_tokens")

    def _drafts(self, blocks: list[ExtractedBlock]) -> list[ChunkDraft]:
        """Нарезает основной текст и дедуплицирует точные копии.

        Args:
            blocks: Блоки очищенного документа.

        Returns:
            Черновики в порядке документа, без повторов текста.
        """
        main_blocks = [
            block for block in blocks if block.kind not in {"header", "footer"}
        ]
        drafts: list[ChunkDraft] = []
        for path, group in self._sections(main_blocks):
            drafts.extend(self._drafts_for_section(path, group))
        unique: list[ChunkDraft] = []
        seen: set[str] = set()
        for draft in drafts:
            if draft.text in seen:
                continue
            seen.add(draft.text)
            unique.append(draft)
        return unique

    def _sections(
        self,
        blocks: list[ExtractedBlock],
    ) -> list[tuple[list[str] | None, list[ExtractedBlock]]]:
        """Группирует подряд идущие блоки с одним section_path.

        Args:
            blocks: Основной текст без колонтитулов.

        Returns:
            Пары (путь, блоки). None и список — разные разделы.
        """
        if not blocks:
            return []
        groups: list[tuple[list[str] | None, list[ExtractedBlock]]] = []
        current_path = blocks[0].section_path
        current: list[ExtractedBlock] = [blocks[0]]
        for block in blocks[1:]:
            if block.section_path == current_path:
                current.append(block)
                continue
            groups.append((current_path, current))
            current_path = block.section_path
            current = [block]
        groups.append((current_path, current))
        return groups

    def _drafts_for_section(
        self,
        path: list[str] | None,
        blocks: list[ExtractedBlock],
    ) -> list[ChunkDraft]:
        """Режет один раздел.

        Args:
            path: Путь раздела.
            blocks: Блоки раздела по порядку.

        Returns:
            Черновики этого раздела.
        """
        text, ranges = self._join_blocks(blocks)
        if text == "":
            return []
        windows = self._windows(text, ranges)
        stored_path = None if path is None else list(path)
        drafts: list[ChunkDraft] = []
        for start, end in windows:
            piece = text[start:end]
            if piece == "":
                continue
            page_from, page_to = self._pages_for_span(ranges, start, end)
            drafts.append(
                ChunkDraft(
                    text=piece,
                    section_path=stored_path,
                    page_from=page_from,
                    page_to=page_to,
                )
            )
        return drafts

    def _join_blocks(
        self,
        blocks: list[ExtractedBlock],
    ) -> tuple[str, list[tuple[int, int, ExtractedBlock]]]:
        """Склеивает блоки раздела через один перевод строки.

        Args:
            blocks: Блоки раздела.

        Returns:
            Текст и диапазоны символов каждого блока внутри него.
        """
        parts: list[str] = []
        ranges: list[tuple[int, int, ExtractedBlock]] = []
        offset = 0
        for index, block in enumerate(blocks):
            if index > 0:
                parts.append("\n")
                offset += 1
            start = offset
            parts.append(block.text)
            offset += len(block.text)
            ranges.append((start, offset, block))
        return "".join(parts), ranges

    def _windows(
        self,
        text: str,
        ranges: list[tuple[int, int, ExtractedBlock]],
    ) -> list[tuple[int, int]]:
        """Возвращает окна символов, покрывающие текст раздела.

        Args:
            text: Текст раздела.
            ranges: Диапазоны блоков.

        Returns:
            Пары (char_start, char_end).
        """
        spans = self._counter.spans(text)
        if len(spans) <= self.config.max_tokens:
            return [(0, len(text))]
        edges = self._block_edges(spans, ranges)
        sentence_starts = self._sentence_token_starts(text, spans)
        windows: list[tuple[int, int]] = []
        start_token = 0
        total = len(spans)
        while start_token < total:
            end_token = self._window_end(start_token, total, edges, sentence_starts)
            if end_token <= start_token:
                end_token = start_token + 1
            if start_token == 0:
                char_start = 0
            else:
                char_start = spans[start_token][0]
            if end_token >= total:
                char_end = len(text)
            else:
                char_end = spans[end_token][0]
            windows.append((char_start, char_end))
            if end_token >= total:
                break
            next_start = end_token - self.config.overlap_tokens
            # Чанк короче перекрытия нельзя сдвинуть на overlap: начало осталось бы на месте.
            if next_start <= start_token:
                next_start = end_token
            if next_start <= start_token:
                next_start = start_token + 1
            start_token = next_start
        return self._merge_short_tail(text, windows)

    def _window_end(
        self,
        start_token: int,
        total: int,
        edges: list[int],
        sentence_starts: set[int],
    ) -> int:
        """Набирает конец окна: целые куски, пока они влезают, длинный блок — изнутри.

        Args:
            start_token: Индекс первого токена окна.
            total: Число токенов раздела.
            edges: Границы блоков, включая 0 и total.
            sentence_starts: Индексы токенов, с которых начинается предложение.

        Returns:
            Индекс конца окна, не включая этот токен.
        """
        if total - start_token <= self.config.max_tokens:
            return total
        limit = start_token + self.config.max_tokens
        cursor = start_token
        edge_index = self._edge_index(edges, start_token)
        while cursor < limit and edge_index + 1 < len(edges):
            piece_end = edges[edge_index + 1]
            if piece_end <= limit:
                cursor = piece_end
                edge_index += 1
                continue
            remainder = piece_end - cursor
            if remainder > self.config.max_tokens:
                return self._pack_sentences(
                    cursor,
                    limit,
                    sentence_starts,
                    piece_end,
                )
            return cursor
        return cursor

    def _edge_index(self, edges: list[int], start_token: int) -> int:
        """Находит блок, внутри которого лежит начало окна.

        Args:
            edges: Границы блоков.
            start_token: Индекс первого токена окна.

        Returns:
            Индекс левой границы этого блока в edges.
        """
        found = 0
        for index, edge in enumerate(edges):
            if edge <= start_token:
                found = index
                continue
            break
        return found

    def _pack_sentences(
        self,
        cursor: int,
        limit: int,
        sentence_starts: set[int],
        piece_end: int,
    ) -> int:
        """Добирает предложения длинного блока до лимита окна.

        Args:
            cursor: Первый токен ещё не взятого хвоста блока.
            limit: Правая граница окна по max_tokens, не включая этот токен.
            sentence_starts: Индексы начал предложений.
            piece_end: Конец блока, не включая этот токен.

        Returns:
            Индекс конца окна. Предложение длиннее лимита режется по словам.
        """
        boundaries: list[int] = []
        for token_index in sorted(sentence_starts):
            if token_index <= cursor:
                continue
            if token_index > piece_end:
                break
            boundaries.append(token_index)
        if not boundaries or boundaries[-1] != piece_end:
            boundaries.append(piece_end)
        position = cursor
        for boundary in boundaries:
            if boundary <= position:
                continue
            if boundary <= limit:
                position = boundary
                continue
            sentence_length = boundary - position
            if sentence_length > self.config.max_tokens:
                return limit
            return position
        return position

    def _block_edges(
        self,
        spans: list[tuple[int, int]],
        ranges: list[tuple[int, int, ExtractedBlock]],
    ) -> list[int]:
        """Собирает границы блоков в индексах токенов.

        Args:
            spans: Границы токенов.
            ranges: Диапазоны блоков.

        Returns:
            Индексы начал блоков, плюс 0 и число токенов в конце.
        """
        edges = [0]
        for token_index in sorted(self._block_token_starts(spans, ranges)):
            if token_index <= 0 or token_index >= len(spans):
                continue
            if edges[-1] == token_index:
                continue
            edges.append(token_index)
        edges.append(len(spans))
        return edges

    def _block_token_starts(
        self,
        spans: list[tuple[int, int]],
        ranges: list[tuple[int, int, ExtractedBlock]],
    ) -> set[int]:
        """Находит токены, с которых начинается каждый блок кроме первого.

        Args:
            spans: Границы токенов.
            ranges: Диапазоны блоков.

        Returns:
            Индексы токенов-границ.
        """
        boundaries: set[int] = set()
        for block_start, _block_end, _block in ranges[1:]:
            for token_index, (char_start, _char_end) in enumerate(spans):
                if char_start >= block_start:
                    boundaries.add(token_index)
                    break
        return boundaries

    def _sentence_token_starts(
        self,
        text: str,
        spans: list[tuple[int, int]],
    ) -> set[int]:
        """Находит токены после `.` `!` `?` `…` и пробела.

        Args:
            text: Текст раздела.
            spans: Границы токенов.

        Returns:
            Индексы токенов, перед которыми можно резать предложение.
        """
        endings = {".", "!", "?", "…"}
        found: set[int] = set()
        for token_index, (char_start, _char_end) in enumerate(spans):
            if token_index == 0:
                continue
            cursor = char_start - 1
            saw_space = False
            while cursor >= 0 and text[cursor].isspace():
                saw_space = True
                cursor -= 1
            if not saw_space or cursor < 0:
                continue
            if text[cursor] in endings:
                found.add(token_index)
        return found

    def _merge_short_tail(
        self,
        text: str,
        windows: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Присоединяет короткий хвост к предыдущему чанку того же раздела.

        Args:
            text: Текст раздела.
            windows: Окна до слияния.

        Returns:
            Окна после правила min_chunk_tokens.
        """
        if self.config.min_chunk_tokens <= 0 or len(windows) < 2:
            return windows
        last_start, last_end = windows[-1]
        last_text = text[last_start:last_end]
        if self._counter.count(last_text) >= self.config.min_chunk_tokens:
            return windows
        prev_start, _prev_end = windows[-2]
        merged = (prev_start, last_end)
        return windows[:-2] + [merged]

    def _pages_for_span(
        self,
        ranges: list[tuple[int, int, ExtractedBlock]],
        start: int,
        end: int,
    ) -> tuple[int, int]:
        """Считает min/max page у блоков, пересекающих окно.

        Args:
            ranges: Диапазоны блоков.
            start: Начало окна.
            end: Конец окна.

        Returns:
            (page_from, page_to). Оба 0, если номеров нет.
        """
        pages: list[int] = []
        for block_start, block_end, block in ranges:
            if block_end <= start or block_start >= end:
                continue
            if block.page is not None:
                pages.append(block.page)
        if not pages:
            return 0, 0
        return min(pages), max(pages)

    def _records(self, drafts: list[ChunkDraft]) -> list[ChunkRecord]:
        """Нумерует чанки сплошняком и связывает соседей.

        Args:
            drafts: Уникальные черновики.

        Returns:
            Список ChunkRecord.
        """
        total = len(drafts)
        records: list[ChunkRecord] = []
        for index, draft in enumerate(drafts):
            if self.config.preserve_section_path:
                path = draft.section_path
            else:
                path = None
            prev_id = None if index == 0 else f"c_{index - 1:04d}"
            next_id = None if index == total - 1 else f"c_{index + 1:04d}"
            records.append(
                ChunkRecord(
                    chunk_id=f"c_{index:04d}",
                    text=draft.text,
                    section_path=path,
                    page_from=draft.page_from,
                    page_to=draft.page_to,
                    chunk_index=index,
                    prev_id=prev_id,
                    next_id=next_id,
                )
            )
        return records
