"""Сборка ContextBundle по режиму section."""

from __future__ import annotations

from chunking.chunker import WhitespaceTokenCounter
from models.artifacts import ContextBundle, ContextCandidate
from models.config import ContextConfig
from models.documents import CandidateBatch, CandidateRecord, ChunkBatch, ChunkRecord
from models.questions import Question
from ports.pipeline import ContextAssembler as ContextAssemblerPort


class ContextFragmentDraft:
    """Фрагмент бандла до присвоения ранга.

    Attributes:
        chunks: Чанки, которые сейчас входят в текст, по порядку документа.
        hit_ids: Идентификаторы хитов реранка этого фрагмента.
        score: Лучший rerank_score хитов фрагмента.
        section_path: Путь секции или None у окна без пути.
        truncated: Из фрагмента уже убран хотя бы один чанк.
        hit_anchor: Минимальный chunk_index хитов, для стабильного порядка.
    """

    def __init__(
        self,
        chunks: list[ChunkRecord],
        hit_ids: set[str],
        score: float,
        section_path: list[str] | None,
        truncated: bool,
    ) -> None:
        """Сохраняет состав фрагмента.

        Args:
            chunks: Чанки фрагмента по возрастанию chunk_index.
            hit_ids: Хиты, которые нельзя снимать бюджетом.
            score: Лучший скор хитов.
            section_path: Путь или None.
            truncated: Уже сокращён ли фрагмент.
        """
        self.chunks = chunks
        self.hit_ids = hit_ids
        self.score = score
        self.section_path = section_path
        self.truncated = truncated
        hit_indexes = [
            chunk.chunk_index
            for chunk in chunks
            if chunk.chunk_id in hit_ids
        ]
        self.hit_anchor = min(hit_indexes)

    def text(self) -> str:
        """Склеивает тексты чанков одним переводом строки.

        Returns:
            Текст фрагмента без префикса нумерации.
        """
        return "\n".join(chunk.text for chunk in self.chunks)

    def farthest_neighbor(self) -> ChunkRecord | None:
        """Выбирает соседа, которого бюджет снимает первым.

        Returns:
            Чанк с наибольшим расстоянием до хита или None, если соседей нет.
        """
        neighbors = [
            chunk for chunk in self.chunks if chunk.chunk_id not in self.hit_ids
        ]
        if not neighbors:
            return None
        hit_indexes = [
            chunk.chunk_index
            for chunk in self.chunks
            if chunk.chunk_id in self.hit_ids
        ]

        def distance_key(chunk: ChunkRecord) -> tuple[int, int]:
            distance = min(
                abs(chunk.chunk_index - hit_index) for hit_index in hit_indexes
            )
            return (distance, chunk.chunk_index)

        return max(neighbors, key=distance_key)

    def drop(self, chunk: ChunkRecord) -> None:
        """Убирает соседний чанк и помечает фрагмент сокращённым.

        Args:
            chunk: Сосед, выбранный farthest_neighbor.
        """
        self.chunks = [
            item for item in self.chunks if item.chunk_id != chunk.chunk_id
        ]
        self.truncated = True


class ContextAssembler(ContextAssemblerPort):
    """Собирает один ContextBundle на пару документ × вопрос.

    Attributes:
        config: Бюджеты и режим расширения.
        embedding_model: Имя модели для retrieval_meta.
        reranker_model: Имя реранкера для retrieval_meta.
        top_k: retrieval.top_k для retrieval_meta.
    """

    def __init__(
        self,
        config: ContextConfig,
        embedding_model: str,
        reranker_model: str,
        top_k: int,
    ) -> None:
        """Сохраняет срез конфигурации, нужный полям бандла.

        Args:
            config: ContextConfig.
            embedding_model: Имя эмбеддера.
            reranker_model: Имя реранкера.
            top_k: Число кандидатов retrieval до реранка.
        """
        self.config = config
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.top_k = top_k
        self._counter = WhitespaceTokenCounter()

    def assemble(
        self,
        doc_id: str,
        question: Question,
        candidates: CandidateBatch,
        empty_search: bool,
        chunks: ChunkBatch,
    ) -> ContextBundle:
        """Собирает бандл в памяти и не пишет файл.

        Args:
            doc_id: Идентификатор документа.
            question: Вопрос с синонимами.
            candidates: Хиты после реранка.
            empty_search: Флаг пустого поиска от фазы.
            chunks: Все чанки документа. При пустом поиске не читаются.

        Returns:
            ContextBundle без хешей: их проставляет фаза.

        Raises:
            ValueError: Чужой режим expansion, пустой пакет при непустом
                поиске, скор None или хит без чанка в пакете.
        """
        if self.config.expansion != "section":
            raise ValueError(
                "expansion кроме section в этой итерации не собирается"
            )
        if empty_search:
            if candidates.candidates:
                raise ValueError(
                    "empty_search=true требует пустой список кандидатов"
                )
            return self._empty_bundle(doc_id, question)
        if not candidates.candidates:
            raise ValueError(
                "empty_search=false требует хотя бы один хит"
            )
        self._require_scores(candidates)
        # Пакет чанков нужен только непустому поиску.
        by_id = {chunk.chunk_id: chunk for chunk in chunks.chunks}
        self._require_known_hits(candidates, by_id)
        fragments = self._build_fragments(candidates, chunks, by_id)
        ordered = self._fit_total_budget(fragments)
        return self._bundle(doc_id, question, candidates, ordered)

    def _empty_bundle(self, doc_id: str, question: Question) -> ContextBundle:
        """Бандл пустого поиска с нулевым бюджетом.

        Args:
            doc_id: Идентификатор документа.
            question: Вопрос.

        Returns:
            ContextBundle с empty_search и пустыми кандидатами.
        """
        return ContextBundle(
            doc_id=doc_id,
            question_id=question.id,
            category=question.category,
            used_synonyms=list(question.synonyms),
            empty_search=True,
            reason="no_relevant_chunks",
            budget_used_tokens=0,
            candidates=[],
            retrieval_meta=self._meta(after_rerank=0, sections_used=[]),
        )

    def _require_scores(self, candidates: CandidateBatch) -> None:
        """Проверяет, что у каждого хита есть rerank_score.

        Args:
            candidates: Пакет после реранка.

        Raises:
            ValueError: Скор не заполнен.
        """
        for candidate in candidates.candidates:
            if candidate.rerank_score is None:
                raise ValueError("rerank_score хита не заполнен")

    def _require_known_hits(
        self,
        candidates: CandidateBatch,
        by_id: dict[str, ChunkRecord],
    ) -> None:
        """Проверяет, что каждый хит есть в ChunkBatch.

        Args:
            candidates: Хиты.
            by_id: Чанки документа по chunk_id.

        Raises:
            ValueError: Идентификатора нет в пакете чанков.
        """
        for candidate in candidates.candidates:
            if candidate.chunk_id not in by_id:
                raise ValueError(
                    f"чанк хита отсутствует в пакете: {candidate.chunk_id}"
                )

    def _build_fragments(
        self,
        candidates: CandidateBatch,
        chunks: ChunkBatch,
        by_id: dict[str, ChunkRecord],
    ) -> list[ContextFragmentDraft]:
        """Группирует хиты с путём и строит окно на каждый хит без пути.

        Args:
            candidates: Хиты.
            chunks: Все чанки документа.
            by_id: Индекс чанков.

        Returns:
            Фрагменты до общего бюджета.
        """
        grouped: dict[tuple[str, ...], list[CandidateRecord]] = {}
        pathless: list[CandidateRecord] = []
        for candidate in candidates.candidates:
            # Путь берётся у чанка документа: копия в кандидате может устареть.
            path = self._path_of_chunk(by_id[candidate.chunk_id])
            if path is None:
                pathless.append(candidate)
                continue
            grouped.setdefault(tuple(path), []).append(candidate)
        fragments: list[ContextFragmentDraft] = []
        for path_key, hits in grouped.items():
            fragments.append(self._section_fragment(path_key, hits, chunks))
        for hit in pathless:
            fragments.append(self._window_fragment(hit, by_id))
        return fragments

    def _section_fragment(
        self,
        path_key: tuple[str, ...],
        hits: list[CandidateRecord],
        chunks: ChunkBatch,
    ) -> ContextFragmentDraft:
        """Берёт секцию целиком или окно хитов, если она длиннее лимита.

        Args:
            path_key: Полный путь секции.
            hits: Хиты этой секции.
            chunks: Все чанки документа.

        Returns:
            Черновик одного фрагмента.
        """
        section_chunks = [
            chunk
            for chunk in chunks.chunks
            if self._path_of_chunk(chunk) == list(path_key)
        ]
        section_chunks.sort(key=lambda chunk: chunk.chunk_index)
        full_text = "\n".join(chunk.text for chunk in section_chunks)
        hit_ids = {hit.chunk_id for hit in hits}
        score = max(float(hit.rerank_score or 0.0) for hit in hits)
        if self._counter.count(full_text) > self.config.max_section_tokens:
            chosen = self._hits_with_section_neighbors(section_chunks, hit_ids)
            truncated = True
        else:
            chosen = section_chunks
            truncated = False
        return ContextFragmentDraft(
            chunks=chosen,
            hit_ids=hit_ids,
            score=score,
            section_path=list(path_key),
            truncated=truncated,
        )

    def _hits_with_section_neighbors(
        self,
        section_chunks: list[ChunkRecord],
        hit_ids: set[str],
    ) -> list[ChunkRecord]:
        """Окно neighbors_fallback вокруг хитов внутри одной секции.

        Args:
            section_chunks: Чанки секции по chunk_index.
            hit_ids: Хиты этой секции.

        Returns:
            Чанки окна без повторов, по порядку секции.
        """
        index_by_id = {
            chunk.chunk_id: index for index, chunk in enumerate(section_chunks)
        }
        chosen_indexes: set[int] = set()
        width = self.config.neighbors_fallback
        for hit_id in hit_ids:
            center = index_by_id[hit_id]
            start = max(0, center - width)
            end = min(len(section_chunks) - 1, center + width)
            for index in range(start, end + 1):
                chosen_indexes.add(index)
        return [section_chunks[index] for index in sorted(chosen_indexes)]

    def _window_fragment(
        self,
        hit: CandidateRecord,
        by_id: dict[str, ChunkRecord],
    ) -> ContextFragmentDraft:
        """Окно по prev_id/next_id, граница раздела его не останавливает.

        Args:
            hit: Хит без section_path.
            by_id: Индекс чанков.

        Returns:
            Черновик окна. Путь фрагмента остаётся None.
        """
        center = by_id[hit.chunk_id]
        collected = {center.chunk_id: center}
        self._walk(center, by_id, collected, backward=True)
        self._walk(center, by_id, collected, backward=False)
        ordered = sorted(collected.values(), key=lambda chunk: chunk.chunk_index)
        return ContextFragmentDraft(
            chunks=ordered,
            hit_ids={hit.chunk_id},
            score=float(hit.rerank_score or 0.0),
            section_path=None,
            truncated=False,
        )

    def _walk(
        self,
        start: ChunkRecord,
        by_id: dict[str, ChunkRecord],
        collected: dict[str, ChunkRecord],
        backward: bool,
    ) -> None:
        """Добавляет до neighbors_fallback шагов по цепочке.

        Args:
            start: Хит, от которого идём.
            by_id: Индекс чанков.
            collected: Уже набранные чанки окна.
            backward: True — по prev_id, иначе по next_id.
        """
        current = start
        steps_left = self.config.neighbors_fallback
        while steps_left > 0:
            neighbor_id = current.prev_id if backward else current.next_id
            if neighbor_id is None or neighbor_id not in by_id:
                return
            current = by_id[neighbor_id]
            collected[current.chunk_id] = current
            steps_left -= 1

    def _fit_total_budget(
        self,
        fragments: list[ContextFragmentDraft],
    ) -> list[ContextFragmentDraft]:
        """Снимает соседей, пока сумма не влезет в max_total_tokens.

        Args:
            fragments: Фрагменты до бюджета.

        Returns:
            Те же фрагменты в порядке убывания скора. Хиты остаются.
        """
        ordered = sorted(
            fragments,
            key=lambda fragment: (-fragment.score, fragment.hit_anchor),
        )
        limit = self.config.max_total_tokens
        while self._total_tokens(ordered) > limit:
            removed = False
            # С худшего фрагмента, с его дальнего края.
            for fragment in reversed(ordered):
                neighbor = fragment.farthest_neighbor()
                if neighbor is None:
                    continue
                fragment.drop(neighbor)
                removed = True
                break
            if not removed:
                break
        return ordered

    def _total_tokens(self, fragments: list[ContextFragmentDraft]) -> int:
        """Суммирует токены текстов фрагментов.

        Args:
            fragments: Текущий состав.

        Returns:
            Сумма WhitespaceTokenCounter по text.
        """
        total = 0
        for fragment in fragments:
            total += self._counter.count(fragment.text())
        return total

    def _bundle(
        self,
        doc_id: str,
        question: Question,
        candidates: CandidateBatch,
        fragments: list[ContextFragmentDraft],
    ) -> ContextBundle:
        """Переносит черновики в ContextCandidate и считает бюджет.

        Args:
            doc_id: Идентификатор документа.
            question: Вопрос.
            candidates: Входные хиты, их число идёт в after_rerank.
            fragments: Фрагменты после бюджета, лучший первый.

        Returns:
            Непустой ContextBundle.
        """
        context_candidates: list[ContextCandidate] = []
        sections_used: list[str] = []
        for index, fragment in enumerate(fragments):
            if fragment.section_path is not None:
                sections_used.append(fragment.section_path[-1])
            context_candidates.append(
                ContextCandidate(
                    rank_after_rerank=index + 1,
                    rerank_score=fragment.score,
                    section_path=fragment.section_path,
                    pages=self._pages(fragment.chunks),
                    text=fragment.text(),
                    chunks_included=[
                        chunk.chunk_id for chunk in fragment.chunks
                    ],
                    truncated=fragment.truncated,
                )
            )
        return ContextBundle(
            doc_id=doc_id,
            question_id=question.id,
            category=question.category,
            used_synonyms=list(question.synonyms),
            empty_search=False,
            reason=None,
            budget_used_tokens=self._total_tokens(fragments),
            candidates=context_candidates,
            retrieval_meta=self._meta(
                after_rerank=len(candidates.candidates),
                sections_used=sections_used,
            ),
        )

    def _pages(self, chunks: list[ChunkRecord]) -> list[int]:
        """Собирает номера страниц чанков без нулей.

        Args:
            chunks: Чанки фрагмента.

        Returns:
            Уникальные номера по возрастанию.
        """
        numbers: set[int] = set()
        for chunk in chunks:
            if chunk.page_from == 0 and chunk.page_to == 0:
                continue
            if chunk.page_from == 0 or chunk.page_to == 0:
                nonzero = chunk.page_from or chunk.page_to
                numbers.add(nonzero)
                continue
            low = min(chunk.page_from, chunk.page_to)
            high = max(chunk.page_from, chunk.page_to)
            for number in range(low, high + 1):
                if number != 0:
                    numbers.add(number)
        return sorted(numbers)

    def _meta(self, after_rerank: int, sections_used: list[str]) -> dict[str, object]:
        """Поля retrieval_meta, которые известны сборщику.

        Args:
            after_rerank: Число входных хитов.
            sections_used: Последние заголовки вошедших секций.

        Returns:
            Словарь метаданных без хешей.
        """
        return {
            "top_k": self.top_k,
            "after_rerank": after_rerank,
            "sections_used": sections_used,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
        }

    def _path_of_chunk(self, chunk: ChunkRecord) -> list[str] | None:
        """Возвращает непустой путь чанка или None.

        Args:
            chunk: Чанк документа.

        Returns:
            Копия пути или None.
        """
        if not chunk.section_path:
            return None
        return list(chunk.section_path)
