"""Токенизация с леммами и скоринг BM25 Okapi."""

from __future__ import annotations

import math
import re
from collections import Counter

import pymorphy3


class Bm25Tokenizer:
    """Режет текст на леммы pymorphy3.

    Attributes:
        morph: Анализатор морфологии.
        _token_pattern: Буквы или цифры Unicode.
    """

    def __init__(self) -> None:
        """Создаёт анализатор один раз на экземпляр."""
        self.morph = pymorphy3.MorphAnalyzer()
        self._token_pattern = re.compile(r"[^\W_]+", re.UNICODE)

    def tokenize(self, text: str) -> list[str]:
        """Возвращает леммы буквенных токенов; цифры без леммы.

        Args:
            text: Исходная строка.

        Returns:
            Список термов в порядке появления.
        """
        folded = text.casefold()
        terms: list[str] = []
        for match in self._token_pattern.finditer(folded):
            token = match.group(0)
            if token.isdigit():
                terms.append(token)
                continue
            # Первый разбор — норма для спеки.
            lemma = self.morph.parse(token)[0].normal_form
            terms.append(lemma)
        return terms


class Bm25Index:
    """Индекс BM25 по чанкам одного вызова search.

    Attributes:
        k1: Насыщение частоты терма.
        b: Штраф длины документа.
        tokenizer: Токенизатор с леммами.
        doc_tokens: Токены каждого чанка в порядке входа.
        doc_lengths: Длины в токенах.
        avgdl: Средняя длина.
        df: Число чанков с термом.
        n_docs: Число чанков.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: Bm25Tokenizer,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Строит индекс из текстов чанков.

        Args:
            texts: Тексты в порядке чанков.
            tokenizer: Общий токенизатор.
            k1: Параметр Okapi.
            b: Параметр Okapi.
        """
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.doc_tokens: list[list[str]] = []
        self.doc_lengths: list[int] = []
        self.df: Counter[str] = Counter()
        for text in texts:
            tokens = tokenizer.tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            unique_terms = set(tokens)
            for term in unique_terms:
                self.df[term] += 1
        self.n_docs = len(texts)
        if self.n_docs == 0:
            self.avgdl = 0.0
        else:
            total_length = sum(self.doc_lengths)
            self.avgdl = total_length / self.n_docs

    def score_query(self, query_text: str) -> list[float]:
        """Считает BM25 по каждому чанку.

        Args:
            query_text: Строка варианта запроса.

        Returns:
            Список скоров той же длины, что и тексты индекса.
        """
        query_terms = self.tokenizer.tokenize(query_text)
        if not query_terms or self.n_docs == 0:
            return [0.0] * self.n_docs
        scores: list[float] = []
        for doc_index, tokens in enumerate(self.doc_tokens):
            scores.append(self._score_document(query_terms, doc_index, tokens))
        return scores

    def _score_document(
        self,
        query_terms: list[str],
        doc_index: int,
        tokens: list[str],
    ) -> float:
        """Суммирует вклад термов запроса для одного чанка.

        Args:
            query_terms: Леммы запроса.
            doc_index: Индекс чанка.
            tokens: Токены чанка.

        Returns:
            Скор BM25.
        """
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        doc_length = self.doc_lengths[doc_index]
        total = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = self.df[term]
            idf = math.log(
                1.0
                + (self.n_docs - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_norm = 1.0 - self.b + self.b * doc_length / self.avgdl
            tf_part = (
                frequency
                * (self.k1 + 1.0)
                / (frequency + self.k1 * length_norm)
            )
            total += idf * tf_part
        return total
