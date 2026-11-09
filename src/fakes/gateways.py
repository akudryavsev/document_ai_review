"""Подделки шлюзов моделей."""

from __future__ import annotations

import struct

from models.gateways import (
    EmbedRequest,
    EmbedResponse,
    EmbedResponseItem,
    GenerateRequest,
    GenerateResponse,
    RerankRequest,
    RerankResponse,
    RerankResponseItem,
)
from ports.gateways import EmbeddingGateway, LlmGateway, RerankerGateway


class FakeEmbeddingGateway(EmbeddingGateway):
    """Эмбеддер-подделка с записью вызовов.

    Attributes:
        name: Имя шлюза.
        load_calls: Число load.
        unload_calls: Число unload.
        infer_calls: Список запросов infer.
        raise_error: Если задано — infer бросает это исключение.
    """

    name = "embedding"

    def __init__(self, raise_error: Exception | None = None) -> None:
        """Инициализирует счётчики.

        Args:
            raise_error: Исключение infer или None.
        """
        super().__init__()
        self.load_calls = 0
        self.unload_calls = 0
        self.infer_calls: list[EmbedRequest] = []
        self.raise_error = raise_error

    def load(self) -> None:
        """Помечает шлюз загруженным."""
        self.load_calls += 1
        self.loaded = True

    def unload(self) -> None:
        """Помечает шлюз выгруженным."""
        self.unload_calls += 1
        self.loaded = False

    def infer(self, request: EmbedRequest) -> EmbedResponse:
        """Возвращает успех по каждому item_id.

        Args:
            request: Запрос эмбеддинга.

        Returns:
            EmbedResponse без тензоров.

        Raises:
            RuntimeError: Шлюз не загружен.
            Exception: Если задан raise_error.
        """
        if not self.loaded:
            raise RuntimeError("EmbeddingGateway не загружен")
        if self.raise_error is not None:
            raise self.raise_error
        self.infer_calls.append(request)
        vector = struct.pack("<e", 1.0)
        items = [
            EmbedResponseItem(
                item_id=item.item_id,
                success=True,
                dim=1,
                vector=vector,
                cache_key=item.cache_key,
            )
            for item in request.items
        ]
        return EmbedResponse(items=items)


class FakeRerankerGateway(RerankerGateway):
    """Реранкер-подделка с записью вызовов.

    Attributes:
        name: Имя шлюза.
        load_calls: Число load.
        unload_calls: Число unload.
        infer_calls: Список запросов infer.
        default_score: Скор по умолчанию.
    """

    name = "reranker"

    def __init__(
        self,
        default_score: float = 0.9,
        scores_by_pair: dict[str, float] | None = None,
        omit_pair_ids: set[str] | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        """Инициализирует счётчики и скор.

        Args:
            default_score: Скор для пар без явного значения.
            scores_by_pair: Явные скоры по pair_id.
            omit_pair_ids: Пары, которые не попадут в ответ.
            raise_error: Если задано — infer бросает это исключение.
        """
        super().__init__()
        self.load_calls = 0
        self.unload_calls = 0
        self.infer_calls: list[RerankRequest] = []
        self.default_score = default_score
        self.scores_by_pair = scores_by_pair or {}
        self.omit_pair_ids = omit_pair_ids or set()
        self.raise_error = raise_error

    def load(self) -> None:
        """Помечает шлюз загруженным."""
        self.load_calls += 1
        self.loaded = True

    def unload(self) -> None:
        """Помечает шлюз выгруженным."""
        self.unload_calls += 1
        self.loaded = False

    def infer(self, request: RerankRequest) -> RerankResponse:
        """Возвращает скоры пар с убывающими рангами.

        Args:
            request: Пары query × passage.

        Returns:
            RerankResponse.

        Raises:
            RuntimeError: Шлюз не загружен.
            Exception: Если задан raise_error.
        """
        if not self.loaded:
            raise RuntimeError("RerankerGateway не загружен")
        self.infer_calls.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        items: list[RerankResponseItem] = []
        rank = 1
        for pair in request.pairs:
            if pair.pair_id in self.omit_pair_ids:
                continue
            score = self.scores_by_pair.get(pair.pair_id, self.default_score)
            items.append(
                RerankResponseItem(
                    pair_id=pair.pair_id,
                    score=score,
                    rank=rank,
                )
            )
            rank += 1
        return RerankResponse(items=items)


class FakeLlmGateway(LlmGateway):
    """LLM-подделка с настраиваемыми отказами.

    Attributes:
        name: Имя шлюза.
        load_calls: Число load.
        unload_calls: Число unload.
        infer_calls: Список запросов.
        fail_times: Сколько первых вызовов бросают ошибку.
        response_text: Текст успешного ответа.
    """

    name = "llm"

    def __init__(
        self,
        fail_times: int = 0,
        response_text: str = (
            '{"found": true, "answer": "ok", "explanation": "stub", '
            '"reason": null, "citations": [{"n": 1, "quote": "ok"}]}'
        ),
    ) -> None:
        """Инициализирует поведение.

        Args:
            fail_times: Число отказов подряд.
            response_text: Текст успешного ответа.
        """
        super().__init__()
        self.load_calls = 0
        self.unload_calls = 0
        self.infer_calls: list[GenerateRequest] = []
        self.fail_times = fail_times
        self._remaining_failures = fail_times
        self.response_text = response_text

    def load(self) -> None:
        """Помечает шлюз загруженным."""
        self.load_calls += 1
        self.loaded = True

    def unload(self) -> None:
        """Помечает шлюз выгруженным."""
        self.unload_calls += 1
        self.loaded = False

    def infer(self, request: GenerateRequest) -> GenerateResponse:
        """Возвращает текст или бросает ошибку по счётчику.

        Args:
            request: Промпт.

        Returns:
            GenerateResponse.

        Raises:
            RuntimeError: Шлюз не загружен или исчерпаны попытки теста.
        """
        if not self.loaded:
            raise RuntimeError("LlmGateway не загружен")
        self.infer_calls.append(request)
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("LLM temporary failure")
        return GenerateResponse(
            text=self.response_text,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
        )
