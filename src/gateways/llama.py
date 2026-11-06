"""Шлюзы эмбеддера, реранкера и чата к одному llama-server."""

from __future__ import annotations

import time
from collections.abc import Sequence

from gateways.client import LlamaHttpClient, LlamaServerError
from gateways.float16 import Float16Codec
from models.gateways import (
    EmbedItem,
    EmbedRequest,
    EmbedResponse,
    EmbedResponseItem,
    GenerateRequest,
    GenerateResponse,
    RerankPair,
    RerankRequest,
    RerankResponse,
    RerankResponseItem,
)
from ports.gateways import EmbeddingGateway, LlmGateway, RerankerGateway


class LlamaPresetControl:
    """Переключает один пресет llama-server.

    Attributes:
        client: HTTP-клиент.
        preset: Имя секции models.ini.
        timeout_sec: Бюджет ожидания load.
        loaded: Пресет считается загруженным.
    """

    def __init__(
        self,
        client: LlamaHttpClient,
        preset: str,
        timeout_sec: float,
    ) -> None:
        """Сохраняет клиент и пресет.

        Args:
            client: LlamaHttpClient.
            preset: Имя секции.
            timeout_sec: Сколько ждать статус loaded.
        """
        self.client = client
        self.preset = preset
        self.timeout_sec = timeout_sec
        self.loaded = False

    def load(self) -> None:
        """Грузит пресет и ждёт статус loaded.

        Повторный вызов при уже загруженном пресете в сеть не ходит.

        Raises:
            LlamaServerError: Сервер не подтвердил пресет.
        """
        if self.loaded:
            return
        deadline = time.monotonic() + self.timeout_sec
        accepted = self.client.request(
            "POST",
            "/models/load",
            {"model": self.preset},
        )
        if accepted.get("success") is not True:
            raise LlamaServerError("нетот модели")
        while True:
            if time.monotonic() > deadline:
                self._abort_load("таймаут загрузки пресета")
            listing = self.client.request("GET", "/models", None)
            status = self._preset_status(listing)
            if status == "loaded":
                self.loaded = True
                return
            if status is None or status == "loading":
                self._sleep_until(deadline)
                continue
            self._abort_load(f"нетот модели: статус {status}")

    def unload(self) -> None:
        """Снимает пресет. Уже снятый пресет (400 или 404) — успех.

        Raises:
            LlamaServerError: Сеть, таймаут или прочий код.
        """
        try:
            payload = self.client.request(
                "POST",
                "/models/unload",
                {"model": self.preset},
            )
        except LlamaServerError as error:
            if error.status_code in (400, 404):
                self.loaded = False
                return
            raise
        if payload.get("success") is not True:
            raise LlamaServerError("выгрузка пресета не подтверждена")
        self.loaded = False

    def _preset_status(self, listing: dict) -> str | None:
        """Ищет status.value пресета в ответе GET /models.

        Args:
            listing: JSON маршрутизатора.

        Returns:
            Строка статуса или None, если пресета в списке ещё нет.

        Raises:
            LlamaServerError: В ответе нет списка data.
        """
        data = listing.get("data")
        if not isinstance(data, list):
            raise LlamaServerError("нетот модели: нет списка моделей")
        for item in data:
            if not isinstance(item, dict) or item.get("id") != self.preset:
                continue
            status = item.get("status")
            if isinstance(status, dict) and isinstance(status.get("value"), str):
                return status["value"]
            return None
        return None

    def _sleep_until(self, deadline: float) -> None:
        """Ждёт интервал опроса, не дольше остатка бюджета.

        Args:
            deadline: Момент таймаута load.

        Raises:
            LlamaServerError: Бюджет уже исчерпан.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._abort_load("таймаут загрузки пресета")
        pause = min(self.client.poll_interval_sec, remaining)
        self.client.wait(pause)

    def _abort_load(self, message: str) -> None:
        """Один раз снимает пресет и бросает ошибку загрузки.

        Args:
            message: Текст LlamaServerError.

        Raises:
            LlamaServerError: Всегда.
        """
        try:
            self.unload()
        except LlamaServerError:
            self.loaded = False
        raise LlamaServerError(message)

    def _require_loaded(self) -> None:
        """Не пускает infer к незагруженному пресету.

        Raises:
            LlamaServerError: Шлюз не загружен.
        """
        if not self.loaded:
            raise LlamaServerError("шлюз не загружен")


class LlamaEmbeddingGateway(LlamaPresetControl, EmbeddingGateway):
    """Эмбеддер: POST /v1/embeddings, на границе байты float16.

    Attributes:
        name: Имя для GpuLease.
        batch_size: Длина одной группы input.
        codec: Кодер float16.
    """

    name = "embedding"

    def __init__(
        self,
        client: LlamaHttpClient,
        preset: str,
        timeout_sec: float,
        batch_size: int,
        dtype: str,
    ) -> None:
        """Сохраняет клиент, пресет и размер батча.

        Args:
            client: LlamaHttpClient.
            preset: Секция embed.
            timeout_sec: Бюджет load.
            batch_size: Сколько строк в одном POST.
            dtype: Должен быть fp16.

        Raises:
            ValueError: batch_size меньше 1 или dtype не fp16.
        """
        super().__init__(client, preset, timeout_sec)
        if batch_size < 1:
            raise ValueError("embedding.batch_size должен быть не меньше 1")
        if dtype != "fp16":
            raise ValueError("embedding.dtype для llama-server должен быть fp16")
        self.batch_size = batch_size
        self.codec = Float16Codec()

    def infer(self, request: EmbedRequest) -> EmbedResponse:
        """Считает эмбеддинги батчами и возвращает байты float16.

        Args:
            request: Тексты и идентификаторы.

        Returns:
            EmbedResponse в порядке запроса.

        Raises:
            LlamaServerError: Шлюз не загружен или ответ сервера не годится.
        """
        if not request.items:
            return EmbedResponse(items=[])
        self._require_loaded()
        encoded: list[tuple[EmbedItem, bytes, int]] = []
        for group in self._batches(request.items):
            body = {
                "model": self.preset,
                "input": [item.text for item in group],
                "encoding_format": "float",
            }
            payload = self.client.request("POST", "/v1/embeddings", body)
            encoded.extend(self._read_group(group, payload))
        self._same_dim(encoded)
        items = [
            EmbedResponseItem(
                item_id=item.item_id,
                success=True,
                dim=dim,
                vector=vector,
                cache_key=item.cache_key,
            )
            for item, vector, dim in encoded
        ]
        return EmbedResponse(items=items)

    def _batches(self, items: Sequence[EmbedItem]) -> list[list[EmbedItem]]:
        """Режет список на группы не длиннее batch_size.

        Args:
            items: Все тексты вызова.

        Returns:
            Группы в исходном порядке.
        """
        groups: list[list[EmbedItem]] = []
        start = 0
        while start < len(items):
            groups.append(list(items[start : start + self.batch_size]))
            start += self.batch_size
        return groups

    def _read_group(
        self,
        group: Sequence[EmbedItem],
        payload: dict,
    ) -> list[tuple[EmbedItem, bytes, int]]:
        """Раскладывает data по index и кодирует векторы.

        Args:
            group: Строки этого POST.
            payload: Ответ сервера.

        Returns:
            Тройки в порядке группы.

        Raises:
            LlamaServerError: Битый index, пустой или разный вектор.
        """
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(group):
            raise LlamaServerError("битый index эмбеддинга")
        by_index: dict[int, bytes] = {}
        dim: int | None = None
        for row in data:
            if not isinstance(row, dict) or not isinstance(row.get("index"), int):
                raise LlamaServerError("битый index эмбеддинга")
            index = row["index"]
            if isinstance(index, bool) or index < 0 or index >= len(group) or index in by_index:
                raise LlamaServerError("битый index эмбеддинга")
            embedding = row.get("embedding")
            if not isinstance(embedding, list):
                raise LlamaServerError("пустой вектор")
            vector = self.codec.encode(embedding)
            row_dim = len(vector) // 2
            if dim is None:
                dim = row_dim
            elif row_dim != dim:
                raise LlamaServerError("разная размерность")
            by_index[index] = vector
        if dim is None or len(by_index) != len(group):
            raise LlamaServerError("битый index эмбеддинга")
        return [
            (item, by_index[index], dim)
            for index, item in enumerate(group)
        ]

    def _same_dim(self, encoded: Sequence[tuple[EmbedItem, bytes, int]]) -> None:
        """Проверяет одну размерность на весь infer.

        Args:
            encoded: Уже разобранные группы.

        Raises:
            LlamaServerError: Размерности групп различаются.
        """
        if not encoded:
            return
        expected = encoded[0][2]
        for _item, vector, dim in encoded:
            if dim != expected or len(vector) != expected * 2 or len(vector) == 0:
                raise LlamaServerError("разная размерность")


class LlamaRerankerGateway(LlamaPresetControl, RerankerGateway):
    """Реранкер: группы POST /v1/rerank по точному тексту query.

    Attributes:
        name: Имя для GpuLease.
    """

    name = "reranker"

    def infer(self, request: RerankRequest) -> RerankResponse:
        """Скорит пары и ставит плотный ранг по всему вызову.

        Args:
            request: Пары query × passage.

        Returns:
            Скоры и ранги в порядке запроса.

        Raises:
            LlamaServerError: Шлюз не загружен или в ответе нет индекса.
        """
        if not request.pairs:
            return RerankResponse(items=[])
        self._require_loaded()
        scores = [0.0] * len(request.pairs)
        for query, indexes in self._groups(request.pairs):
            documents = [request.pairs[index].passage for index in indexes]
            payload = self.client.request(
                "POST",
                "/v1/rerank",
                {
                    "model": self.preset,
                    "query": query,
                    "documents": documents,
                },
            )
            self._apply_scores(payload, indexes, scores)
        ranks = self._ranks(scores)
        items = [
            RerankResponseItem(
                pair_id=pair.pair_id,
                score=scores[index],
                rank=ranks[index],
            )
            for index, pair in enumerate(request.pairs)
        ]
        return RerankResponse(items=items)

    def _groups(self, pairs: Sequence[RerankPair]) -> list[tuple[str, list[int]]]:
        """Группирует индексы по тексту query, сохраняя порядок появления.

        Args:
            pairs: Пары вызова.

        Returns:
            Пары (query, индексы в запросе).
        """
        order: list[str] = []
        grouped: dict[str, list[int]] = {}
        for index, pair in enumerate(pairs):
            if pair.query not in grouped:
                order.append(pair.query)
                grouped[pair.query] = []
            grouped[pair.query].append(index)
        return [(query, grouped[query]) for query in order]

    def _apply_scores(
        self,
        payload: dict,
        indexes: Sequence[int],
        scores: list[float],
    ) -> None:
        """Кладёт relevance_score группы в общий список.

        Args:
            payload: Ответ /v1/rerank.
            indexes: Индексы пар этой группы.
            scores: Общий список скоров.

        Raises:
            LlamaServerError: Пропущенный или повторный index.
        """
        results = payload.get("results")
        if not isinstance(results, list):
            raise LlamaServerError("битый index реранка")
        seen: set[int] = set()
        for row in results:
            if not isinstance(row, dict) or not isinstance(row.get("index"), int):
                raise LlamaServerError("битый index реранка")
            local = row["index"]
            if isinstance(local, bool) or local < 0 or local >= len(indexes) or local in seen:
                raise LlamaServerError("битый index реранка")
            score = row.get("relevance_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise LlamaServerError("битый index реранка")
            seen.add(local)
            scores[indexes[local]] = float(score)
        if seen != set(range(len(indexes))):
            raise LlamaServerError("битый index реранка")

    def _ranks(self, scores: Sequence[float]) -> list[int]:
        """Ставит плотный ранг: скор по убыванию, при равенстве раньше в запросе.

        Args:
            scores: Скоры в порядке запроса.

        Returns:
            Ранги с 1 той же длины.
        """
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        ranks = [0] * len(scores)
        for rank, index in enumerate(order, start=1):
            ranks[index] = rank
        return ranks


class LlamaLlmGateway(LlamaPresetControl, LlmGateway):
    """Чат: один POST /v1/chat/completions на infer.

    Attributes:
        name: Имя для GpuLease.
    """

    name = "llm"

    def infer(self, request: GenerateRequest) -> GenerateResponse:
        """Отправляет сообщения и возвращает текст, usage и latency.

        Args:
            request: Сообщения или плоский prompt.

        Returns:
            GenerateResponse.

        Raises:
            LlamaServerError: Пустой запрос, шлюз не загружен или нет usage.
        """
        self._require_loaded()
        messages = self._messages(request)
        body: dict = {
            "model": self.preset,
            "messages": messages,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.max_output_tokens is not None:
            body["max_tokens"] = request.max_output_tokens
        started = time.perf_counter()
        payload = self.client.request("POST", "/v1/chat/completions", body)
        latency_ms = (time.perf_counter() - started) * 1000.0
        text, prompt_tokens, completion_tokens = self._read_completion(payload)
        return GenerateResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    def _messages(self, request: GenerateRequest) -> list[dict[str, str]]:
        """Собирает messages без добавления prompt поверх готового списка.

        Args:
            request: Запрос генерации.

        Returns:
            Список сообщений для сервера.

        Raises:
            LlamaServerError: Нет ни messages, ни prompt.
        """
        if request.messages:
            return [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ]
        if request.prompt:
            return [{"role": "user", "content": request.prompt}]
        raise LlamaServerError("пустой запрос к чату")

    def _read_completion(self, payload: dict) -> tuple[str, int, int]:
        """Достаёт текст и счётчики usage.

        Args:
            payload: Ответ chat completions.

        Returns:
            Текст, prompt_tokens и completion_tokens.

        Raises:
            LlamaServerError: Нет choices, content или usage.
        """
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlamaServerError("пустой choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise LlamaServerError("пустой choices")
        message = first.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LlamaServerError("нет content")
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise LlamaServerError("нет usage")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
            raise LlamaServerError("нет usage")
        if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int):
            raise LlamaServerError("нет usage")
        return message["content"], prompt_tokens, completion_tokens
