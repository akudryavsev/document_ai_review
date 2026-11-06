"""T-GW: шлюзы llama-server и запись векторов фазой A1."""

from __future__ import annotations

import time

from application.app import Application
from fakes.pipeline import FakeCleaner, FakeExtractor
from gate import ArtifactGate
from gateways.client import LlamaHttpClient, LlamaServerError
from gateways.llama import (
    LlamaEmbeddingGateway,
    LlamaLlmGateway,
    LlamaRerankerGateway,
)
from gpu import GpuLease
from models.config import AppConfig
from models.documents import ChunkBatch, ChunkRecord, DocumentRef
from models.gateways import (
    ChatMessage,
    EmbedItem,
    EmbedRequest,
    EmbedResponse,
    EmbedResponseItem,
    GenerateRequest,
    RerankPair,
    RerankRequest,
)
from models.phases import PhaseContext
from observe.logging_sink import LoggingTraceSink
from phases.indexing import IndexingPhase
from ports.gateways import EmbeddingGateway
from ports.pipeline import Chunker
from retrieval.engine import HybridSearcher
from retrieval.store_index import StoreEmbeddingIndex
from store.memory_store import InMemoryArtifactStore


class ScriptedLlamaClient(LlamaHttpClient):
    """Клиент с заранее заданными ответами, без сокета.

    Attributes:
        calls: Журнал method, path, payload.
        steps: Очередь ответов: dict, LlamaServerError или (код, тело).
    """

    def __init__(self) -> None:
        """Ставит тестовый адрес и пустой скрипт."""
        super().__init__("http://llama.test", timeout_sec=5.0, poll_interval_sec=0.0)
        self.calls: list[tuple[str, str, dict | None]] = []
        self.steps: list[dict | LlamaServerError | tuple[int, str]] = []

    def request(self, method: str, path: str, payload: dict | None) -> dict:
        """Отдаёт следующий шаг скрипта.

        Args:
            method: GET или POST.
            path: Путь.
            payload: Тело или None.

        Returns:
            Словарь JSON при шаге-объекте.

        Raises:
            LlamaServerError: Шаг — ошибка или код не 200.
        """
        self.calls.append((method, path, payload))
        step = self.steps.pop(0)
        if isinstance(step, LlamaServerError):
            raise step
        if isinstance(step, tuple):
            status, body = step
            raise self.error_for_status(status, body)
        if path == "/v1/chat/completions":
            time.sleep(0.002)
        return step

    def wait(self, seconds: float) -> None:
        """Не ждёт: опрос в тесте мгновенный.

        Args:
            seconds: Интервал, который боевой клиент проспал бы.
        """
        return None


class FixedChunker(Chunker):
    """Отдаёт заранее заданные чанки одного документа.

    Attributes:
        chunks_by_doc: doc_id → список чанков.
    """

    def __init__(self, chunks_by_doc: dict[str, list[ChunkRecord]]) -> None:
        """Сохраняет чанки.

        Args:
            chunks_by_doc: Пакеты по документам.
        """
        self.chunks_by_doc = chunks_by_doc

    def chunk(self, document):  # noqa: ANN001 — порт CleanedDocument
        """Возвращает пакет документа.

        Args:
            document: Очищенный документ.

        Returns:
            ChunkBatch.
        """
        chunks = self.chunks_by_doc.get(document.doc_id, [])
        return ChunkBatch(doc_id=document.doc_id, chunks=list(chunks))


class BytesEmbedding(EmbeddingGateway):
    """Эмбеддер без сети и без хранилища: фиксированные байты.

    Attributes:
        vector: Байты одного вектора.
        dim: Размерность.
        fail_times: Сколько первых infer бросают ошибку.
        infer_calls: Журнал запросов.
    """

    def __init__(self, vector: bytes, dim: int, fail_times: int = 0) -> None:
        """Сохраняет вектор и счётчик сбоев.

        Args:
            vector: Байты float16.
            dim: Размерность.
            fail_times: Число отказов в начале.
        """
        super().__init__()
        self.vector = vector
        self.dim = dim
        self.fail_times = fail_times
        self.infer_calls: list[EmbedRequest] = []

    def load(self) -> None:
        """Помечает шлюз загруженным."""
        self.loaded = True

    def unload(self) -> None:
        """Помечает шлюз выгруженным."""
        self.loaded = False

    def infer(self, request: EmbedRequest) -> EmbedResponse:
        """Возвращает один и тот же вектор на каждый текст.

        Args:
            request: Тексты.

        Returns:
            EmbedResponse.

        Raises:
            LlamaServerError: Остались запланированные отказы.
        """
        self.infer_calls.append(request)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise LlamaServerError("сбой infer")
        items = [
            EmbedResponseItem(
                item_id=item.item_id,
                success=True,
                dim=self.dim,
                vector=self.vector,
            )
            for item in request.items
        ]
        return EmbedResponse(items=items)


class TestLlamaGateways:
    """Каталог T-GW-EMB, T-GW-RNK, T-GW-LLM, T-GW-LOAD, T-GW-ERR, T-GW-PHASE, T-GW-APP."""

    def test_t_gw_emb_01_one_post_keeps_order(self) -> None:
        """Два текста уходят одним POST в исходном порядке."""
        client = ScriptedLlamaClient()
        gateway = self._loaded_embedding(client, batch_size=8)
        client.steps.append(self._embedding_body(2))
        response = gateway.infer(
            EmbedRequest(
                items=[
                    EmbedItem(item_id="a", text="первый"),
                    EmbedItem(item_id="b", text="второй"),
                ]
            )
        )
        method, path, payload = client.calls[-1]
        assert method == "POST"
        assert path == "/v1/embeddings"
        assert payload is not None
        assert payload["model"] == "embed"
        assert payload["encoding_format"] == "float"
        assert payload["input"] == ["первый", "второй"]
        assert [item.item_id for item in response.items] == ["a", "b"]

    def test_t_gw_emb_02_shuffled_index_becomes_float16(self) -> None:
        """Переставленные index собираются в порядок запроса, 1.0 → 00 3c."""
        client = ScriptedLlamaClient()
        gateway = self._loaded_embedding(client, batch_size=8)
        client.steps.append(
            {
                "data": [
                    {"index": 1, "embedding": [1.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        )
        response = gateway.infer(
            EmbedRequest(
                items=[
                    EmbedItem(item_id="a", text="первый"),
                    EmbedItem(item_id="b", text="второй"),
                ]
            )
        )
        assert [item.item_id for item in response.items] == ["a", "b"]
        assert response.items[0].dim == 2
        assert response.items[0].vector == b"\x00<\x00\x00"

    def test_t_gw_emb_03_batches_keep_request_order(self) -> None:
        """Пять текстов при batch_size 2 дают группы 2, 2, 1."""
        client = ScriptedLlamaClient()
        gateway = self._loaded_embedding(client, batch_size=2)
        client.steps.extend(
            [
                self._embedding_body(2),
                self._embedding_body(2),
                self._embedding_body(1),
            ]
        )
        items = [EmbedItem(item_id=f"t{index}", text=f"текст {index}") for index in range(5)]
        response = gateway.infer(EmbedRequest(items=items))
        embedding_calls = [
            payload
            for method, path, payload in client.calls
            if path == "/v1/embeddings"
        ]
        assert [len(payload["input"]) for payload in embedding_calls if payload] == [2, 2, 1]
        assert [item.item_id for item in response.items] == [f"t{index}" for index in range(5)]

    def test_t_gw_emb_04_bad_vectors_raise(self) -> None:
        """Пустой вектор, разная длина и 1e6 не возвращают EmbedResponse."""
        client = ScriptedLlamaClient()
        gateway = self._loaded_embedding(client, batch_size=8)
        client.steps.append({"data": [{"index": 0, "embedding": []}]})
        self._assert_no_response(
            gateway,
            EmbedRequest(items=[EmbedItem(item_id="a", text="пусто")]),
        )
        client.steps.append(
            {
                "data": [
                    {"index": 0, "embedding": [0.1]},
                    {"index": 1, "embedding": [0.1, 0.2]},
                ]
            }
        )
        self._assert_no_response(
            gateway,
            EmbedRequest(
                items=[
                    EmbedItem(item_id="a", text="один"),
                    EmbedItem(item_id="b", text="два"),
                ]
            ),
        )
        client.steps.append({"data": [{"index": 0, "embedding": [1e6]}]})
        self._assert_no_response(
            gateway,
            EmbedRequest(items=[EmbedItem(item_id="a", text="большой")]),
        )

    def test_t_gw_emb_05_empty_request_skips_http(self) -> None:
        """Пустой список не открывает HTTP."""
        client = ScriptedLlamaClient()
        gateway = LlamaEmbeddingGateway(client, "embed", 5.0, 8, "fp16")
        response = gateway.infer(EmbedRequest(items=[]))
        assert response.items == []
        assert client.calls == []

    def test_t_gw_rnk_01_groups_queries_and_ranks(self) -> None:
        """Два query дают два POST без top_n, ранги 2, 1, 3."""
        client = ScriptedLlamaClient()
        gateway = LlamaRerankerGateway(client, "rerank", 5.0)
        self._mark_loaded(client, gateway, "rerank")
        client.steps.extend(
            [
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.2},
                        {"index": 1, "relevance_score": 0.2},
                    ]
                },
                {"results": [{"index": 0, "relevance_score": 0.9}]},
            ]
        )
        response = gateway.infer(
            RerankRequest(
                pairs=[
                    RerankPair(pair_id="p1", query="категория", passage="первый", passage_id="c1"),
                    RerankPair(pair_id="p2", query="синоним", passage="второй", passage_id="c2"),
                    RerankPair(pair_id="p3", query="категория", passage="третий", passage_id="c3"),
                ]
            )
        )
        rerank_calls = [
            payload
            for _method, path, payload in client.calls
            if path == "/v1/rerank"
        ]
        assert len(rerank_calls) == 2
        assert "top_n" not in rerank_calls[0]
        assert "top_n" not in rerank_calls[1]
        assert rerank_calls[0]["documents"] == ["первый", "третий"]
        assert rerank_calls[1]["query"] == "синоним"
        assert [(item.pair_id, item.score, item.rank) for item in response.items] == [
            ("p1", 0.2, 2),
            ("p2", 0.9, 1),
            ("p3", 0.2, 3),
        ]

    def test_t_gw_llm_01_no_retry_inside_one_infer(self) -> None:
        """HTTP 500 срывает один infer; следующий вызов шлёт второй POST."""
        client = ScriptedLlamaClient()
        gateway = LlamaLlmGateway(client, "answer", 5.0)
        self._mark_loaded(client, gateway, "answer")
        client.steps.append((500, "unavailable"))
        request = GenerateRequest(
            messages=[
                ChatMessage(role="system", content="системный"),
                ChatMessage(role="user", content="вопрос"),
            ],
            max_output_tokens=None,
        )
        try:
            gateway.infer(request)
        except LlamaServerError as error:
            assert error.status_code == 500
        else:
            raise AssertionError("первый infer обязан бросить")
        assert len(client.calls) == 1
        assert "max_tokens" not in (client.calls[0][2] or {})
        client.steps.append(
            {
                "choices": [{"message": {"role": "assistant", "content": "ответ"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        )
        success = gateway.infer(request)
        assert len(client.calls) == 2
        assert success.text == "ответ"
        assert success.prompt_tokens == 3
        assert success.completion_tokens == 1
        assert success.latency_ms > 0

    def test_t_gw_llm_02_prompt_becomes_user_message(self) -> None:
        """Плоский prompt уходит одним сообщением user."""
        client = ScriptedLlamaClient()
        gateway = LlamaLlmGateway(client, "answer", 5.0)
        self._mark_loaded(client, gateway, "answer")
        client.steps.append(
            {
                "choices": [{"message": {"content": "да"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )
        gateway.infer(GenerateRequest(prompt="только промпт"))
        payload = client.calls[-1][2]
        assert payload is not None
        assert payload["messages"] == [{"role": "user", "content": "только промпт"}]

    def test_t_gw_llm_03_missing_usage(self) -> None:
        """Нет блока usage — LlamaServerError, нули не подставляются."""
        client = ScriptedLlamaClient()
        gateway = LlamaLlmGateway(client, "answer", 5.0)
        self._mark_loaded(client, gateway, "answer")
        client.steps.append({"choices": [{"message": {"content": "текст"}}]})
        try:
            gateway.infer(GenerateRequest(prompt="вопрос"))
        except LlamaServerError:
            return
        raise AssertionError("infer без usage обязан бросить")

    def test_t_gw_load_01_embed_preset(self) -> None:
        """load пресета embed: POST /models/load и GET /models со статусом loaded."""
        client = ScriptedLlamaClient()
        gateway = LlamaEmbeddingGateway(client, "embed", 5.0, 8, "fp16")
        client.steps.extend(
            [
                {"success": True},
                {"data": [{"id": "embed", "status": {"value": "loaded"}}]},
            ]
        )
        gateway.load()
        assert client.calls[0] == ("POST", "/models/load", {"model": "embed"})
        assert client.calls[1][0:2] == ("GET", "/models")
        assert gateway.loaded is True

    def test_t_gw_load_02_lease_switches_one_preset(self) -> None:
        """GpuLease грузит embedding, reranker, llm по очереди и выгружает предыдущий."""
        client = ScriptedLlamaClient()
        embedding = LlamaEmbeddingGateway(client, "embed", 5.0, 8, "fp16")
        reranker = LlamaRerankerGateway(client, "rerank", 5.0)
        llm = LlamaLlmGateway(client, "answer", 5.0)
        lease = GpuLease(LoggingTraceSink())
        for preset in ("embed", "rerank", "answer"):
            client.steps.append({"success": True})
            client.steps.append(
                {"data": [{"id": preset, "status": {"value": "loaded"}}]}
            )
        client.steps.insert(2, {"success": True})
        client.steps.insert(5, {"success": True})
        lease.acquire(embedding)
        assert [embedding.loaded, reranker.loaded, llm.loaded] == [True, False, False]
        lease.acquire(reranker)
        assert [embedding.loaded, reranker.loaded, llm.loaded] == [False, True, False]
        lease.acquire(llm)
        assert [embedding.loaded, reranker.loaded, llm.loaded] == [False, False, True]
        assert lease.acquire_order == ["embedding", "reranker", "llm"]
        unload_models = [
            payload["model"]
            for method, path, payload in client.calls
            if path == "/models/unload" and payload is not None
        ]
        assert unload_models == ["embed", "rerank"]

    def test_t_gw_load_03_unload_404_is_success(self) -> None:
        """Выгрузка уже снятого пресета с кодом 404 не бросает исключение."""
        client = ScriptedLlamaClient()
        gateway = LlamaEmbeddingGateway(client, "embed", 5.0, 8, "fp16")
        client.steps.append((404, "already unloaded"))
        gateway.unload()
        assert gateway.loaded is False

    def test_t_gw_err_01_connection_timeout_and_unknown_model(self) -> None:
        """Обрыв и таймаут без кода, 404 с телом model — с кодом и фразой."""
        client = ScriptedLlamaClient()
        gateway = LlamaEmbeddingGateway(client, "embed", 5.0, 8, "fp16")
        client.steps.append(LlamaServerError("обрыв соединения"))
        try:
            gateway.load()
        except LlamaServerError as error:
            assert error.status_code is None
        else:
            raise AssertionError("обрыв обязан бросить")
        client.steps.append(LlamaServerError("таймаут"))
        try:
            gateway.load()
        except LlamaServerError as error:
            assert error.status_code is None
        else:
            raise AssertionError("таймаут обязан бросить")
        client.steps.append((404, "model was not found"))
        try:
            gateway.load()
        except LlamaServerError as error:
            assert error.status_code == 404
            assert "нетот модели" in str(error)
            assert "404" in str(error)
        else:
            raise AssertionError("404 обязан бросить")

    def test_t_gw_phase_01_saves_concatenated_bytes(self) -> None:
        """Фаза пишет склейку байтов шлюза и не передаёт ему хранилище."""
        vector = b"\x00<\x00\x00"
        gateway = BytesEmbedding(vector=vector, dim=2)
        store, report = self._run_indexing(
            gateway,
            {
                "doc_a.docx": [
                    ChunkRecord(chunk_id="c_0000", text="альфа", chunk_index=0),
                    ChunkRecord(chunk_id="c_0001", text="бета", chunk_index=1),
                ]
            },
            [self._document("doc_a.docx")],
        )
        matrix = store.load_chunk_vectors("doc_a.docx")
        assert report.failed == 0
        assert matrix is not None
        assert matrix.dim == 2
        assert matrix.payload == vector + vector
        assert "store" not in vars(gateway)

    def test_t_gw_phase_02_same_text_is_embedded_once(self) -> None:
        """Второй чанк с тем же текстом не порождает второй HTTP-эмбеддинг."""
        client = ScriptedLlamaClient()
        gateway = LlamaEmbeddingGateway(client, "embed", 5.0, 8, "fp16")
        client.steps.extend(
            [
                {"success": True},
                {"data": [{"id": "embed", "status": {"value": "loaded"}}]},
                self._embedding_body(1),
            ]
        )
        store, _report = self._run_indexing(
            gateway,
            {
                "doc_a.docx": [
                    ChunkRecord(chunk_id="c_0000", text="одинаковый", chunk_index=0),
                    ChunkRecord(chunk_id="c_0001", text="одинаковый", chunk_index=1),
                ]
            },
            [self._document("doc_a.docx")],
        )
        embedding_inputs = [
            payload["input"]
            for _method, path, payload in client.calls
            if path == "/v1/embeddings" and payload is not None
        ]
        assert embedding_inputs == [["одинаковый"]]
        matrix = store.load_chunk_vectors("doc_a.docx")
        assert matrix is not None
        assert matrix.chunk_ids == ["c_0000", "c_0001"]
        assert matrix.payload == matrix.payload[: matrix.dim * 2] * 2

    def test_t_gw_phase_03_infer_failure_skips_save(self) -> None:
        """Сбой infer не пишет чанки документа, следующий документ считается."""
        gateway = BytesEmbedding(vector=b"\x00<", dim=1, fail_times=1)
        store, report = self._run_indexing(
            gateway,
            {
                "doc_a.docx": [
                    ChunkRecord(chunk_id="c_0000", text="первый", chunk_index=0),
                ],
                "doc_b.pdf": [
                    ChunkRecord(chunk_id="c_0000", text="второй", chunk_index=0),
                ],
            },
            [self._document("doc_a.docx"), self._document("doc_b.pdf", suffix=".pdf")],
        )
        assert store.load_chunks("doc_a.docx") is None
        assert store.load_chunk_vectors("doc_a.docx") is None
        assert store.load_chunk_vectors("doc_b.pdf") is not None
        assert report.failed == 1
        assert report.computed == 1

    def test_t_gw_app_01_build_uses_llama_and_hybrid_search(self) -> None:
        """Корень собирает три шлюза llama-server и HybridSearcher."""
        service = Application(AppConfig()).build()
        retriever = service.retrieval.retriever
        assert isinstance(service.indexing.embedding, LlamaEmbeddingGateway)
        assert isinstance(service.retrieval.reranker, LlamaRerankerGateway)
        assert isinstance(service.answering.llm, LlamaLlmGateway)
        assert isinstance(retriever, HybridSearcher)
        assert isinstance(retriever.embeddings, StoreEmbeddingIndex)
        assert service.retrieval.vector_index is retriever.embeddings
        assert service.answering.verifier is None

    def _loaded_embedding(
        self,
        client: ScriptedLlamaClient,
        batch_size: int,
    ) -> LlamaEmbeddingGateway:
        """Грузит эмбеддер заранее заскриптованными ответами.

        Args:
            client: Скриптованный клиент.
            batch_size: Размер батча.

        Returns:
            Загруженный шлюз. Журнал load очищен.
        """
        gateway = LlamaEmbeddingGateway(client, "embed", 5.0, batch_size, "fp16")
        self._mark_loaded(client, gateway, "embed")
        return gateway

    def _mark_loaded(
        self,
        client: ScriptedLlamaClient,
        gateway: LlamaEmbeddingGateway | LlamaRerankerGateway | LlamaLlmGateway,
        preset: str,
    ) -> None:
        """Выполняет load и забывает его вызовы.

        Args:
            client: Скриптованный клиент.
            gateway: Шлюз.
            preset: Имя секции.
        """
        client.steps.extend(
            [
                {"success": True},
                {"data": [{"id": preset, "status": {"value": "loaded"}}]},
            ]
        )
        gateway.load()
        client.calls.clear()

    def _embedding_body(self, count: int) -> dict:
        """Ответ /v1/embeddings с вектором [1.0, 0.0] на каждую строку.

        Args:
            count: Число строк входа.

        Returns:
            JSON-объект data.
        """
        return {
            "data": [
                {"index": index, "embedding": [1.0, 0.0]} for index in range(count)
            ]
        }

    def _assert_no_response(self, gateway: LlamaEmbeddingGateway, request: EmbedRequest) -> None:
        """Проверяет, что infer бросает и не возвращает объект.

        Args:
            gateway: Загруженный эмбеддер.
            request: Запрос, который сервер испортил.

        Raises:
            AssertionError: infer вернул ответ.
            LlamaServerError: Ожидаемый исход, перехватывается здесь.
        """
        try:
            gateway.infer(request)
        except LlamaServerError:
            return
        raise AssertionError("infer обязан бросить LlamaServerError")

    def _run_indexing(
        self,
        embedding: EmbeddingGateway,
        chunks_by_doc: dict[str, list[ChunkRecord]],
        documents: list[DocumentRef],
    ) -> tuple[InMemoryArtifactStore, object]:
        """Прогоняет A1 на одном хранилище в памяти.

        Args:
            embedding: Шлюз эмбеддера.
            chunks_by_doc: Чанки документов.
            documents: Список DocumentRef.

        Returns:
            Хранилище и отчёт фазы.
        """
        trace = LoggingTraceSink()
        store = InMemoryArtifactStore()
        phase = IndexingPhase(
            extractor=FakeExtractor(),
            cleaner=FakeCleaner(),
            chunker=FixedChunker(chunks_by_doc),
            embedding=embedding,
            store=store,
            gate=ArtifactGate(),
            lease=GpuLease(trace),
            trace_sink=trace,
        )
        context = PhaseContext(
            config=AppConfig(),
            documents=documents,
            questions=None,
            requested_phases=["A1"],
        )
        report = phase.execute(phase.plan(context), context)
        return store, report

    def _document(self, doc_id: str, suffix: str = ".docx") -> DocumentRef:
        """Собирает ссылку на документ.

        Args:
            doc_id: Идентификатор.
            suffix: Расширение.

        Returns:
            DocumentRef.
        """
        return DocumentRef(doc_id=doc_id, path=f"/tmp/{doc_id}", suffix=suffix)
