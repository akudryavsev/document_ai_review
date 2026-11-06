"""Порядок аренды GpuLease."""

from __future__ import annotations

from fakes.gateways import (
    FakeEmbeddingGateway,
    FakeLlmGateway,
    FakeRerankerGateway,
)
from fakes.trace import RecordingTraceSink
from gpu import GpuLease


class TestGpuLease:
    """Инвариант: не больше одного загруженного шлюза."""

    def test_acquire_unloads_previous(self) -> None:
        """Новый acquire выгружает предыдущий шлюз."""
        sink = RecordingTraceSink()
        lease = GpuLease(sink)
        embedding = FakeEmbeddingGateway()
        reranker = FakeRerankerGateway()
        lease.acquire(embedding)
        assert embedding.loaded is True
        lease.acquire(reranker)
        assert embedding.loaded is False
        assert reranker.loaded is True
        assert lease.acquire_order == ["embedding", "reranker"]

    def test_release_unloads_current(self) -> None:
        """release выгружает текущий шлюз."""
        sink = RecordingTraceSink()
        lease = GpuLease(sink)
        llm = FakeLlmGateway()
        lease.acquire(llm)
        lease.release()
        assert llm.loaded is False
        assert lease.current is None
