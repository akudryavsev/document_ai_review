"""T-GPU: порядок загрузок шлюзов."""

from __future__ import annotations

from tests.helpers import ServiceFactory
from tests.support import RunHarness


class TestGpuOrder:
    """Каталог T-GPU-01…02."""

    def test_t_gpu_01_full_run_order(self, harness: RunHarness) -> None:
        """Полный A1→A2→B: эмбеддер → реранкер → LLM; два шлюза не вместе."""
        service = ServiceFactory().build(harness)
        service.run(harness.run_request())
        assert harness.lease.acquire_order == ["embedding", "reranker", "llm"]
        # После прогона ничего не загружено.
        assert harness.embedding.loaded is False
        assert harness.reranker.loaded is False
        assert harness.llm.loaded is False

    def test_t_gpu_02_skipped_phase_does_not_load(self, harness: RunHarness) -> None:
        """Пропуск фазы по хешам — модель пропущенной фазы не загружается."""
        service = ServiceFactory().build(harness)
        service.run(harness.run_request())
        harness.embedding.load_calls = 0
        harness.reranker.load_calls = 0
        harness.llm.load_calls = 0
        harness.lease.acquire_order.clear()
        service.run(harness.run_request())
        assert harness.embedding.load_calls == 0
        assert harness.reranker.load_calls == 0
        assert harness.llm.load_calls == 0
        assert harness.lease.acquire_order == []
