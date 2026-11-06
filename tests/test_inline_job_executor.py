"""T-JOB-02: InlineJobExecutor.wait для analyze."""

from __future__ import annotations

from jobs.inline import InlineJobExecutor
from models.jobs import Job
from models.summary import RunSummary
from tests.helpers import ServiceFactory
from tests.support import RunHarness


class TestInlineJobExecutor:
    """Каталог T-JOB-02."""

    def test_t_job_02_wait_returns_run_summary(self, harness: RunHarness) -> None:
        """wait для analyze возвращает RunSummary."""
        service = ServiceFactory().build(harness)
        executor = InlineJobExecutor(service=service)
        job = Job(
            job_id="job-analyze-1",
            kind="analyze",
            request=harness.run_request(),
        )
        receipt = executor.submit(job)
        result = executor.wait(receipt)
        assert isinstance(result, RunSummary)
        assert "A1" in result.phases
