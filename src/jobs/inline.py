"""Синхронный InlineJobExecutor для PoC."""

from __future__ import annotations

from datetime import datetime, timezone

from application.service import DocumentAnalysisService
from models.jobs import Job, JobReceipt
from models.summary import EvalReport, RunSummary
from ports.jobs import JobExecutor


class InlineJobExecutor(JobExecutor):
    """Вызывает DocumentAnalysisService в том же процессе.

    Attributes:
        service: Фасад прогона.
        _jobs: Принятые задания по job_id.
    """

    def __init__(self, service: DocumentAnalysisService) -> None:
        """Сохраняет фасад.

        Args:
            service: DocumentAnalysisService.
        """
        self.service = service
        self._jobs: dict[str, Job] = {}

    def submit(self, job: Job) -> JobReceipt:
        """Регистрирует задание и сразу возвращает квитанцию.

        Args:
            job: Job.

        Returns:
            JobReceipt.
        """
        self._jobs[job.job_id] = job
        return JobReceipt(
            job_id=job.job_id,
            accepted_at=datetime.now(timezone.utc),
        )

    def wait(self, receipt: JobReceipt) -> RunSummary | EvalReport:
        """Исполняет analyze; eval отложен до спеки метрик.

        Args:
            receipt: JobReceipt.

        Returns:
            RunSummary для analyze.

        Raises:
            KeyError: Неизвестная квитанция.
            NotImplementedError: kind=eval до спеки eval.
        """
        job = self._jobs[receipt.job_id]
        if job.kind == "analyze":
            return self.service.run(job.request)
        raise NotImplementedError(
            "InlineJobExecutor.eval появится после спецификации eval"
        )
