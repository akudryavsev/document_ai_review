"""Абстрактный исполнитель заданий."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.jobs import Job, JobReceipt
from models.summary import EvalReport, RunSummary


class JobExecutor(ABC):
    """Очередь заданий над фасадом."""

    @abstractmethod
    def submit(self, job: Job) -> JobReceipt:
        """Принимает задание.

        Args:
            job: Job.

        Returns:
            JobReceipt.
        """

    @abstractmethod
    def wait(self, receipt: JobReceipt) -> RunSummary | EvalReport:
        """Ждёт завершения и возвращает результат.

        Args:
            receipt: JobReceipt.

        Returns:
            RunSummary или EvalReport.
        """
