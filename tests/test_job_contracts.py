"""T-JOB-01: JSON-раундтрип Job."""

from __future__ import annotations

from models.config import AppConfig
from models.jobs import Job
from models.requests import EvalRequest, RunRequest


class TestJobContracts:
    """Каталог T-JOB-01."""

    def test_t_job_01_json_roundtrip_analyze(self) -> None:
        """Поля Job analyze восстанавливаются идентично."""
        job = Job(
            job_id="job-1",
            kind="analyze",
            request=RunRequest(
                config=AppConfig(),
                phases=["A1", "B"],
                force_phases={"A1"},
            ),
        )
        restored = Job.from_json(job.to_json())
        assert restored.job_id == job.job_id
        assert restored.kind == "analyze"
        assert restored.request.phases == ["A1", "B"]
        assert restored.request.force_phases == {"A1"}

    def test_t_job_01_json_roundtrip_eval(self) -> None:
        """Поля Job eval восстанавливаются идентично."""
        job = Job(
            job_id="job-2",
            kind="eval",
            request=EvalRequest(
                config=AppConfig(),
                ground_truth_path="./gt.json",
                predictions_dir="./results",
            ),
        )
        restored = Job.from_json(job.to_json())
        assert restored.kind == "eval"
        assert restored.request.ground_truth_path == "./gt.json"
        assert restored.request.predictions_dir == "./results"
