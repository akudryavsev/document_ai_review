"""Решения ArtifactGate."""

from __future__ import annotations

from gate import ArtifactGate
from models.artifacts import ArtifactMeta


class TestArtifactGate:
    """Юнит-тесты инвалидации без I/O."""

    def test_missing_artifact_computes(self) -> None:
        """Нет meta — compute/missing_artifact."""
        decision = ArtifactGate().decide(
            None,
            schema_version="1.0",
            chunking_hash="abc",
        )
        assert decision.action == "compute"
        assert decision.reason == "missing_artifact"

    def test_force_computes(self) -> None:
        """force=True — compute/force даже при валидных хешах."""
        meta = ArtifactMeta(schema_version="1.0", chunking_hash="abc")
        decision = ArtifactGate().decide(
            meta,
            schema_version="1.0",
            force=True,
            chunking_hash="abc",
        )
        assert decision.action == "compute"
        assert decision.reason == "force"

    def test_hashes_valid_skips(self) -> None:
        """Совпавшие хеши — skip/hashes_valid."""
        meta = ArtifactMeta(
            schema_version="1.0",
            chunking_hash="c1",
            retrieval_config_hash="r1",
            reranker_config_hash="rr1",
        )
        decision = ArtifactGate().decide(
            meta,
            schema_version="1.0",
            chunking_hash="c1",
            retrieval_config_hash="r1",
            reranker_config_hash="rr1",
        )
        assert decision.action == "skip"
        assert decision.reason == "hashes_valid"

    def test_hash_mismatch_computes(self) -> None:
        """Смена chunking_hash — compute/hash_mismatch."""
        meta = ArtifactMeta(schema_version="1.0", chunking_hash="old")
        decision = ArtifactGate().decide(
            meta,
            schema_version="1.0",
            chunking_hash="new",
        )
        assert decision.action == "compute"
        assert decision.reason == "hash_mismatch"
