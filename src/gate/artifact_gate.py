"""Решение о пересчёте или reuse артефакта."""

from __future__ import annotations

from models.artifacts import ArtifactMeta
from models.phases import ReuseDecision


class ArtifactGate:
    """Чистый класс инвалидации артефактов без диска и GPU."""

    def decide(
        self,
        meta: ArtifactMeta | None,
        *,
        schema_version: str,
        force: bool = False,
        chunking_hash: str | None = None,
        retrieval_config_hash: str | None = None,
        reranker_config_hash: str | None = None,
    ) -> ReuseDecision:
        """Сравнивает meta с ожидаемыми хешами.

        Args:
            meta: Метаданные артефакта или None при отсутствии.
            schema_version: Ожидаемая версия схемы.
            force: Принудительный пересчёт.
            chunking_hash: Ожидаемый chunking_hash, если применим.
            retrieval_config_hash: Ожидаемый retrieval hash.
            reranker_config_hash: Ожидаемый reranker hash.

        Returns:
            ReuseDecision с action compute или skip.
        """
        if force:
            return ReuseDecision(action="compute", reason="force")
        if meta is None:
            return ReuseDecision(action="compute", reason="missing_artifact")
        if meta.schema_version != schema_version:
            return ReuseDecision(action="compute", reason="hash_mismatch")
        expected = {
            "chunking_hash": chunking_hash,
            "retrieval_config_hash": retrieval_config_hash,
            "reranker_config_hash": reranker_config_hash,
        }
        for field_name, expected_value in expected.items():
            if expected_value is None:
                continue
            actual_value = getattr(meta, field_name)
            if actual_value != expected_value:
                return ReuseDecision(action="compute", reason="hash_mismatch")
        return ReuseDecision(action="skip", reason="hashes_valid")
