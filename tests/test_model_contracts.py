"""T-MOD: инварианты pydantic-моделей."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.artifacts import ContextBundle, ContextCandidate
from models.summary import PhaseCounters, RunSummary


class TestModelContracts:
    """Каталог T-MOD-01…02."""

    def test_t_mod_01_empty_search_with_candidates_invalid(self) -> None:
        """empty_search=true и непустые candidates — ошибка валидации."""
        with pytest.raises(ValidationError):
            ContextBundle(
                doc_id="doc1.docx",
                question_id="Q001",
                category="Срок",
                empty_search=True,
                reason="no_relevant_chunks",
                candidates=[
                    ContextCandidate(
                        rank_after_rerank=1,
                        rerank_score=0.9,
                        text="chunk",
                    )
                ],
            )

    def test_t_mod_02_run_summary_missing_phase_counters(self) -> None:
        """RunSummary без счётчиков запрошенной фазы — ошибка."""
        with pytest.raises(ValidationError):
            RunSummary(
                documents=1,
                questions=1,
                requested_phases=["A1", "A2"],
                phases={"A1": PhaseCounters(computed=1)},
            )
