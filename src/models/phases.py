"""Модели планирования и отчётов фаз."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from models.config import AppConfig
from models.documents import DocumentRef
from models.questions import QuestionSet
from models.requests import PhaseName


class ReuseDecision(BaseModel):
    """Решение: пересчитать артефакт или взять готовый.

    Attributes:
        action: compute или skip.
        reason: Причина решения.
    """

    action: Literal["compute", "skip"]
    reason: str


class PhasePlanItem(BaseModel):
    """Элемент плана фазы для одного документа.

    Attributes:
        document: Ссылка на документ.
        decision: Решение reuse.
    """

    document: DocumentRef
    decision: ReuseDecision


class PhasePlan(BaseModel):
    """План исполнения фазы.

    Attributes:
        phase: Имя фазы.
        items: Решения по документам.
    """

    phase: PhaseName
    items: list[PhasePlanItem] = Field(default_factory=list)

    @property
    def compute_count(self) -> int:
        """Число документов к пересчёту.

        Returns:
            Количество action=compute.
        """
        return sum(1 for item in self.items if item.decision.action == "compute")

    @property
    def skip_count(self) -> int:
        """Число документов к пропуску.

        Returns:
            Количество action=skip.
        """
        return sum(1 for item in self.items if item.decision.action == "skip")


class DocumentPhaseOutcome(BaseModel):
    """Исход фазы для одного документа.

    Attributes:
        doc_id: Идентификатор документа.
        status: computed, skipped или failed.
        reason: Причина failure/skip, если есть.
    """

    doc_id: str
    status: Literal["computed", "skipped", "failed"]
    reason: str | None = None


class PhaseReport(BaseModel):
    """Отчёт исполнения фазы.

    Attributes:
        phase: Имя фазы.
        computed: Число пересчитанных документов.
        skipped: Число пропущенных.
        failed: Число упавших.
        outcomes: Детали по документам.
    """

    phase: PhaseName
    computed: int = 0
    skipped: int = 0
    failed: int = 0
    outcomes: list[DocumentPhaseOutcome] = Field(default_factory=list)


class PhaseContext(BaseModel):
    """Контекст планирования и исполнения фазы.

    Attributes:
        config: Резолвнутый AppConfig.
        documents: Документы прогона.
        force: Принудительный пересчёт этой фазы.
        questions: Загруженные вопросы, если применимо.
        requested_phases: Все фазы текущего прогона.
    """

    config: AppConfig
    documents: list[DocumentRef] = Field(default_factory=list)
    force: bool = False
    questions: QuestionSet | None = None
    requested_phases: list[PhaseName] = Field(default_factory=list)
