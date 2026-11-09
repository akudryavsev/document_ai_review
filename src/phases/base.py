"""Абстрактная фаза plan/execute."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.observe import TraceEvent
from models.phases import DocumentPhaseOutcome, PhaseContext, PhasePlan, PhaseReport


class Phase(ABC):
    """Базовый контракт фазы пайплайна.

    Attributes:
        trace_stage: Имя фазы в TraceEvent отказа документа.
    """

    trace_stage = ""

    def fail_document(self, doc_id: str, reason: str) -> DocumentPhaseOutcome:
        """Пишет отказ документа в трассу и возвращает исход.

        Исключение не бросает: фаза продолжает следующие документы.
        Строку лога собирает TraceSink, не эта фаза.

        Args:
            doc_id: Идентификатор документа.
            reason: Причина. Попадает в TraceEvent.error.

        Returns:
            DocumentPhaseOutcome со status=failed.
        """
        self.trace_sink.emit(
            TraceEvent(
                stage=self.trace_stage,
                status="failed",
                doc_id=doc_id,
                error=reason,
            )
        )
        return DocumentPhaseOutcome(
            doc_id=doc_id,
            status="failed",
            reason=reason,
        )

    @abstractmethod
    def plan(self, context: PhaseContext) -> PhasePlan:
        """Строит план compute/skip по документам.

        Args:
            context: Контекст фазы.

        Returns:
            PhasePlan.
        """

    @abstractmethod
    def execute(self, plan: PhasePlan, context: PhaseContext) -> PhaseReport:
        """Исполняет план и возвращает отчёт.

        Args:
            plan: PhasePlan.
            context: Контекст фазы.

        Returns:
            PhaseReport.
        """
