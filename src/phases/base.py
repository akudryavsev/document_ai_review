"""Абстрактная фаза plan/execute."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.phases import PhaseContext, PhasePlan, PhaseReport


class Phase(ABC):
    """Базовый контракт фазы пайплайна."""

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
