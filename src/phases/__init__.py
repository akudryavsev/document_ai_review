"""Фазы индексации, поиска и ответа."""

from phases.answering import AnsweringPhase
from phases.base import Phase
from phases.indexing import IndexingPhase
from phases.retrieval import RetrievalPhase

__all__ = [
    "AnsweringPhase",
    "IndexingPhase",
    "Phase",
    "RetrievalPhase",
]
