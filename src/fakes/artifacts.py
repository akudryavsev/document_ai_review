"""In-memory ArtifactStore для тестов.

Реализация живёт в store.memory_store и подчиняется тем же правилам, что файл.
"""

from store.memory_store import InMemoryArtifactStore

__all__ = ["InMemoryArtifactStore"]
