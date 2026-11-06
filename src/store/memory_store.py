"""Хранилище артефактов в словаре для тестов."""

from __future__ import annotations

from ports.trace import TraceSink
from store.base import CheckingArtifactStore


class InMemoryArtifactStore(CheckingArtifactStore):
    """Держит те же байты, что и файл, но в словаре.

    Attributes:
        blobs: Логический ключ → содержимое.
    """

    def __init__(self, trace_sink: TraceSink | None = None) -> None:
        """Создаёт пустой словарь.

        Args:
            trace_sink: TraceSink или None.
        """
        super().__init__(trace_sink)
        self.blobs: dict[str, bytes] = {}

    def _read(self, key: str) -> bytes | None:
        """Возвращает байты ключа.

        Args:
            key: Логический ключ.

        Returns:
            Байты или None.
        """
        return self.blobs.get(key)

    def _write_many(self, items: dict[str, bytes]) -> None:
        """Заменяет ключи в словаре.

        Args:
            items: Ключ → байты.
        """
        for key, payload in items.items():
            self.blobs[key] = payload
