"""Файловое хранилище артефактов JSONL + float16."""

from __future__ import annotations

from pathlib import Path

from ports.trace import TraceSink
from store.atomic import AtomicFileBatch
from store.base import CheckingArtifactStore


class FileArtifactStore(CheckingArtifactStore):
    """Пишет артефакты A1/A2/B на диск без pyarrow.

    Attributes:
        cache_dir: Корень emb, questions, embed_cache и answers.
        index_dir: Корень candidates.
        contexts_dir: Корень context-бандлов.
        trace_sink: Приёмник событий, если передан в базу.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        index_dir: str | Path,
        contexts_dir: str | Path,
        trace_sink: TraceSink | None = None,
    ) -> None:
        """Запоминает корни каталогов, не создавая их заранее.

        Args:
            cache_dir: Каталог `.cache`.
            index_dir: Каталог index.
            contexts_dir: Каталог contexts.
            trace_sink: TraceSink или None.
        """
        super().__init__(trace_sink)
        self.cache_dir = Path(cache_dir)
        self.index_dir = Path(index_dir)
        self.contexts_dir = Path(contexts_dir)
        self._batch = AtomicFileBatch()

    def _read(self, key: str) -> bytes | None:
        """Читает файл ключа.

        Args:
            key: Логический ключ.

        Returns:
            Байты или None, если файла нет.
        """
        path = self._path_for(key)
        if not path.is_file():
            return None
        return path.read_bytes()

    def _write_many(self, items: dict[str, bytes]) -> None:
        """Атомарно заменяет файлы набора.

        Args:
            items: Логический ключ → байты.
        """
        files = {self._path_for(key): payload for key, payload in items.items()}
        self._batch.commit(files)

    def _path_for(self, key: str) -> Path:
        """Переводит логический ключ в путь внутри одного из корней.

        Args:
            key: Ключ с префиксом cache/, index/ или contexts/.

        Returns:
            Абсолютный путь внутри корня.

        Raises:
            ValueError: Ключ неизвестен или выходит за корень.
        """
        if key.startswith("cache/"):
            root = self.cache_dir
            relative = key[len("cache/") :]
        elif key.startswith("index/"):
            root = self.index_dir
            relative = key[len("index/") :]
        elif key.startswith("contexts/"):
            root = self.contexts_dir
            relative = key[len("contexts/") :]
        else:
            raise ValueError(f"неизвестный ключ артефакта: {key}")
        root_resolved = root.resolve()
        path = (root / relative).resolve()
        if path != root_resolved and root_resolved not in path.parents:
            raise ValueError("путь артефакта выходит за корень")
        return path
