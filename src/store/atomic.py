"""Атомарная замена набора файлов."""

from __future__ import annotations

import os
from pathlib import Path


class AtomicFileBatch:
    """Пишет набор файлов во временные имена и затем переименовывает их."""

    def commit(self, files: dict[Path, bytes]) -> None:
        """Заменяет файлы только после успешной записи всех временных копий.

        Args:
            files: Целевой путь → содержимое.

        Raises:
            OSError: Диск не принял запись или переименование.
        """
        written: list[tuple[Path, Path]] = []
        try:
            for path, payload in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.parent / f".{path.name}.tmp"
                temporary.write_bytes(payload)
                written.append((temporary, path))
            for temporary, path in written:
                os.replace(temporary, path)
        except Exception:
            for temporary, _path in written:
                if temporary.exists():
                    temporary.unlink()
            raise
