"""JSON-дамп промежуточных шагов retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RetrievalDebugDump:
    """Пишет `{cache}/debug/{doc_id}/{question_id}.retrieval.json`.

    Attributes:
        enabled: При false методы ничего не создают.
        cache_dir: Корень с каталогом debug.
    """

    def __init__(self, enabled: bool = False, cache_dir: str = "./.cache") -> None:
        """Сохраняет флаг и каталог.

        Args:
            enabled: Писать ли файлы.
            cache_dir: runtime.cache_dir.
        """
        self.enabled = enabled
        self.cache_dir = cache_dir

    def path_for(self, doc_id: str, question_id: str) -> Path:
        """Строит путь файла дампа.

        Args:
            doc_id: Документ.
            question_id: Вопрос.

        Returns:
            Путь к JSON.
        """
        return Path(self.cache_dir) / "debug" / doc_id / f"{question_id}.retrieval.json"

    def write_search(self, payload: dict[str, Any]) -> str | None:
        """Пишет блок поиска; `rerank` ставит null, если ключа нет.

        Args:
            payload: Документ дампа с ключами search.

        Returns:
            Путь или None при выключенном флаге.
        """
        if not self.enabled:
            return None
        document = dict(payload)
        if "rerank" not in document:
            document["rerank"] = None
        return self._write(document)

    def update_rerank(
        self,
        doc_id: str,
        question_id: str,
        rerank: dict[str, Any],
    ) -> str | None:
        """Дописывает блок реранка в существующий файл или создаёт оболочку.

        Args:
            doc_id: Документ.
            question_id: Вопрос.
            rerank: Блок `rerank` из спеки.

        Returns:
            Путь или None при выключенном флаге.
        """
        if not self.enabled:
            return None
        path = self.path_for(doc_id, question_id)
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
        else:
            document = {
                "doc_id": doc_id,
                "question_id": question_id,
            }
        document["rerank"] = rerank
        return self._write(document)

    def _write(self, document: dict[str, Any]) -> str:
        """Сохраняет JSON с отступом 2.

        Args:
            document: Полное тело файла.

        Returns:
            Строковый путь.
        """
        path = self.path_for(str(document["doc_id"]), str(document["question_id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(document, ensure_ascii=False, indent=2)
        path.write_text(text + "\n", encoding="utf-8")
        return str(path)
