"""Отладочные файлы пары документ × вопрос."""

from __future__ import annotations

import json
from pathlib import Path

from answer.exchange import ModelAttempt
from models.artifacts import AnswerRecord


class AnswerDebugDump:
    """Пишет prompt.txt, attempts.jsonl и pair.json в cache/debug.

    Attributes:
        enabled: При false методы ничего не создают.
        cache_dir: Корень, внутри которого лежит каталог debug.
    """

    def __init__(self, enabled: bool = False, cache_dir: str = "./.cache") -> None:
        """Сохраняет флаг и каталог.

        Args:
            enabled: Писать ли файлы.
            cache_dir: runtime.cache_dir.
        """
        self.enabled = enabled
        self.cache_dir = cache_dir

    def write_prompt(
        self,
        doc_id: str,
        question_id: str,
        prompt: str | None,
    ) -> str | None:
        """Пишет точный текст промпта.

        Args:
            doc_id: Документ.
            question_id: Вопрос.
            prompt: Текст промпта или None, если модель не вызывалась.

        Returns:
            Путь к файлу или None, если дамп выключен либо промпта нет.
        """
        if not self.enabled or prompt is None:
            return None
        path = self._path(doc_id, question_id, "prompt.txt")
        path.write_text(prompt, encoding="utf-8")
        return str(path)

    def write_attempts(
        self,
        doc_id: str,
        question_id: str,
        attempts: list[ModelAttempt],
    ) -> str | None:
        """Пишет по строке JSON на каждую попытку.

        Args:
            doc_id: Документ.
            question_id: Вопрос.
            attempts: Попытки в порядке вызовов.

        Returns:
            Путь к файлу или None, если дамп выключен либо попыток нет.
        """
        if not self.enabled or not attempts:
            return None
        lines = [
            json.dumps(
                {
                    "attempt": attempt.attempt,
                    "outcome": attempt.outcome,
                    "text": attempt.text,
                    "error": attempt.error,
                },
                ensure_ascii=False,
            )
            for attempt in attempts
        ]
        path = self._path(doc_id, question_id, "attempts.jsonl")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def write_pair(self, record: AnswerRecord) -> str | None:
        """Пишет итог пары без промпта и сырого текста модели.

        Args:
            record: Запись ответа.

        Returns:
            Путь к файлу или None, если дамп выключен.
        """
        if not self.enabled:
            return None
        payload = {
            "doc_id": record.doc_id,
            "question_id": record.question_id,
            "status": record.status,
            "reason": record.reason,
            "found": record.found,
            "answer": record.answer,
        }
        path = self._path(record.doc_id, record.question_id, "pair.json")
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        path.write_text(text + "\n", encoding="utf-8")
        return str(path)

    def _path(self, doc_id: str, question_id: str, name: str) -> Path:
        """Создаёт каталог пары и возвращает путь файла.

        Args:
            doc_id: Документ.
            question_id: Вопрос.
            name: Имя файла.

        Returns:
            Путь внутри cache/debug.
        """
        directory = Path(self.cache_dir) / "debug" / doc_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{question_id}.{name}"
