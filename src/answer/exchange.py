"""Промпт и попытки последнего вызова модели. В jsonl не попадают."""

from __future__ import annotations


class ModelAttempt:
    """Одна попытка ответа модели.

    Attributes:
        attempt: Номер попытки с 1.
        outcome: exception, empty, invalid_json, invalid_schema,
            citation_range или ok.
        text: Сырой текст ответа. При исключении до ответа — пустая строка.
        error: Короткий текст ошибки или None.
    """

    def __init__(
        self,
        attempt: int,
        outcome: str,
        text: str,
        error: str | None,
    ) -> None:
        """Сохраняет поля попытки.

        Args:
            attempt: Номер с 1.
            outcome: Код исхода.
            text: Сырой текст или пустая строка.
            error: Текст ошибки или None.
        """
        self.attempt = attempt
        self.outcome = outcome
        self.text = text
        self.error = error


class AnswerExchange:
    """Промпт и попытки последнего generate.

    Attributes:
        prompt: Текст, ушедший в шлюз, или None, если модель не вызывалась.
        attempts: Попытки в порядке вызовов.
    """

    def __init__(
        self,
        prompt: str | None = None,
        attempts: list[ModelAttempt] | None = None,
    ) -> None:
        """Сохраняет обмен с моделью.

        Args:
            prompt: Промпт или None.
            attempts: Список попыток. Пустой список, если вызовов не было.
        """
        self.prompt = prompt
        self.attempts = [] if attempts is None else attempts
