"""Модели файла вопросов."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Question(BaseModel):
    """Один вопрос из входного JSON.

    Attributes:
        id: Уникальный идентификатор вопроса.
        category: Каноническая формулировка для LLM.
        synonyms: Варианты запроса только для поиска.
        expected_answerable: Ожидается ли ответ в документе.
    """

    id: str
    category: str
    synonyms: list[str] = Field(default_factory=list)
    expected_answerable: bool = True


class QuestionSet(BaseModel):
    """Набор вопросов с инвариантами уникальности.

    Attributes:
        version: Версия файла вопросов.
        questions: Список вопросов.
    """

    version: str = "1.0"
    questions: list[Question]

    @model_validator(mode="after")
    def validate_uniqueness(self) -> QuestionSet:
        """Проверяет непустой список и уникальность id и category.

        Returns:
            Тот же экземпляр при успехе.

        Raises:
            ValueError: Пустой список или дубликаты id/category.
        """
        if not self.questions:
            raise ValueError("Список вопросов пуст")
        question_ids = [item.id for item in self.questions]
        categories = [item.category for item in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Дубликат id в файле вопросов")
        if len(categories) != len(set(categories)):
            raise ValueError("Дубликат category в файле вопросов")
        return self
