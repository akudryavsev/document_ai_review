"""Сведение ключей model и models к одному списку имён файлов."""

from __future__ import annotations


class ModelListResolver:
    """Превращает строку model или список models в пару (текущее имя, список).

    Attributes:
        default_name: Имя, если в секции не задан ни один ключ.
    """

    def __init__(self, default_name: str) -> None:
        """Сохраняет имя по умолчанию для роли.

        Args:
            default_name: Имя, если оба ключа пусты.
        """
        self.default_name = default_name

    def resolve(
        self,
        model: str | None,
        models: list[str] | None,
    ) -> tuple[str, list[str]]:
        """Собирает список имён роли.

        Args:
            model: Одиночное имя или None.
            models: Список имён или None.

        Returns:
            Текущее имя (первый элемент) и полный список в порядке конфига.

        Raises:
            ValueError: Заданы оба ключа или список пуст.
        """
        if model is not None and models is not None:
            raise ValueError("одновременно заданы model и models")
        if models is not None:
            if not models:
                raise ValueError("список моделей пуст")
            return models[0], list(models)
        if model is None:
            return self.default_name, [self.default_name]
        return model, [model]
