"""Записи GGUF: имя файла, параметры секции и сведение model/models."""

from __future__ import annotations

import math
import re

from pydantic import BaseModel, Field


class ModelListError(Exception):
    """Список имён или файл модели не прошёл проверку до фаз."""


class ModelPreset(BaseModel):
    """Одна GGUF и скаляры её секции в models.ini.

    Attributes:
        name: Имя файла в каталоге models/.
        params: Ключи секции в порядке записи в YAML.
    """

    name: str
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)


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


class ModelPresetNormalizer:
    """Переписывает секции ролей в YAML: объекты становятся именами и entries."""

    _roles = ("embedding", "reranker", "answer")

    def apply(self, payload: dict) -> None:
        """Нормализует три секции на месте.

        Строка вместо объекта — ошибка файла конфига. Уже разобранный
        список имён (внутренний JSON задания) сюда не приходит.

        Args:
            payload: Корень YAML.

        Raises:
            ValueError: Запись модели не объект с name либо заданы оба ключа.
        """
        for role in self._roles:
            section = payload.get(role)
            if not isinstance(section, dict):
                continue
            self._normalize_section(section)

    def _normalize_section(self, section: dict) -> None:
        """Сводит model и models секции к списку имён и entries.

        Args:
            section: Словарь роли в YAML.

        Raises:
            ValueError: Форма записи не проходит правила спеки.
        """
        raw_model = section.get("model")
        raw_models = section.get("models")
        if raw_model is not None and raw_models is not None:
            raise ValueError("одновременно заданы model и models")
        if raw_models is None and raw_model is None:
            return
        if raw_models is not None:
            if not isinstance(raw_models, list) or not raw_models:
                raise ValueError("список моделей пуст")
            source = raw_models
        else:
            source = [raw_model]
        entries = [self._entry(item) for item in source]
        section["models"] = [entry["name"] for entry in entries]
        section["entries"] = entries
        section.pop("model", None)

    def _entry(self, item: object) -> dict:
        """Проверяет одну запись и возвращает name и params.

        Args:
            item: Элемент списка или значение ключа model.

        Returns:
            Словарь с ключами name и params.

        Raises:
            ValueError: Элемент не объект с именем или params не объект.
        """
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("элемент списка моделей — объект с name")
        name = item["name"]
        if not isinstance(name, str):
            raise ValueError("элемент списка моделей — объект с name")
        params = item.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("params — объект")
        return {"name": name, "params": params}


class ModelPresetRules:
    """Проверяет скаляры params для роли и готовит список пресетов."""

    _key_pattern = re.compile(r"^[a-z][a-z0-9-]*$")
    _forbidden = {
        "embedding": {"model", "embedding", "reranking"},
        "reranker": {"model", "embedding", "reranking"},
        "answer": {"model", "embedding", "reranking", "pooling"},
    }
    _pooling = {
        "embedding": {"mean", "cls", "last"},
        "reranker": {"rank"},
    }

    def read(self, role_config: object) -> list[ModelPreset]:
        """Возвращает пресеты роли: entries, иначе имена без параметров.

        Args:
            role_config: Секция конфига с полями entries и models.

        Returns:
            Список ModelPreset в порядке конфига.
        """
        entries = getattr(role_config, "entries", None) or []
        if entries:
            return list(entries)
        names = getattr(role_config, "models", None) or []
        return [ModelPreset(name=name) for name in names]

    def check(self, role: str, preset: ModelPreset) -> None:
        """Проверяет ключи и обязательный pooling роли.

        Args:
            role: embedding, reranker или answer.
            preset: Запись модели.

        Raises:
            ModelListError: Ключ, значение или pooling не проходят правила.
        """
        forbidden = self._forbidden[role]
        for key, value in preset.params.items():
            if not self._key_pattern.fullmatch(key):
                raise ModelListError(f"недопустимый ключ параметра {key}")
            if key in forbidden:
                raise ModelListError(f"параметр {key} нельзя задавать в params")
            self._check_scalar(key, value)
        allowed = self._pooling.get(role)
        if allowed is None:
            return
        pooling = preset.params.get("pooling")
        if pooling is None:
            raise ModelListError(f"у модели {preset.name} нет параметра pooling")
        if pooling not in allowed:
            expected = ", ".join(sorted(allowed))
            raise ModelListError(
                f"pooling модели {preset.name} должен быть одним из: {expected}"
            )

    def _check_scalar(self, key: str, value: object) -> None:
        """Отсекает нескаляр и пустую строку.

        Args:
            key: Имя параметра.
            value: Значение из YAML.

        Raises:
            ModelListError: Значение нельзя записать одной строкой ini.
        """
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ModelListError(f"значение параметра {key} должно быть конечным числом")
            return
        if isinstance(value, str):
            if value == "" or "\n" in value or "\r" in value:
                raise ModelListError(f"значение параметра {key} пустое или многострочное")
            return
        raise ModelListError(f"значение параметра {key} должно быть скаляром")
