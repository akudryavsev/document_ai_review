"""Проверка списков GGUF и канонический models.ini."""

from __future__ import annotations

from pathlib import Path

from models.config import AppConfig
from models.model_list import ModelListError, ModelPreset, ModelPresetRules


class ModelIniError(Exception):
    """models.ini переписан; контейнер inference должен перечитать пресет."""


class ModelCatalog:
    """Проверяет имена файлов трёх ролей до старта фаз.

    Attributes:
        models_dir: Каталог models/ на хосте или в контейнере app.
    """

    def __init__(self, models_dir: str | Path) -> None:
        """Сохраняет каталог весов.

        Args:
            models_dir: Каталог с GGUF.
        """
        self.models_dir = Path(models_dir)

    def check(self, config: AppConfig) -> None:
        """Проверяет три списка и наличие файлов.

        Args:
            config: Полный конфиг файла, до сужения списков флагами CLI.

        Raises:
            ModelListError: Пустой список, параметры, дубликат или нет файла.
        """
        rules = ModelPresetRules()
        roles = (
            ("embedding", config.embedding),
            ("reranker", config.reranker),
            ("answer", config.answer),
        )
        seen_global: dict[str, str] = {}
        for role, section in roles:
            presets = rules.read(section)
            if not presets:
                raise ModelListError(f"список {role} пуст")
            seen_role: set[str] = set()
            for preset in presets:
                self._check_name(preset.name)
                rules.check(role, preset)
                if preset.name in seen_role:
                    raise ModelListError(f"имя {preset.name} повторяется в списке {role}")
                seen_role.add(preset.name)
                previous = seen_global.get(preset.name)
                if previous is not None:
                    raise ModelListError(
                        f"имя {preset.name} стоит в ролях {previous} и {role}"
                    )
                seen_global[preset.name] = role
                if not (self.models_dir / preset.name).is_file():
                    raise ModelListError(f"нет файла models/{preset.name}")

    def _check_name(self, name: str) -> None:
        """Проверяет, что имя — один файл .gguf.

        Args:
            name: Имя из списка.

        Raises:
            ModelListError: Имя не проходит правила спеки.
        """
        if (
            name in {".", ".."}
            or "/" in name
            or "\\" in name
            or not name.endswith(".gguf")
        ):
            raise ModelListError(f"недопустимое имя модели: {name}")


class LlamaModelsIni:
    """Собирает текст deploy/llama/models.ini из списков конфига.

    Для команды run при расхождении пишет файл и останавливает прогон.
    """

    def render(self, config: AppConfig) -> str:
        """Строит канонический текст пресета.

        Args:
            config: Конфиг со списками моделей.

        Returns:
            Текст ini с завершающим переводом строки.
        """
        rules = ModelPresetRules()
        blocks = ["[*]\nctx-size = 4096"]
        blocks.extend(
            self._sections(
                rules.read(config.embedding),
                ["embedding = true"],
            )
        )
        blocks.extend(
            self._sections(
                rules.read(config.reranker),
                ["embedding = true", "reranking = true"],
            )
        )
        blocks.extend(self._sections(rules.read(config.answer), []))
        return "\n\n".join(blocks) + "\n"

    def _sections(self, presets: list[ModelPreset], role_flags: list[str]) -> list[str]:
        """Собирает секции роли: путь, флаги роли, затем params.

        Args:
            presets: Записи роли в порядке конфига.
            role_flags: Строки, которые роль пишет сама.

        Returns:
            Тексты секций без разделительной пустой строки.
        """
        blocks: list[str] = []
        for preset in presets:
            lines = [f"[{preset.name}]", f"model = /models/{preset.name}"]
            lines.extend(role_flags)
            for key, value in preset.params.items():
                lines.append(f"{key} = {self._format_value(value)}")
            blocks.append("\n".join(lines))
        return blocks

    def _format_value(self, value: object) -> str:
        """Пишет скаляр params одной строкой ini.

        Args:
            value: Строка, целое, число или логическое.

        Returns:
            Текст справа от знака равенства.
        """
        if isinstance(value, bool):
            if value:
                return "true"
            return "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            text = format(value, "f").rstrip("0").rstrip(".")
            if "." not in text:
                text = f"{text}.0"
            return text
        return str(value)

    def write(self, config: AppConfig, path: str | Path) -> None:
        """Перезаписывает файл каноническим текстом.

        Args:
            config: Конфиг со списками.
            path: Путь к models.ini.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(config), encoding="utf-8")

    def prepare_for_run(self, config: AppConfig, path: str | Path) -> None:
        """Пишет канонический текст при расхождении и тогда останавливает прогон.

        Args:
            config: Конфиг со списками.
            path: Путь к models.ini.

        Raises:
            ModelIniError: Текст файла отличался; после записи контейнер
                inference должен перечитать пресет.
        """
        target = Path(path)
        actual = target.read_text(encoding="utf-8") if target.is_file() else ""
        canonical = self.render(config)
        if actual == canonical:
            return
        self.write(config, path)
        raise ModelIniError(
            "models.ini обновлён по спискам конфига.\n"
            "Пересоздайте контейнер inference, чтобы он перечитал пресет, "
            "затем повторите run:\n"
            "\n"
            "  docker compose --profile poc up -d --force-recreate inference"
        )
