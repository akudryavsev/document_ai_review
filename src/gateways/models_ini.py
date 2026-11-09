"""Проверка списков GGUF и канонический models.ini."""

from __future__ import annotations

from pathlib import Path

from models.config import AppConfig


class ModelListError(Exception):
    """Список имён или файл модели не прошёл проверку до фаз."""


class ModelIniError(Exception):
    """Файл models.ini не совпадает со списками конфига."""


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
            config: Конфиг после подмены флагов CLI.

        Raises:
            ModelListError: Пустой список, плохое имя, дубликат или нет файла.
        """
        roles = (
            ("embedding", list(config.embedding.models or [])),
            ("reranker", list(config.reranker.models or [])),
            ("answer", list(config.answer.models or [])),
        )
        seen_global: dict[str, str] = {}
        for role, names in roles:
            if not names:
                raise ModelListError(f"список {role} пуст")
            seen_role: set[str] = set()
            for name in names:
                self._check_name(name)
                if name in seen_role:
                    raise ModelListError(f"имя {name} повторяется в списке {role}")
                seen_role.add(name)
                previous = seen_global.get(name)
                if previous is not None:
                    raise ModelListError(
                        f"имя {name} стоит в ролях {previous} и {role}"
                    )
                seen_global[name] = role
                if not (self.models_dir / name).is_file():
                    raise ModelListError(f"нет файла models/{name}")

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
    """Собирает текст deploy/llama/models.ini из списков конфига."""

    def render(self, config: AppConfig) -> str:
        """Строит канонический текст пресета.

        Args:
            config: Конфиг со списками моделей.

        Returns:
            Текст ini с завершающим переводом строки.
        """
        blocks = ["[*]\nctx-size = 4096"]
        for name in config.embedding.models or []:
            blocks.append(
                f"[{name}]\nmodel = /models/{name}\nembedding = true"
            )
        for name in config.reranker.models or []:
            blocks.append(
                "\n".join(
                    [
                        f"[{name}]",
                        f"model = /models/{name}",
                        "embedding = true",
                        "reranking = true",
                        "pooling = rank",
                    ]
                )
            )
        for name in config.answer.models or []:
            blocks.append(f"[{name}]\nmodel = /models/{name}")
        return "\n\n".join(blocks) + "\n"

    def write(self, config: AppConfig, path: str | Path) -> None:
        """Перезаписывает файл каноническим текстом.

        Args:
            config: Конфиг со списками.
            path: Путь к models.ini.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(config), encoding="utf-8")

    def ensure_matches(self, config: AppConfig, path: str | Path) -> None:
        """Останавливает прогон, если файл не равен каноническому тексту.

        Args:
            config: Конфиг со списками.
            path: Путь к models.ini.

        Raises:
            ModelIniError: Файла нет или текст другой.
        """
        target = Path(path)
        actual = target.read_text(encoding="utf-8") if target.is_file() else ""
        if actual != self.render(config):
            raise ModelIniError(
                "models.ini не совпадает со списками конфига. "
                "Перезапишите его командой write-models-ini "
                "и пересоздайте контейнер inference, чтобы он перечитал пресет."
            )
