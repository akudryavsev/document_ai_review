"""T-MDL: записи моделей, params и models.ini."""

from __future__ import annotations

from pathlib import Path

import pytest

from entry import CliApp
from gateways.models_ini import LlamaModelsIni
from models.config import AppConfig
from models.model_list import ModelListError
from models.requests import RunRequest
from tests.helpers import ServiceFactory
from tests.support import build_harness

_SECTION_TWO = """
embedding:
  models:
    - name: bge-m3-q8.gguf
      params:
        pooling: cls
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
    - name: qwen-reranker.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
    - name: Llama-3.1-8B-Q4_K_M.gguf
      params:
        ctx-size: 8192
"""

_SECTION_FOUR = """[*]
ctx-size = 4096

[bge-m3-q8.gguf]
model = /models/bge-m3-q8.gguf
embedding = true
pooling = cls

[bge-reranker.gguf]
model = /models/bge-reranker.gguf
embedding = true
reranking = true
pooling = rank

[qwen-reranker.gguf]
model = /models/qwen-reranker.gguf
embedding = true
reranking = true
pooling = rank

[Qwen3-8B-Q4_K_M.gguf]
model = /models/Qwen3-8B-Q4_K_M.gguf

[Llama-3.1-8B-Q4_K_M.gguf]
model = /models/Llama-3.1-8B-Q4_K_M.gguf
ctx-size = 8192
"""


class TestModelLists:
    """Каталог T-MDL-02…07 и T-MDL-11, T-MDL-15…19."""

    def test_t_mdl_02_empty_list_or_both_keys(self, tmp_path: Path) -> None:
        """Пустой models или оба ключа model и models — ошибка до фаз."""
        empty = self._config_path(tmp_path, "embedding:\n  models: []\n")
        both = self._config_path(
            tmp_path,
            "embedding:\n  model:\n    name: a.gguf\n  models:\n    - name: b.gguf\n",
        )
        assert self._command(tmp_path, "write-models-ini", empty) == 1
        assert self._command(tmp_path, "write-models-ini", both) == 1

    def test_t_mdl_03_missing_file_does_not_write_ini(self, tmp_path: Path) -> None:
        """Нет файла GGUF — ошибка до фаз, ini не меняется."""
        config_path = self._config_path(tmp_path, _SECTION_TWO)
        models = tmp_path / "models"
        self._touch(models, "bge-m3-q8.gguf", "bge-reranker.gguf", "qwen-reranker.gguf")
        ini_path = self._sentinel(tmp_path)
        code = self._command(tmp_path, "write-models-ini", config_path, models, ini_path)
        assert code == 1
        assert ini_path.read_text(encoding="utf-8") == "sentinel"

    def test_t_mdl_04_duplicate_names(self, tmp_path: Path) -> None:
        """Повтор в одном списке или одно имя в двух ролях — ошибка до фаз."""
        repeated = """
embedding:
  models:
    - name: bge-m3-q8.gguf
      params:
        pooling: cls
    - name: bge-m3-q8.gguf
      params:
        pooling: mean
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
"""
        shared = """
embedding:
  models:
    - name: same.gguf
      params:
        pooling: cls
reranker:
  models:
    - name: same.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
"""
        ini_path = self._sentinel(tmp_path)
        assert self._command(tmp_path, "write-models-ini", self._config_path(tmp_path, repeated), ini_path=ini_path) == 1
        assert self._command(tmp_path, "write-models-ini", self._config_path(tmp_path, shared), ini_path=ini_path) == 1
        assert ini_path.read_text(encoding="utf-8") == "sentinel"

    def test_t_mdl_05_single_model_object_keeps_params(self, tmp_path: Path) -> None:
        """Ключ model — одна запись, список имён и её params заполнены."""
        text = """
embedding:
  model:
    name: bge-m3-q8.gguf
    params:
      pooling: cls
"""
        config = AppConfig.from_yaml(self._config_path(tmp_path, text))
        assert config.embedding.models == ["bge-m3-q8.gguf"]
        assert config.embedding.model == "bge-m3-q8.gguf"
        assert config.embedding.entries[0].params["pooling"] == "cls"

    def test_t_mdl_06_write_models_ini_matches_section_four(self, tmp_path: Path) -> None:
        """write-models-ini на списках раздела 2 пишет текст раздела 4."""
        config_path = self._config_path(tmp_path, _SECTION_TWO)
        models = tmp_path / "models"
        self._touch(
            models,
            "bge-m3-q8.gguf",
            "bge-reranker.gguf",
            "qwen-reranker.gguf",
            "Qwen3-8B-Q4_K_M.gguf",
            "Llama-3.1-8B-Q4_K_M.gguf",
        )
        ini_path = tmp_path / "models.ini"
        assert self._command(tmp_path, "write-models-ini", config_path, models, ini_path) == 0
        assert ini_path.read_text(encoding="utf-8") == _SECTION_FOUR
        scalars = """
embedding:
  models:
    - name: bge-m3-q8.gguf
      params:
        pooling: mean
        flash-attn: true
        rope-scale: 1.5
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
"""
        rendered = LlamaModelsIni().render(AppConfig.from_yaml(self._config_path(tmp_path, scalars)))
        assert "flash-attn = true" in rendered
        assert "rope-scale = 1.5" in rendered
        repo_root = Path(__file__).resolve().parents[1]
        repo_ini = LlamaModelsIni().render(AppConfig.from_yaml(repo_root / "config.yaml"))
        assert (repo_root / "deploy/llama/models.ini").read_text(encoding="utf-8") == repo_ini

    def test_t_mdl_07_run_writes_ini_and_stops(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """run при другом ini пишет канонический текст и останавливается до фаз."""
        config_path = self._config_path(tmp_path, _SECTION_TWO)
        models = tmp_path / "models"
        self._touch(
            models,
            "bge-m3-q8.gguf",
            "bge-reranker.gguf",
            "qwen-reranker.gguf",
            "Qwen3-8B-Q4_K_M.gguf",
            "Llama-3.1-8B-Q4_K_M.gguf",
        )
        ini_path = tmp_path / "models.ini"
        ini_path.write_text("другой пресет\n", encoding="utf-8")
        code = self._command(tmp_path, "run", config_path, models, ini_path)
        captured = capsys.readouterr()
        assert code == 1
        assert ini_path.read_text(encoding="utf-8") == _SECTION_FOUR
        assert "models.ini обновлён" in captured.err
        assert "docker compose --profile poc up -d --force-recreate inference" in captured.err
        assert "run complete" not in captured.out
        LlamaModelsIni().prepare_for_run(AppConfig.from_yaml(config_path), ini_path)

    def test_t_mdl_11_flag_keeps_full_ini(self, tmp_path: Path) -> None:
        """Флаг оставляет одну модель ответа, канонический ini остаётся полным."""
        config = AppConfig.from_yaml(self._config_path(tmp_path, _SECTION_TWO))
        ini_path = tmp_path / "models.ini"
        LlamaModelsIni().write(config, ini_path)
        before = ini_path.read_text(encoding="utf-8")
        request = RunRequest(config=config)
        request.overrides.answer_model = "Llama-3.1-8B-Q4_K_M.gguf"
        resolved = request.resolved_config()
        assert resolved.answer.models == ["Llama-3.1-8B-Q4_K_M.gguf"]
        assert "Qwen3-8B-Q4_K_M.gguf" in LlamaModelsIni().render(config)
        harness = build_harness()
        harness.config = config
        run_request = harness.run_request(phases=["A1"])
        run_request.overrides.answer_model = "Llama-3.1-8B-Q4_K_M.gguf"
        summary = ServiceFactory().build(harness).run(run_request)
        answer_names = {item.answer_model for item in summary.combinations}
        assert answer_names == {"Llama-3.1-8B-Q4_K_M.gguf"}
        assert ini_path.read_text(encoding="utf-8") == before

    def test_t_mdl_15_two_embedders_keep_one_ini(self, tmp_path: Path) -> None:
        """Два pooling в одном ini, цикл комбинаций файл не меняет."""
        text = """
embedding:
  models:
    - name: frida.gguf
      params:
        pooling: mean
    - name: bge-m3-q8.gguf
      params:
        pooling: cls
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
"""
        config = AppConfig.from_yaml(self._config_path(tmp_path, text))
        rendered = LlamaModelsIni().render(config)
        assert "pooling = mean" in rendered
        assert "pooling = cls" in rendered
        ini_path = tmp_path / "models.ini"
        LlamaModelsIni().write(config, ini_path)
        before = ini_path.read_bytes()
        harness = build_harness()
        harness.config = config
        summary = ServiceFactory().build(harness).run(harness.run_request(phases=["A1"]))
        embedding_names = [item.embedding_model for item in summary.combinations]
        assert embedding_names == ["frida.gguf", "bge-m3-q8.gguf"]
        assert ini_path.read_bytes() == before

    def test_t_mdl_16_bad_pooling_does_not_write_ini(self, tmp_path: Path) -> None:
        """Нет pooling, none у эмбеддера или не rank у реранкера — ini не меняется."""
        cases = (
            """
embedding:
  models:
    - name: bge-m3-q8.gguf
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
""",
            """
embedding:
  models:
    - name: bge-m3-q8.gguf
      params:
        pooling: none
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
""",
            """
embedding:
  models:
    - name: bge-m3-q8.gguf
      params:
        pooling: mean
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: mean
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
""",
        )
        ini_path = self._sentinel(tmp_path)
        for index, text in enumerate(cases):
            config_path = self._config_path(tmp_path, text, f"bad-{index}.yaml")
            assert self._command(tmp_path, "write-models-ini", config_path, ini_path=ini_path) == 1
        assert ini_path.read_text(encoding="utf-8") == "sentinel"

    def test_t_mdl_17_forbidden_and_duplicate_params(self, tmp_path: Path) -> None:
        """Запрещённый ключ, нескаляр и повтор ключа — ошибка до фаз."""
        forbidden_cases = (
            """
embedding:
  models:
    - name: bge-m3-q8.gguf
      params:
        pooling: mean
        embedding: true
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
""",
            """
embedding:
  models:
    - name: bge-m3-q8.gguf
      params:
        pooling: mean
        model: /models/other.gguf
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
""",
            """
embedding:
  models:
    - name: bge-m3-q8.gguf
      params:
        pooling: mean
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
        reranking: true
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
""",
            """
embedding:
  models:
    - name: bge-m3-q8.gguf
      params:
        pooling: mean
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
      params:
        pooling: mean
""",
        )
        nested = """
embedding:
  models:
    - name: bge-m3-q8.gguf
      params:
        pooling: mean
        batch-size:
          - 1
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
"""
        duplicated = """
embedding:
  models:
    - name: bge-m3-q8.gguf
      params:
        pooling: mean
        pooling: cls
reranker:
  models:
    - name: bge-reranker.gguf
      params:
        pooling: rank
answer:
  models:
    - name: Qwen3-8B-Q4_K_M.gguf
"""
        ini_path = self._sentinel(tmp_path)
        texts = (*forbidden_cases, nested, duplicated)
        for index, text in enumerate(texts):
            config_path = self._config_path(tmp_path, text, f"forbid-{index}.yaml")
            code = self._command(tmp_path, "write-models-ini", config_path, ini_path=ini_path)
            assert code == 1
        assert ini_path.read_text(encoding="utf-8") == "sentinel"

    def test_t_mdl_18_string_item_rejected(self, tmp_path: Path) -> None:
        """Элемент списка или ключ model строкой — ошибка до фаз."""
        listed = "embedding:\n  models:\n    - bge-m3-q8.gguf\n"
        single = "embedding:\n  model: bge-m3-q8.gguf\n"
        with pytest.raises(ValueError, match="объект с name"):
            AppConfig.from_yaml(self._config_path(tmp_path, listed))
        with pytest.raises(ValueError, match="объект с name"):
            AppConfig.from_yaml(self._config_path(tmp_path, single, "single.yaml"))

    def test_t_mdl_19_unknown_flag_name(self, tmp_path: Path) -> None:
        """Флаг с именем не из списка конфига — ошибка до фаз."""
        config = AppConfig.from_yaml(self._config_path(tmp_path, _SECTION_TWO))
        request = RunRequest(config=config)
        request.overrides.answer_model = "missing.gguf"
        with pytest.raises(ModelListError, match="missing.gguf"):
            request.resolved_config()

    def _config_path(self, root: Path, text: str, filename: str = "config.yaml") -> Path:
        """Пишет фрагмент YAML в каталог теста.

        Args:
            root: Временный каталог.
            text: Текст конфига.
            filename: Имя файла.

        Returns:
            Путь к записанному файлу.
        """
        path = root / filename
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return path

    def _touch(self, models: Path, *names: str) -> None:
        """Создаёт пустые файлы GGUF для проверки каталога.

        Args:
            models: Каталог models.
            names: Имена файлов.
        """
        models.mkdir(parents=True, exist_ok=True)
        for name in names:
            (models / name).write_bytes(b"gguf")

    def _sentinel(self, root: Path) -> Path:
        """Кладёт ini, который ошибка записи не должна заменить.

        Args:
            root: Временный каталог.

        Returns:
            Путь к файлу со словом sentinel.
        """
        path = root / "models.ini"
        path.write_text("sentinel", encoding="utf-8")
        return path

    def _command(
        self,
        root: Path,
        command: str,
        config_path: Path,
        models: Path | None = None,
        ini_path: Path | None = None,
    ) -> int:
        """Запускает команду CLI с подменёнными путями моделей и ini.

        Args:
            root: Временный каталог, если пути не заданы.
            command: run или write-models-ini.
            config_path: YAML конфига.
            models: Каталог GGUF.
            ini_path: Файл пресета.

        Returns:
            Код выхода.
        """
        app = CliApp()
        app.models_dir = models or (root / "models")
        app.models_ini_path = ini_path or (root / "models.ini")
        argv = [command, "--config", str(config_path)]
        if command == "run":
            argv.extend(["--phase", "A1"])
        return app.main(argv)
