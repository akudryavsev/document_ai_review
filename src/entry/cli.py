"""Точка входа CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from application import Application
from gateways.models_ini import LlamaModelsIni, ModelCatalog, ModelIniError, ModelListError
from models.config import AppConfig
from models.requests import EvalRequest, PhaseName, RunOverrides, RunRequest
from runs.evaluation import EvaluationError


class CliApp:
    """Разбор argv → RunRequest/EvalRequest → Application."""

    def main(self, argv: list[str]) -> int:
        """Точка входа процесса.

        Args:
            argv: Аргументы без имени интерпретатора.

        Returns:
            Код выхода процесса.
        """
        parser = self._build_parser()
        try:
            namespace = parser.parse_args(argv)
        except SystemExit as exit_error:
            code = exit_error.code
            return int(code) if isinstance(code, int) else 1
        if namespace.command is None:
            parser.print_help()
            return 2
        if namespace.command == "run":
            return self._run(namespace)
        if namespace.command == "eval":
            return self._eval(namespace)
        if namespace.command == "write-models-ini":
            return self._write_models_ini(namespace)
        parser.print_help()
        return 2

    def _build_parser(self) -> argparse.ArgumentParser:
        """Собирает argparse-парсер команд.

        Returns:
            ArgumentParser.
        """
        parser = argparse.ArgumentParser(prog="main.py")
        subparsers = parser.add_subparsers(dest="command")

        run_parser = subparsers.add_parser("run", help="Пакетный прогон фаз")
        run_parser.add_argument("--config", required=True)
        run_parser.add_argument("--phase", default="A1,A2,B")
        run_parser.add_argument(
            "--force-phase",
            action="append",
            default=[],
            dest="force_phases",
        )
        run_parser.add_argument("--answer-model", default=None)
        run_parser.add_argument("--reranker-model", default=None)
        run_parser.add_argument("--embedding-model", default=None)
        run_parser.add_argument("--cache-dir", default=None)
        run_parser.add_argument("--contexts-dir", default=None)
        run_parser.add_argument("--results-dir", default=None)
        run_parser.add_argument("--index-dir", default=None)

        eval_parser = subparsers.add_parser("eval", help="Оценка на gt.json")
        eval_parser.add_argument("--config", required=True)
        eval_parser.add_argument("--ground-truth", required=True)
        eval_parser.add_argument("--predictions", required=True)

        ini_parser = subparsers.add_parser(
            "write-models-ini",
            help="Перезаписать deploy/llama/models.ini по спискам конфига",
        )
        ini_parser.add_argument("--config", required=True)
        return parser

    def _run(self, namespace: argparse.Namespace) -> int:
        """Исполняет команду run.

        Args:
            namespace: Разобранные аргументы.

        Returns:
            Код выхода.
        """
        try:
            config = AppConfig.from_yaml(namespace.config)
        except ValidationError as error:
            print(str(error), file=sys.stderr)
            return 1
        phases = self._parse_phases(namespace.phase)
        force_phases = {
            self._parse_single_phase(item) for item in namespace.force_phases
        }
        request = RunRequest(
            config=config,
            config_path=namespace.config,
            phases=phases,
            force_phases=force_phases,
            overrides=RunOverrides(
                answer_model=namespace.answer_model,
                reranker_model=namespace.reranker_model,
                embedding_model=namespace.embedding_model,
                cache_dir=namespace.cache_dir,
                contexts_dir=namespace.contexts_dir,
                results_dir=namespace.results_dir,
                index_dir=namespace.index_dir,
            ),
        )
        resolved = request.resolved_config()
        if not self._accept_models(resolved, check_ini=True):
            return 1
        service = Application(resolved).build()
        summary = service.run(request)
        print(
            f"run complete documents={summary.documents} "
            f"questions={summary.questions} error_rate={summary.error_rate}"
        )
        return 0

    def _eval(self, namespace: argparse.Namespace) -> int:
        """Исполняет команду eval (пока не реализована).

        Args:
            namespace: Разобранные аргументы.

        Returns:
            Код выхода.
        """
        try:
            config = AppConfig.from_yaml(namespace.config)
        except ValidationError as error:
            print(str(error), file=sys.stderr)
            return 1
        request = EvalRequest(
            config=config,
            config_path=namespace.config,
            ground_truth_path=namespace.ground_truth,
            predictions_dir=namespace.predictions,
        )
        service = Application(config).build()
        try:
            reports = service.evaluate(request)
        except EvaluationError as error:
            print(str(error), file=sys.stderr)
            return 1
        payload = [report.model_dump() for report in reports]
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    def _write_models_ini(self, namespace: argparse.Namespace) -> int:
        """Перезаписывает models.ini каноническим текстом списков.

        Args:
            namespace: Разобранные аргументы.

        Returns:
            Код выхода. При ошибке списка файл не меняется.
        """
        try:
            config = AppConfig.from_yaml(namespace.config)
        except ValidationError as error:
            print(str(error), file=sys.stderr)
            return 1
        if not self._accept_models(config, check_ini=False):
            return 1
        LlamaModelsIni().write(config, Path("deploy/llama/models.ini"))
        return 0

    def _accept_models(self, config: AppConfig, check_ini: bool) -> bool:
        """Проверяет файлы моделей и, для run, совпадение ini.

        Args:
            config: Конфиг после подмены флагов CLI.
            check_ini: Сравнивать deploy/llama/models.ini.

        Returns:
            True, если проверки прошли.
        """
        try:
            ModelCatalog(Path("models")).check(config)
            if check_ini:
                LlamaModelsIni().ensure_matches(
                    config,
                    Path("deploy/llama/models.ini"),
                )
        except (ModelListError, ModelIniError) as error:
            print(str(error), file=sys.stderr)
            return False
        return True

    def _parse_phases(self, raw: str) -> list[PhaseName]:
        """Разбирает список фаз из --phase.

        Args:
            raw: Строка вида A1,A2,B.

        Returns:
            Список PhaseName.
        """
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        return [self._parse_single_phase(part) for part in parts]

    def _parse_single_phase(self, raw: str) -> PhaseName:
        """Проверяет имя одной фазы.

        Args:
            raw: A1, A2 или B.

        Returns:
            PhaseName.

        Raises:
            SystemExit: Неизвестная фаза.
        """
        if raw not in {"A1", "A2", "B"}:
            raise SystemExit(f"Неизвестная фаза: {raw}")
        return raw  # type: ignore[return-value]
