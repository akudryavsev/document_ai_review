"""T-MDL: списки моделей, ini и группировка eval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from application import Application
from application.result_sink import FileResultSink
from gateways.models_ini import LlamaModelsIni, ModelCatalog, ModelListError
from models.artifacts import AnswerRecord
from models.config import AppConfig
from models.requests import EvalRequest, RunOverrides
from models.summary import EvalReport
from runs.evaluation import EvaluationError, EvaluationRun
from tests.helpers import ServiceFactory
from tests.support import build_harness
from entry.cli import CliApp


_CANONICAL_INI = """[*]
ctx-size = 4096

[bge-m3-q8.gguf]
model = /models/bge-m3-q8.gguf
embedding = true

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
"""


class TestModelLists:
    """Каталог T-MDL-01…14."""

    def test_t_mdl_01_two_answer_models_one_indexing(self) -> None:
        """Две модели ответа дают одну A1, одну A2 и две строки на вопрос."""
        harness = build_harness()
        self._set_lists(harness.config, answers=["a.gguf", "b.gguf"])
        service = ServiceFactory().build(harness)
        summary = service.run(harness.run_request())
        assert len(harness.extractor.calls) == len(harness.documents)
        assert harness.embedding.load_calls == 1
        assert harness.reranker.load_calls == 1
        assert harness.llm.load_calls == 2
        assert summary.phases["A1"].computed == len(harness.documents)
        assert summary.phases["B"].computed == len(harness.documents) * 2
        q001 = [
            answer
            for answer in harness.result_sink.answers
            if answer.doc_id == "doc_a.docx" and answer.question_id == "Q001"
        ]
        assert {answer.answer_model for answer in q001} == {"a.gguf", "b.gguf"}

    def test_t_mdl_02_empty_list_and_both_keys_fail_before_phases(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Пустой список и оба ключа сразу — ошибка до фаз."""
        monkeypatch.chdir(tmp_path)
        self._forbid_build(monkeypatch)
        empty_path = tmp_path / "empty.yaml"
        empty_path.write_text("embedding:\n  models: []\n", encoding="utf-8")
        both_path = tmp_path / "both.yaml"
        both_path.write_text(
            "embedding:\n  model: a.gguf\n  models:\n    - a.gguf\n",
            encoding="utf-8",
        )
        assert CliApp().main(["run", "--config", str(empty_path)]) == 1
        assert CliApp().main(["run", "--config", str(both_path)]) == 1
        with pytest.raises(ValidationError, match="список моделей пуст"):
            AppConfig.model_validate({"embedding": {"models": []}})
        with pytest.raises(ValidationError, match="одновременно заданы model и models"):
            AppConfig.model_validate(
                {"embedding": {"model": "a.gguf", "models": ["a.gguf"]}}
            )

    def test_t_mdl_03_missing_file_does_not_rewrite_ini(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Нет файла модели — ошибка до фаз, ini остаётся прежним."""
        monkeypatch.chdir(tmp_path)
        self._forbid_build(monkeypatch)
        config_path = self._write_yaml(
            tmp_path,
            ["missing.gguf"],
            ["rerank.gguf"],
            ["answer.gguf"],
        )
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "rerank.gguf").write_bytes(b"r")
        (tmp_path / "models" / "answer.gguf").write_bytes(b"a")
        ini_path = tmp_path / "deploy" / "llama" / "models.ini"
        ini_path.parent.mkdir(parents=True)
        ini_path.write_text("старый\n", encoding="utf-8")
        code = CliApp().main(["write-models-ini", "--config", str(config_path)])
        assert code == 1
        assert ini_path.read_text(encoding="utf-8") == "старый\n"

    def test_t_mdl_04_duplicate_names_fail(self, tmp_path: Path) -> None:
        """Повтор в одном списке или одно имя в двух ролях — ошибка."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "same.gguf").write_bytes(b"s")
        repeated = self._config(["same.gguf", "same.gguf"], ["rerank.gguf"], ["a.gguf"])
        shared = self._config(["same.gguf"], ["same.gguf"], ["a.gguf"])
        with pytest.raises(ModelListError):
            ModelCatalog(models_dir).check(repeated)
        with pytest.raises(ModelListError):
            ModelCatalog(models_dir).check(shared)

    def test_t_mdl_05_single_model_key_becomes_list(self) -> None:
        """Ключ model без models даёт список из одного имени."""
        config = AppConfig.model_validate({"embedding": {"model": "file.gguf"}})
        assert config.embedding.model == "file.gguf"
        assert config.embedding.models == ["file.gguf"]

    def test_t_mdl_06_write_models_ini_matches_spec(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """write-models-ini пишет канонический текст списков."""
        monkeypatch.chdir(tmp_path)
        names = [
            "bge-m3-q8.gguf",
            "bge-reranker.gguf",
            "qwen-reranker.gguf",
            "Qwen3-8B-Q4_K_M.gguf",
            "Llama-3.1-8B-Q4_K_M.gguf",
        ]
        for name in names:
            path = tmp_path / "models" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"w")
        config_path = self._write_yaml(
            tmp_path,
            ["bge-m3-q8.gguf"],
            ["bge-reranker.gguf", "qwen-reranker.gguf"],
            ["Qwen3-8B-Q4_K_M.gguf", "Llama-3.1-8B-Q4_K_M.gguf"],
        )
        code = CliApp().main(["write-models-ini", "--config", str(config_path)])
        assert code == 0
        written = (tmp_path / "deploy" / "llama" / "models.ini").read_text(
            encoding="utf-8"
        )
        assert written == _CANONICAL_INI

    def test_t_mdl_07_run_stops_when_ini_differs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """run при чужом ini завершается до фаз."""
        monkeypatch.chdir(tmp_path)
        self._forbid_build(monkeypatch)
        config_path = self._write_yaml(
            tmp_path,
            ["embed.gguf"],
            ["rerank.gguf"],
            ["answer.gguf"],
        )
        for name in ("embed.gguf", "rerank.gguf", "answer.gguf"):
            path = tmp_path / "models" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"w")
        ini_path = tmp_path / "deploy" / "llama" / "models.ini"
        ini_path.parent.mkdir(parents=True)
        ini_path.write_text("другой\n", encoding="utf-8")
        code = CliApp().main(["run", "--config", str(config_path), "--phase", "A1"])
        assert code == 1
        assert "write-models-ini" in capsys.readouterr().err

    def test_t_mdl_08_second_answer_is_computed(self) -> None:
        """Уже посчитанная модель ответа не отменяет B следующей."""
        harness = build_harness()
        self._set_lists(harness.config, answers=["a.gguf"])
        service = ServiceFactory().build(harness)
        service.run(harness.run_request())
        loads_after_first = harness.llm.load_calls
        harness.config.answer.models = ["a.gguf", "b.gguf"]
        summary = service.run(harness.run_request())
        assert harness.llm.load_calls == loads_after_first + 1
        assert summary.phases["B"].computed == len(harness.documents)
        assert summary.phases["B"].skipped == len(harness.documents)
        meta_keys = [
            key
            for key in harness.store.blobs
            if key.startswith("cache/answers/") and key.endswith("/meta.json")
        ]
        assert any("/a.gguf/" in key for key in meta_keys)
        assert any("/b.gguf/" in key for key in meta_keys)

    def test_t_mdl_09_same_triple_replaces_jsonl_line(self, tmp_path: Path) -> None:
        """Повтор той же тройки оставляет одну новую строку."""
        sink = FileResultSink(tmp_path)
        first = self._record("старый")
        second = self._record("новый")
        other = self._record("другая")
        other.answer_model = "other.gguf"
        other.model = "other.gguf"
        sink.write_answer(first)
        sink.write_answer(other)
        sink.write_answer(second)
        lines = (tmp_path / "doc_a.docx.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        kept_other = AnswerRecord.model_validate_json(lines[0])
        kept_new = AnswerRecord.model_validate_json(lines[1])
        assert kept_other.answer == "другая"
        assert kept_new.answer == "новый"

    def test_t_mdl_10_empty_search_fills_triple(self) -> None:
        """Пустой поиск пишет три имени, model совпадает с answer_model."""
        harness = build_harness(empty_pairs={("doc_a.docx", "Q001")})
        self._set_lists(harness.config, answers=["a.gguf"])
        service = ServiceFactory().build(harness)
        service.run(harness.run_request())
        record = next(
            answer
            for answer in harness.result_sink.answers
            if answer.doc_id == "doc_a.docx" and answer.question_id == "Q001"
        )
        assert record.embedding_model == "e.gguf"
        assert record.reranker_model == "r.gguf"
        assert record.answer_model == "a.gguf"
        assert record.model == record.answer_model

    def test_t_mdl_11_answer_flag_replaces_answer_list(self) -> None:
        """Флаг --answer-model оставляет в запуске одну модель ответа."""
        harness = build_harness()
        self._set_lists(harness.config, answers=["a.gguf", "b.gguf"])
        request = harness.run_request()
        request.overrides = RunOverrides(answer_model="other.gguf")
        service = ServiceFactory().build(harness)
        summary = service.run(request)
        assert summary.model_versions.answer == ["other.gguf"]
        assert summary.model_versions.embedding == ["e.gguf"]
        assert summary.model_versions.reranker == ["r.gguf"]
        assert {item.answer_model for item in summary.combinations} == {"other.gguf"}

    def test_t_mdl_12_eval_matches_each_triple(self, tmp_path: Path) -> None:
        """Две тройки и один эталон дают два отчёта с путём кандидатов."""
        index_dir = tmp_path / "index"
        predictions = tmp_path / "predictions"
        predictions.mkdir()
        triples = [
            ("e.gguf", "r.gguf", "a.gguf"),
            ("e.gguf", "r.gguf", "b.gguf"),
        ]
        self._write_predictions(predictions / "doc.docx.jsonl", triples)
        for embedding_name, reranker_name, _answer_name in triples:
            path = (
                index_dir
                / embedding_name
                / reranker_name
                / "doc.docx"
                / "candidates.jsonl"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        reports = self._evaluate(tmp_path, index_dir, ["e.gguf"])
        assert len(reports) == 2
        assert all(report.n_matched == 1 for report in reports)
        assert all(report.n_missing_candidates == 0 for report in reports)
        assert all(
            "e.gguf" in report.candidates_path and "r.gguf" in report.candidates_path
            for report in reports
        )

    def test_t_mdl_13_duplicate_pair_in_triple_fails(self, tmp_path: Path) -> None:
        """Повтор пары внутри тройки останавливает eval до отчёта."""
        predictions = tmp_path / "predictions"
        predictions.mkdir()
        self._write_predictions(
            predictions / "doc.docx.jsonl",
            [("e.gguf", "r.gguf", "a.gguf"), ("e.gguf", "r.gguf", "a.gguf")],
        )
        request = self._eval_request(tmp_path, tmp_path / "index", ["e.gguf"])
        with pytest.raises(EvaluationError, match="повторяется"):
            EvaluationRun().execute(request)

    def test_t_mdl_14_cosine_uses_first_embedding(self, tmp_path: Path) -> None:
        """Косинус обоих отчётов — первое имя embedding.models."""
        predictions = tmp_path / "predictions"
        predictions.mkdir()
        self._write_predictions(
            predictions / "doc.docx.jsonl",
            [
                ("second.gguf", "r.gguf", "a.gguf"),
                ("first.gguf", "r.gguf", "a.gguf"),
            ],
        )
        reports = self._evaluate(
            tmp_path,
            tmp_path / "index",
            ["first.gguf", "second.gguf"],
        )
        assert [report.cosine_embedding_model for report in reports] == [
            "first.gguf",
            "first.gguf",
        ]

    def _set_lists(self, config: AppConfig, answers: list[str]) -> None:
        """Подставляет короткие списки файлов в конфиг харнесса.

        Args:
            config: Конфиг прогона.
            answers: Имена моделей ответа.
        """
        config.embedding.model = "e.gguf"
        config.embedding.models = ["e.gguf"]
        config.reranker.model = "r.gguf"
        config.reranker.models = ["r.gguf"]
        config.answer.model = answers[0]
        config.answer.models = list(answers)

    def _config(
        self,
        embedding: list[str],
        reranker: list[str],
        answer: list[str],
    ) -> AppConfig:
        """Собирает конфиг только со списками моделей.

        Args:
            embedding: Имена эмбеддеров.
            reranker: Имена реранкеров.
            answer: Имена моделей ответа.

        Returns:
            AppConfig.
        """
        return AppConfig.model_validate(
            {
                "embedding": {"models": embedding},
                "reranker": {"models": reranker},
                "answer": {"models": answer},
            }
        )

    def _write_yaml(
        self,
        root: Path,
        embedding: list[str],
        reranker: list[str],
        answer: list[str],
    ) -> Path:
        """Пишет минимальный YAML со списками.

        Args:
            root: Каталог теста.
            embedding: Имена эмбеддеров.
            reranker: Имена реранкеров.
            answer: Имена моделей ответа.

        Returns:
            Путь к YAML.
        """
        payload = {
            "embedding": {"models": embedding},
            "reranker": {"models": reranker},
            "answer": {"models": answer},
        }
        path = root / "config.yaml"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _forbid_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Подменяет сборку приложения, чтобы фазы не стартовали.

        Args:
            monkeypatch: Фикстура pytest.
        """

        def fail_build(self: Application) -> None:
            """Падает, если проверка списков пропустила сборку."""
            raise AssertionError("фазы не должны стартовать")

        monkeypatch.setattr(Application, "build", fail_build)

    def _record(self, answer: str) -> AnswerRecord:
        """Строка ответа одной тройки.

        Args:
            answer: Текст ответа.

        Returns:
            AnswerRecord.
        """
        return AnswerRecord(
            doc_id="doc_a.docx",
            question_id="Q001",
            category="Срок",
            answer=answer,
            embedding_model="e.gguf",
            reranker_model="r.gguf",
            answer_model="a.gguf",
            model="a.gguf",
        )

    def _write_predictions(
        self,
        path: Path,
        triples: list[tuple[str, str, str]],
    ) -> None:
        """Пишет по строке на тройку для одной пары документа и вопроса.

        Args:
            path: Файл jsonl.
            triples: Тройки моделей.
        """
        lines = []
        for embedding_name, reranker_name, answer_name in triples:
            record = AnswerRecord(
                doc_id="doc.docx",
                question_id="Q1",
                category="Срок",
                embedding_model=embedding_name,
                reranker_model=reranker_name,
                answer_model=answer_name,
                model=answer_name,
            )
            lines.append(record.model_dump_json())
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _eval_request(
        self,
        root: Path,
        index_dir: Path,
        embedding_names: list[str],
    ) -> EvalRequest:
        """Собирает запрос eval с одним эталоном.

        Args:
            root: Каталог предсказаний и эталона.
            index_dir: Корень кандидатов.
            embedding_names: Список эмбеддеров конфига.

        Returns:
            EvalRequest.
        """
        ground_truth = root / "gt.json"
        ground_truth.write_text(
            json.dumps(
                [
                    {
                        "doc_id": "doc.docx",
                        "question_id": "Q1",
                        "expected_found": True,
                        "expected_answer": "год",
                    }
                ]
            ),
            encoding="utf-8",
        )
        config = AppConfig.model_validate(
            {
                "embedding": {"models": embedding_names},
                "phases": {"index_dir": str(index_dir)},
            }
        )
        return EvalRequest(
            config=config,
            ground_truth_path=str(ground_truth),
            predictions_dir=str(root / "predictions"),
        )

    def _evaluate(
        self,
        root: Path,
        index_dir: Path,
        embedding_names: list[str],
    ) -> list[EvalReport]:
        """Запускает группировку eval.

        Args:
            root: Каталог предсказаний и эталона.
            index_dir: Корень кандидатов.
            embedding_names: Список эмбеддеров конфига.

        Returns:
            Отчёты троек.
        """
        return EvaluationRun().execute(
            self._eval_request(root, index_dir, embedding_names)
        )
