"""Группировка предсказаний по тройке моделей."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from models.artifacts import AnswerRecord, GroundTruthItem
from models.requests import EvalRequest
from models.summary import EvalReport


class EvaluationError(Exception):
    """Eval остановлен до отчёта."""


class EvaluationRun:
    """Делит jsonl на тройки и сверяет пары с одним gt.json.

    Формулы Recall, MRR, nDCG и exact match остаются в спецификации eval.
    """

    def execute(self, request: EvalRequest) -> list[EvalReport]:
        """Собирает по отчёту на тройку в порядке первого появления.

        Args:
            request: EvalRequest с конфигом, эталоном и каталогом jsonl.

        Returns:
            Список EvalReport.

        Raises:
            EvaluationError: Строка без тройки, повтор пары или битый эталон.
        """
        ground_truth = self._load_ground_truth(request.ground_truth_path)
        records = self._load_predictions(request.predictions_dir)
        groups = self._group(records)
        cosine_name = (request.config.embedding.models or [""])[0]
        index_dir = request.config.phases.index_dir
        contexts_dir = request.config.phases.contexts_dir
        reports: list[EvalReport] = []
        for embedding_name, reranker_name, answer_name, rows in groups:
            matched, missing, candidates_path = self._match(
                ground_truth,
                rows,
                index_dir,
                embedding_name,
                reranker_name,
            )
            reports.append(
                EvalReport(
                    embedding_model=embedding_name,
                    reranker_model=reranker_name,
                    answer_model=answer_name,
                    cosine_embedding_model=cosine_name,
                    ground_truth_path=request.ground_truth_path,
                    predictions_dir=request.predictions_dir,
                    index_dir=index_dir,
                    contexts_dir=contexts_dir,
                    n_matched=matched,
                    n_missing_candidates=missing,
                    candidates_path=candidates_path,
                )
            )
        return reports

    def _load_ground_truth(self, path: str) -> list[GroundTruthItem]:
        """Читает массив эталона.

        Args:
            path: Путь к gt.json.

        Returns:
            Список GroundTruthItem.

        Raises:
            EvaluationError: Корень файла не массив или элемент не проходит схему.
        """
        raw_document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw_document, list):
            raise EvaluationError("gt.json должен быть массивом")
        items: list[GroundTruthItem] = []
        for raw_item in raw_document:
            try:
                items.append(GroundTruthItem.model_validate(raw_item))
            except ValidationError as error:
                raise EvaluationError("элемент gt.json не прошёл проверку") from error
        return items

    def _load_predictions(self, directory: str) -> list[AnswerRecord]:
        """Читает jsonl предсказаний в каталоге, без вложенных папок.

        Args:
            directory: Каталог results.

        Returns:
            Строки в порядке файлов и строк.

        Raises:
            EvaluationError: Строка не разбирается или в ней нет тройки.
        """
        records: list[AnswerRecord] = []
        for path in sorted(Path(directory).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = AnswerRecord.model_validate_json(line)
                except ValidationError as error:
                    raise EvaluationError(
                        "строка предсказания не разобрана"
                    ) from error
                if (
                    record.embedding_model == ""
                    or record.reranker_model == ""
                    or record.answer_model == ""
                ):
                    raise EvaluationError(
                        "в строке предсказания нет тройки моделей"
                    )
                records.append(record)
        return records

    def _group(
        self,
        records: list[AnswerRecord],
    ) -> list[tuple[str, str, str, list[AnswerRecord]]]:
        """Группирует строки и запрещает повтор пары внутри тройки.

        Args:
            records: Разобранные предсказания.

        Returns:
            Тройка и её строки в порядке первого появления.

        Raises:
            EvaluationError: Пара документа и вопроса повторяется в тройке.
        """
        order: list[tuple[str, str, str]] = []
        grouped: dict[tuple[str, str, str], list[AnswerRecord]] = {}
        seen_pairs: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
        for record in records:
            key = (
                record.embedding_model,
                record.reranker_model,
                record.answer_model,
            )
            pair = (record.doc_id, record.question_id)
            if key not in grouped:
                grouped[key] = []
                seen_pairs[key] = set()
                order.append(key)
            if pair in seen_pairs[key]:
                raise EvaluationError(
                    "пара документа и вопроса повторяется внутри тройки"
                )
            seen_pairs[key].add(pair)
            grouped[key].append(record)
        return [
            (key[0], key[1], key[2], grouped[key])
            for key in order
        ]

    def _match(
        self,
        ground_truth: list[GroundTruthItem],
        rows: list[AnswerRecord],
        index_dir: str,
        embedding_name: str,
        reranker_name: str,
    ) -> tuple[int, int, str]:
        """Считает сопоставленные пары и отсутствие файла кандидатов.

        Args:
            ground_truth: Эталон.
            rows: Строки одной тройки.
            index_dir: Корень кандидатов.
            embedding_name: Файл эмбеддера тройки.
            reranker_name: Файл реранкера тройки.

        Returns:
            Число совпавших пар, число пар без candidates.jsonl
            и путь кандидатов первой совпавшей пары.
        """
        present = {(row.doc_id, row.question_id) for row in rows}
        matched = 0
        missing = 0
        candidates_path = ""
        for item in ground_truth:
            pair = (item.doc_id, item.question_id)
            if pair not in present:
                continue
            matched += 1
            path = (
                Path(index_dir)
                / embedding_name
                / reranker_name
                / item.doc_id
                / "candidates.jsonl"
            )
            if candidates_path == "":
                candidates_path = str(path)
            if not path.is_file():
                missing += 1
        return matched, missing, candidates_path
