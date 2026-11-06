"""Файловые реализации источников."""

from __future__ import annotations

import json
from pathlib import Path

from models.artifacts import GroundTruthItem
from models.documents import DocumentRef
from models.questions import QuestionSet
from ports.sources import DocumentSource, GroundTruthSource, QuestionSource


class FileDocumentSource(DocumentSource):
    """Читает DocumentRef из каталога рекурсивно.

    Attributes:
        documents_dir: Корневой каталог документов.
    """

    def __init__(self, documents_dir: str | Path) -> None:
        """Сохраняет путь к каталогу.

        Args:
            documents_dir: Каталог с .docx / .pdf.
        """
        self.documents_dir = Path(documents_dir)

    def list_documents(self) -> list[DocumentRef]:
        """Обходит каталог и собирает .docx / .pdf.

        Returns:
            Список DocumentRef, отсортированный по doc_id.
        """
        if not self.documents_dir.exists():
            return []
        documents: list[DocumentRef] = []
        for path in sorted(self.documents_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in {".docx", ".pdf"}:
                continue
            documents.append(
                DocumentRef(
                    doc_id=path.name,
                    path=str(path),
                    suffix=suffix,
                )
            )
        return documents


class FileQuestionSource(QuestionSource):
    """Загружает и валидирует JSON вопросов.

    Attributes:
        path: Путь к файлу вопросов.
    """

    def __init__(self, path: str | Path) -> None:
        """Сохраняет путь к файлу.

        Args:
            path: Путь к questions.json.
        """
        self.path = Path(path)

    def load(self) -> QuestionSet:
        """Читает JSON и проверяет схему QuestionSet.

        Returns:
            QuestionSet.

        Raises:
            FileNotFoundError: Файл отсутствует.
            ValueError: Схема или инварианты нарушены.
            json.JSONDecodeError: Невалидный JSON.
        """
        raw_text = self.path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
        return QuestionSet.model_validate(payload)


class FileGroundTruthSource(GroundTruthSource):
    """Загружает gt.json.

    Attributes:
        path: Путь к gt.json.
    """

    def __init__(self, path: str | Path) -> None:
        """Сохраняет путь к файлу.

        Args:
            path: Путь к gt.json.
        """
        self.path = Path(path)

    def load(self) -> list[GroundTruthItem]:
        """Читает список эталонов.

        Returns:
            Список GroundTruthItem.
        """
        raw_text = self.path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
        return [GroundTruthItem.model_validate(item) for item in payload]
