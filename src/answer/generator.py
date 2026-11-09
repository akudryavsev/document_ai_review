"""Генерация AnswerRecord по ContextBundle."""

from __future__ import annotations

import json
import time

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from models.artifacts import (
    AnswerRecord,
    Citation,
    ContextBundle,
    ContextCandidate,
    RetrievalStats,
    TokenUsage,
)
from models.config import AnswerConfig, RuntimeConfig
from models.gateways import GenerateRequest, GenerateResponse
from models.questions import Question, QuestionSet
from ports.gateways import LlmGateway
from ports.pipeline import AnswerGenerator as AnswerGeneratorPort


class ModelCitation(BaseModel):
    """Одна цитата в JSON модели.

    Attributes:
        n: Номер фрагмента в промпте, с 1.
        quote: Цитата из фрагмента.
    """

    model_config = ConfigDict(extra="forbid")

    n: int
    quote: str

    @field_validator("quote")
    @classmethod
    def strip_quote(cls, value: str) -> str:
        """Обрезает пробелы и запрещает пустую цитату.

        Args:
            value: Сырой quote из JSON.

        Returns:
            Строка без крайних пробелов.

        Raises:
            ValueError: После strip цитата пустая.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("пустая цитата")
        return stripped


class ModelAnswer(BaseModel):
    """JSON одного ответа модели. Лишние поля запрещены.

    Attributes:
        found: Нашла ли модель ответ во фрагментах.
        answer: Текст ответа.
        explanation: Пояснение отдельно от ответа.
        reason: Код причины или null.
        citations: Цитаты с номерами фрагментов.
    """

    model_config = ConfigDict(extra="forbid")

    found: bool
    answer: str
    explanation: str
    reason: str | None = None
    citations: list[ModelCitation]

    @field_validator("answer", "explanation")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """Обрезает пробелы и запрещает пустой ответ или пояснение.

        Args:
            value: Сырое поле JSON.

        Returns:
            Строка без крайних пробелов.

        Raises:
            ValueError: После strip строка пустая.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("пустая строка")
        return stripped

    @model_validator(mode="after")
    def check_reason(self) -> ModelAnswer:
        """Связывает found с допустимым reason.

        Returns:
            Тот же объект, если пара полей согласована.

        Raises:
            ValueError: found=true с непустым reason или found=false
                с кодом вне списка первого запуска.
        """
        if self.found:
            if self.reason is not None:
                raise ValueError("found=true требует reason=null")
            return self
        allowed = {"not_present", "insufficient_context", "ambiguous"}
        if self.reason not in allowed:
            raise ValueError("недопустимый reason")
        return self


class AnswerGenerator(AnswerGeneratorPort):
    """Пишет ответ пустого поиска или разбирает JSON модели.

    Attributes:
        answer_config: Параметры ответа, кроме лимита длины.
        runtime_config: Число попыток и пауза между ними.
        llm: Уже загруженный фазой шлюз.
    """

    _empty_answer = "В поиске по этому вопросу ничего не найдено."
    _unavailable_answer = "Модель недоступна."
    _unavailable_explanation = "Найденные фрагменты — в prepared_context."
    _citation_rule = "Каждый факт сопровождай ссылкой [n] и цитатой в JSON."

    def __init__(
        self,
        answer_config: AnswerConfig,
        runtime_config: RuntimeConfig,
        llm: LlmGateway,
    ) -> None:
        """Сохраняет конфиг и шлюз. Модель генератор не загружает.

        Args:
            answer_config: AnswerConfig.
            runtime_config: RuntimeConfig с retries и retry_pause_sec.
            llm: LlmGateway, который фаза уже загрузила.
        """
        self.answer_config = answer_config
        self.runtime_config = runtime_config
        self.llm = llm
        self.embedding_model = ""
        self.reranker_model = ""

    def _combination(self) -> dict[str, str]:
        """Имена файлов текущей тройки для строки ответа.

        Returns:
            Поля model, embedding_model, reranker_model и answer_model.
        """
        answer_name = self.answer_config.model or ""
        return {
            "model": answer_name,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "answer_model": answer_name,
        }

    def generate(
        self,
        bundle: ContextBundle,
        question: Question,
        questions: QuestionSet | None = None,
    ) -> AnswerRecord:
        """Собирает AnswerRecord. Набор вопросов на текст не влияет.

        Args:
            bundle: Контекст пары. Пустой поиск уже помечен флагом.
            question: Вопрос; в промпт идёт question.category.
            questions: Полный набор. В v1 не подменяет category.

        Returns:
            Запись со status ok или failed. Исключения шлюза наружу не идут.
        """
        del questions
        if bundle.empty_search:
            return self._not_found(bundle, question, response=None)
        prompt = self._build_prompt(bundle, question)
        last_response: GenerateResponse | None = None
        attempt_index = 0
        while attempt_index < self.runtime_config.retries:
            # Пауза только между попытками, не перед первой и не после последней.
            if attempt_index > 0:
                self._pause(self.runtime_config.retry_pause_sec)
            attempt_index += 1
            try:
                response = self._call_model(prompt, bundle, question)
            except Exception:  # noqa: BLE001 — любая ошибка шлюза это неудачная попытка
                continue
            last_response = response
            parsed = self._parse(response.text, len(bundle.candidates))
            if parsed is None:
                continue
            # Пустые цитаты при found=true — не найденный ответ и не повод повторять.
            if parsed.found and not parsed.citations:
                return self._not_found(bundle, question, response=response)
            return self._from_model(bundle, question, parsed, response)
        return self._unavailable(bundle, question, prompt, last_response)

    def _call_model(
        self,
        prompt: str,
        bundle: ContextBundle,
        question: Question,
    ) -> GenerateResponse:
        """Вызывает шлюз без лимита длины ответа.

        Args:
            prompt: Текст, который уходит в модель.
            bundle: Бандл пары.
            question: Вопрос пары.

        Returns:
            GenerateResponse шлюза.
        """
        request = GenerateRequest(
            prompt=prompt,
            temperature=self.answer_config.temperature,
            doc_id=bundle.doc_id,
            question_id=question.id,
        )
        return self.llm.infer(request)

    def _pause(self, seconds: float) -> None:
        """Ждёт перед следующей попыткой.

        Args:
            seconds: runtime.retry_pause_sec. Ноль паузу не ставит.
        """
        if seconds <= 0:
            return
        time.sleep(seconds)

    def _build_prompt(self, bundle: ContextBundle, question: Question) -> str:
        """Собирает промпт: вопрос, схема JSON и фрагменты с 1.

        Args:
            bundle: Непустой бандл.
            question: Вопрос; в текст берётся category.

        Returns:
            Промпт одним блоком.
        """
        lines = [
            f"Вопрос: {question.category}",
            "Дай ответ и отдельно пояснение с обоснованием.",
            "Отвечай только по фрагментам, без выхода за их пределы.",
            (
                "Если ответа в фрагментах нет, верни found=false и reason "
                "одну из: not_present, insufficient_context, ambiguous."
            ),
            "Верни один JSON-объект и ничего вокруг него.",
            (
                "Поля JSON: found, answer, explanation, reason, "
                "citations (список объектов с полями n и quote)."
            ),
        ]
        if self.answer_config.require_citations:
            lines.append(self._citation_rule)
        lines.append(self._sought_line(question.category, bundle.used_synonyms))
        lines.append("")
        lines.append("Фрагменты:")
        ordered = self._ordered_fragments(bundle)
        for index, fragment in enumerate(ordered, start=1):
            lines.append(self._heading(index, fragment))
            lines.append(fragment.text)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _heading(self, number: int, fragment: ContextCandidate) -> str:
        """Заголовок фрагмента в промпте. Это не префикс поля text бандла.

        Args:
            number: Номер в промпте, с 1.
            fragment: Фрагмент бандла.

        Returns:
            Строка вида [n] стр. …, разд. …
        """
        page_label = self._page_label(fragment.pages)
        heading = f"[{number}] {page_label}"
        if fragment.section_path:
            last_title = fragment.section_path[-1]
            heading = f"{heading}, разд. {last_title}"
        return heading

    def _page_label(self, pages: list[int]) -> str:
        """Форматирует минимум и максимум страниц.

        Args:
            pages: Номера страниц фрагмента.

        Returns:
            Подпись стр. from–to, стр. page или стр. —.
        """
        if not pages:
            return "стр. —"
        first = min(pages)
        last = max(pages)
        if first == last:
            return f"стр. {first}"
        return f"стр. {first}–{last}"

    def _sought_line(self, category: str, synonyms: list[str]) -> str:
        """Строка «Искали по» с category и синонимами.

        Args:
            category: Канонический вопрос.
            synonyms: Синонимы в исходном порядке.

        Returns:
            Одна строка. Пустой список синонимов оставляет только category.
        """
        parts = [category]
        parts.extend(synonyms)
        joined = "; ".join(parts)
        return f"Искали по: {joined}"

    def _ordered_fragments(self, bundle: ContextBundle) -> list[ContextCandidate]:
        """Сортирует фрагменты по rank_after_rerank.

        Args:
            bundle: Бандл.

        Returns:
            Копия списка от меньшего ранга к большему.
        """
        ordered = list(bundle.candidates)
        ordered.sort(key=lambda fragment: fragment.rank_after_rerank)
        return ordered

    def _parse(self, raw_text: str, fragment_count: int) -> ModelAnswer | None:
        """Разбирает один JSON-объект. Ограда и чужие поля — неудача.

        Args:
            raw_text: Текст ответа модели.
            fragment_count: Сколько фрагментов было в промпте.

        Returns:
            ModelAnswer или None, если попытку надо повторить.
        """
        stripped = raw_text.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        try:
            parsed = ModelAnswer.model_validate(payload)
        except ValidationError:
            return None
        for citation in parsed.citations:
            if citation.n < 1 or citation.n > fragment_count:
                return None
        return parsed

    def _not_found(
        self,
        bundle: ContextBundle,
        question: Question,
        response: GenerateResponse | None,
    ) -> AnswerRecord:
        """Системная запись «ничего не найдено». Текст модели не копируется.

        Args:
            bundle: Бандл пары.
            question: Вопрос.
            response: Ответ шлюза, если модель уже ответила пустыми цитатами.
                Для пустого поиска — None, токены тогда нули.

        Returns:
            AnswerRecord со status ok.
        """
        usage, latency_ms = self._usage(response)
        return AnswerRecord(
            doc_id=bundle.doc_id,
            question_id=question.id,
            category=question.category,
            used_synonyms=list(bundle.used_synonyms),
            found=False,
            answer=self._empty_answer,
            explanation=self._sought_line(question.category, bundle.used_synonyms),
            reason="no_relevant_chunks",
            citations=[],
            confidence=None,
            retrieval=self._retrieval(bundle),
            tokens=usage,
            **self._combination(),
            latency_ms=latency_ms,
            status="ok",
            verification_skipped=False,
            prepared_context=None,
        )

    def _from_model(
        self,
        bundle: ContextBundle,
        question: Question,
        parsed: ModelAnswer,
        response: GenerateResponse,
    ) -> AnswerRecord:
        """Переносит разобранный JSON в AnswerRecord.

        Args:
            bundle: Бандл, из которого берутся страницы и chunk_ids.
            question: Вопрос.
            parsed: Проверенный JSON модели.
            response: Успешный ответ шлюза.

        Returns:
            AnswerRecord со status ok.
        """
        usage, latency_ms = self._usage(response)
        citations = self._citations(bundle, parsed)
        return AnswerRecord(
            doc_id=bundle.doc_id,
            question_id=question.id,
            category=question.category,
            used_synonyms=list(bundle.used_synonyms),
            found=parsed.found,
            answer=parsed.answer,
            explanation=parsed.explanation,
            reason=parsed.reason,
            citations=citations,
            confidence=None,
            retrieval=self._retrieval(bundle),
            tokens=usage,
            **self._combination(),
            latency_ms=latency_ms,
            status="ok",
            verification_skipped=False,
            prepared_context=None,
        )

    def _unavailable(
        self,
        bundle: ContextBundle,
        question: Question,
        prompt: str,
        response: GenerateResponse | None,
    ) -> AnswerRecord:
        """Запись, когда годный JSON так и не получен.

        Args:
            bundle: Бандл пары.
            question: Вопрос.
            prompt: Промпт, который уходил в модель.
            response: Последний GenerateResponse или None, если каждый
                вызов только бросил исключение.

        Returns:
            AnswerRecord со status failed. Исключения нет.
        """
        usage, latency_ms = self._usage(response)
        return AnswerRecord(
            doc_id=bundle.doc_id,
            question_id=question.id,
            category=question.category,
            used_synonyms=list(bundle.used_synonyms),
            found=False,
            answer=self._unavailable_answer,
            explanation=self._unavailable_explanation,
            reason="model_unavailable",
            citations=[],
            confidence=None,
            retrieval=self._retrieval(bundle),
            tokens=usage,
            **self._combination(),
            latency_ms=latency_ms,
            status="failed",
            verification_skipped=False,
            prepared_context=prompt,
        )

    def _citations(
        self,
        bundle: ContextBundle,
        parsed: ModelAnswer,
    ) -> list[Citation]:
        """Строит цитаты записи из номеров промпта.

        Args:
            bundle: Бандл с фрагментами.
            parsed: JSON модели.

        Returns:
            Список Citation. chunk_ids копирует chunks_included фрагмента.
        """
        ordered = self._ordered_fragments(bundle)
        citations: list[Citation] = []
        for item in parsed.citations:
            fragment = ordered[item.n - 1]
            page_from, page_to = self._page_bounds(fragment.pages)
            path = None
            if fragment.section_path is not None:
                path = list(fragment.section_path)
            citations.append(
                Citation(
                    chunk_ids=list(fragment.chunks_included),
                    page_from=page_from,
                    page_to=page_to,
                    section_path=path,
                    quote=item.quote,
                )
            )
        return citations

    def _page_bounds(self, pages: list[int]) -> tuple[int, int]:
        """Минимум и максимум страниц цитаты.

        Args:
            pages: Номера страниц фрагмента.

        Returns:
            Пара (page_from, page_to). Пустой список даёт нули.
        """
        if not pages:
            return (0, 0)
        return (min(pages), max(pages))

    def _retrieval(self, bundle: ContextBundle) -> RetrievalStats:
        """Собирает блок retrieval из метаданных бандла.

        Args:
            bundle: Бандл пары.

        Returns:
            RetrievalStats. after_verify равен числу фрагментов бандла.
        """
        meta = bundle.retrieval_meta
        top_k = meta.get("top_k")
        if top_k is None:
            candidates = 0
        else:
            candidates = int(top_k)
        after_rerank_raw = meta.get("after_rerank")
        if after_rerank_raw is None:
            after_rerank = len(bundle.candidates)
        else:
            after_rerank = int(after_rerank_raw)
        sections_raw = meta.get("sections_used")
        if sections_raw is None:
            sections_used: list[str] = []
        else:
            sections_used = [str(item) for item in sections_raw]
        return RetrievalStats(
            candidates=candidates,
            after_rerank=after_rerank,
            after_verify=len(bundle.candidates),
            sections_used=sections_used,
        )

    def _usage(
        self,
        response: GenerateResponse | None,
    ) -> tuple[TokenUsage, float]:
        """Берёт токены и задержку с ответа шлюза.

        Args:
            response: Ответ или None.

        Returns:
            Пара (TokenUsage, latency_ms). Без ответа оба нули.
        """
        if response is None:
            return TokenUsage(prompt=0, completion=0), 0.0
        usage = TokenUsage(
            prompt=response.prompt_tokens,
            completion=response.completion_tokens,
        )
        return usage, response.latency_ms
