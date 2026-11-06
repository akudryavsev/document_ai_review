"""T-GEN: ответ по готовому контексту."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from answer.generator import AnswerGenerator
from models.artifacts import AnswerRecord, Citation, ContextBundle, ContextCandidate
from models.config import AnswerConfig, AppConfig, RuntimeConfig
from models.gateways import GenerateRequest, GenerateResponse
from models.questions import Question, QuestionSet
from ports.gateways import LlmGateway


class ScriptedLlm(LlmGateway):
    """LLM-шлюз с заранее заданной очередью ответов.

    Attributes:
        steps: Ответы или исключения в порядке вызовов.
        calls: Запросы, которые получил infer.
    """

    name = "llm"

    def __init__(self, steps: list[GenerateResponse | Exception]) -> None:
        """Сохраняет очередь шагов.

        Args:
            steps: Что вернуть или бросить на каждом infer.
        """
        super().__init__()
        self.steps = list(steps)
        self.calls: list[GenerateRequest] = []

    def load(self) -> None:
        """Помечает шлюз загруженным."""
        self.loaded = True

    def unload(self) -> None:
        """Помечает шлюз выгруженным."""
        self.loaded = False

    def infer(self, request: GenerateRequest) -> GenerateResponse:
        """Возвращает следующий шаг очереди.

        Args:
            request: Промпт генератора.

        Returns:
            Следующий GenerateResponse.

        Raises:
            Exception: Если шаг очереди — исключение.
            RuntimeError: Очередь пуста или шлюз не загружен.
        """
        if not self.loaded:
            raise RuntimeError("LlmGateway не загружен")
        self.calls.append(request)
        if not self.steps:
            raise RuntimeError("очередь ответов пуста")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class PauseLog:
    """Запоминает аргументы time.sleep.

    Attributes:
        pauses: Длительности, с которыми вызывали sleep.
    """

    def __init__(self) -> None:
        """Создаёт пустой журнал пауз."""
        self.pauses: list[float] = []

    def sleep(self, seconds: float) -> None:
        """Записывает паузу вместо ожидания.

        Args:
            seconds: Секунды, которые просил генератор.
        """
        self.pauses.append(seconds)


class TestAnswerGenerator:
    """Каталог T-GEN-01…11."""

    def test_t_gen_01_empty_search_skips_llm(self) -> None:
        """Пустой поиск не вызывает infer и пишет системную фразу."""
        category = "Срок хранения персональных данных"
        synonyms = ["период хранения ПДн", "retention period"]
        bundle = self._bundle(
            category=category,
            synonyms=synonyms,
            empty=True,
        )
        question = self._question(category, synonyms)
        llm = self._llm([])
        generator = self._generator(llm)
        record = generator.generate(bundle, question)
        assert llm.calls == []
        assert record.answer == "В поиске по этому вопросу ничего не найдено."
        assert record.explanation.startswith("Искали по: ")
        category_at = record.explanation.index(category)
        first_at = record.explanation.index(synonyms[0])
        second_at = record.explanation.index(synonyms[1])
        assert category_at < first_at < second_at
        assert record.found is False
        assert record.reason == "no_relevant_chunks"
        assert record.status == "ok"
        assert record.prepared_context is None
        assert record.citations == []
        assert record.confidence is None
        assert record.tokens.prompt == 0
        assert record.tokens.completion == 0
        assert record.latency_ms == 0
        assert record.model == "Qwen/Qwen3-8B"

    def test_t_gen_02_prompt_contains_required_parts(self) -> None:
        """Промпт содержит вопрос, схему, причины и заголовок фрагмента."""
        category = "Срок хранения персональных данных"
        synonym = "период хранения ПДн"
        fragment = self._fragment(
            rank=1,
            text="Данные хранятся пять лет.",
            pages=[12, 13],
            path=["3. Регламент", "3.2.1. Сроки"],
            chunks=["c_0412"],
        )
        bundle = self._bundle(
            category=category,
            synonyms=[synonym],
            fragments=[fragment],
        )
        question = self._question(category, [synonym])
        other = QuestionSet(
            questions=[
                question,
                Question(id="Q009", category="Совсем другой вопрос", synonyms=["zzz"]),
            ]
        )
        llm = self._llm([self._response(self._valid_json())])
        generator = self._generator(llm, require_citations=True)
        generator.generate(bundle, question, questions=other)
        prompt = llm.calls[0].prompt or ""
        assert f"Вопрос: {category}" in prompt
        assert prompt.count("Вопрос:") == 1
        assert f"Вопрос: {synonym}" not in prompt
        assert "Совсем другой вопрос" not in prompt
        assert "zzz" not in prompt
        assert "answer" in prompt
        assert "explanation" in prompt
        assert "not_present" in prompt
        assert "insufficient_context" in prompt
        assert "ambiguous" in prompt
        assert "[1]" in prompt
        assert "стр. 12–13" in prompt
        assert "разд. 3.2.1. Сроки" in prompt
        assert "ссылкой [n]" in prompt
        assert synonym in prompt
        assert llm.calls[0].temperature == 0
        assert llm.calls[0].max_output_tokens is None
        assert llm.calls[0].doc_id == bundle.doc_id
        assert llm.calls[0].question_id == question.id

    def test_t_gen_03_renumbers_fragments_from_one(self) -> None:
        """Ранги 1 и 3 становятся [1] и [2] в прежнем относительном порядке."""
        low = self._fragment(
            rank=1,
            text="текст младшего ранга",
            pages=[4],
            path=["Глава", "Сроки"],
            chunks=["c1"],
        )
        high = self._fragment(
            rank=3,
            text="текст старшего ранга",
            pages=[8, 9],
            path=None,
            chunks=["c3"],
        )
        bundle = self._bundle(
            category="Срок хранения",
            synonyms=[],
            fragments=[high, low],
        )
        question = self._question("Срок хранения", [])
        llm = self._llm([self._response(self._valid_json())])
        generator = self._generator(llm)
        generator.generate(bundle, question)
        prompt = llm.calls[0].prompt or ""
        first = prompt.index("[1]")
        low_text = prompt.index("текст младшего ранга")
        second = prompt.index("[2]")
        high_text = prompt.index("текст старшего ранга")
        assert first < low_text < second < high_text
        assert "[3]" not in prompt
        heading = next(line for line in prompt.splitlines() if line.startswith("[2]"))
        assert "стр. 8–9" in heading
        assert "разд." not in heading
        assert "Искали по: Срок хранения" in prompt

    def test_t_gen_04_citations_not_required(self) -> None:
        """require_citations=false не добавляет требование ссылки [n]."""
        bundle = self._bundle(
            category="Срок хранения",
            synonyms=[],
            fragments=[
                self._fragment(
                    rank=1,
                    text="фрагмент",
                    pages=[1],
                    path=["Раздел"],
                    chunks=["c0"],
                )
            ],
        )
        question = self._question("Срок хранения", [])
        llm = self._llm([self._response(self._valid_json())])
        generator = self._generator(llm, require_citations=False)
        generator.generate(bundle, question)
        prompt = llm.calls[0].prompt or ""
        assert "ссылкой [n]" not in prompt
        assert "по фрагментам" in prompt
        assert "[1]" in prompt

    def test_t_gen_05_valid_json_becomes_record(self) -> None:
        """Валидный JSON с цитатой собирает запись без prepared_context."""
        fragment = self._fragment(
            rank=1,
            text="хранятся в течение 5 лет",
            pages=[12, 13],
            path=["3. Регламент", "3.2.1. Сроки"],
            chunks=["c_0412", "c_0413"],
        )
        bundle = self._bundle(
            category="Срок хранения",
            synonyms=["период"],
            fragments=[fragment],
            top_k=20,
            after_rerank=5,
            sections_used=["3.2.1. Сроки"],
        )
        question = self._question("Срок хранения", ["период"])
        body = self._valid_json(
            explanation="Срок назван во фрагменте",
            quote="хранятся 5 лет",
        )
        response = self._response(
            f"\n  {body}\n",
            prompt_tokens=42,
            completion_tokens=7,
            latency_ms=15.5,
        )
        llm = self._llm([response])
        generator = self._generator(llm, model="Qwen/Qwen3-8B")
        record = generator.generate(bundle, question)
        assert record.status == "ok"
        assert record.found is True
        assert record.explanation == "Срок назван во фрагменте"
        assert len(record.citations) == 1
        assert record.citations[0].chunk_ids == ["c_0412", "c_0413"]
        assert record.citations[0].quote == "хранятся 5 лет"
        assert record.citations[0].page_from == 12
        assert record.citations[0].page_to == 13
        assert record.citations[0].section_path == ["3. Регламент", "3.2.1. Сроки"]
        assert record.confidence is None
        assert record.verification_skipped is False
        assert record.tokens.prompt == 42
        assert record.tokens.completion == 7
        assert record.latency_ms == 15.5
        assert record.model == "Qwen/Qwen3-8B"
        assert record.prepared_context is None
        assert record.retrieval.candidates == 20
        assert record.retrieval.after_rerank == 5
        assert record.retrieval.after_verify == 1
        assert record.retrieval.sections_used == ["3.2.1. Сроки"]

    def test_t_gen_06_all_attempts_raise(self) -> None:
        """Все попытки бросают исключение — failed без исключения наружу."""
        bundle = self._bundle(
            category="Срок хранения",
            synonyms=["период"],
            fragments=[
                self._fragment(
                    rank=1,
                    text="текст фрагмента",
                    pages=[2],
                    path=["Сроки"],
                    chunks=["c0"],
                )
            ],
        )
        question = self._question("Срок хранения", ["период"])
        llm = self._llm([RuntimeError("down"), RuntimeError("down")])
        generator = self._generator(llm, retries=2)
        record = generator.generate(bundle, question)
        assert len(llm.calls) == 2
        assert record.status == "failed"
        assert record.reason == "model_unavailable"
        assert record.answer == "Модель недоступна."
        assert record.explanation == "Найденные фрагменты — в prepared_context."
        assert record.found is False
        assert record.citations == []
        assert record.prepared_context == llm.calls[0].prompt
        assert "текст фрагмента" in (record.prepared_context or "")
        assert record.tokens.prompt == 0
        assert record.tokens.completion == 0
        assert record.latency_ms == 0
        assert record.confidence is None

    def test_t_gen_07_retry_after_non_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Первый ответ не JSON, второй валиден; между ними одна пауза."""
        pauses = PauseLog()
        monkeypatch.setattr("answer.generator.time.sleep", pauses.sleep)
        bundle = self._nonempty("Срок хранения")
        question = self._question("Срок хранения", [])
        fenced = "```json\n" + self._valid_json(answer="первый") + "\n```"
        second = self._response(
            self._valid_json(answer="второй ответ", explanation="пояснение два"),
            prompt_tokens=3,
            completion_tokens=4,
            latency_ms=9.0,
        )
        llm = self._llm(
            [
                self._response(fenced, prompt_tokens=1, latency_ms=1.0),
                second,
            ]
        )
        generator = self._generator(llm, retries=2, pause=0.5)
        record = generator.generate(bundle, question)
        assert len(llm.calls) == 2
        assert pauses.pauses == [0.5]
        assert record.answer == "второй ответ"
        assert record.explanation == "пояснение два"
        assert record.status == "ok"
        assert record.tokens.prompt == 3
        assert record.latency_ms == 9.0

    def test_t_gen_08_invalid_reason_then_valid(self) -> None:
        """found=true вместе с reason не из null — неудачная попытка."""
        bundle = self._nonempty("Срок хранения")
        question = self._question("Срок хранения", [])
        first = self._valid_json(found=True, reason="not_present")
        second = self._valid_json(answer="годный", explanation="обоснование")
        llm = self._llm(
            [
                self._response(first),
                self._response(second, prompt_tokens=8, latency_ms=2.0),
            ]
        )
        generator = self._generator(llm, retries=2)
        record = generator.generate(bundle, question)
        assert len(llm.calls) == 2
        assert record.answer == "годный"
        assert record.explanation == "обоснование"
        assert record.tokens.prompt == 8

    def test_t_gen_09_citation_number_out_of_range(self) -> None:
        """Номер цитаты вне диапазона фрагментов уходит в повтор."""
        bundle = self._nonempty("Срок хранения")
        question = self._question("Срок хранения", [])
        bad = self._valid_json(n=9)
        good = self._valid_json(answer="после повтора", explanation="уже в диапазоне")
        llm = self._llm([self._response(bad), self._response(good)])
        generator = self._generator(llm, retries=2)
        record = generator.generate(bundle, question)
        assert len(llm.calls) == 2
        assert record.answer == "после повтора"
        assert record.status == "ok"

    def test_t_gen_10_citation_keeps_all_chunk_ids(self) -> None:
        """chunk_ids цитаты совпадают с chunks_included фрагмента."""
        chunks = ["c_a", "c_b", "c_c"]
        fragment = self._fragment(
            rank=1,
            text="три чанка одной секции",
            pages=[3],
            path=["Секция"],
            chunks=chunks,
        )
        bundle = self._bundle(
            category="Срок хранения",
            synonyms=[],
            fragments=[fragment],
        )
        question = self._question("Срок хранения", [])
        llm = self._llm([self._response(self._valid_json(quote="цитата из трёх"))])
        generator = self._generator(llm)
        record = generator.generate(bundle, question)
        assert len(record.citations) == 1
        assert record.citations[0].chunk_ids == chunks
        assert record.citations[0].quote == "цитата из трёх"

    def test_t_gen_11_found_without_citations_is_empty_search_record(self) -> None:
        """found=true и пустые citations не повторяются и не сохраняются как найденные."""
        category = "Срок хранения"
        synonyms = ["период", "retention"]
        bundle = self._bundle(
            category=category,
            synonyms=synonyms,
            fragments=[
                self._fragment(
                    rank=1,
                    text="есть фрагмент",
                    pages=[1],
                    path=["Секция"],
                    chunks=["c0"],
                )
            ],
            top_k=4,
            after_rerank=1,
        )
        question = self._question(category, synonyms)
        body = json.dumps(
            {
                "found": True,
                "answer": "Этого текста в записи быть не должно",
                "explanation": "И это пояснение модели тоже",
                "reason": None,
                "citations": [],
            },
            ensure_ascii=False,
        )
        llm = self._llm(
            [self._response(body, prompt_tokens=17, completion_tokens=3, latency_ms=6.0)]
        )
        generator = self._generator(llm, retries=2, require_citations=False)
        record = generator.generate(bundle, question)
        assert len(llm.calls) == 1
        assert record.answer == "В поиске по этому вопросу ничего не найдено."
        assert record.found is False
        assert record.reason == "no_relevant_chunks"
        assert record.status == "ok"
        assert record.citations == []
        assert record.prepared_context is None
        assert "Этого текста" not in record.answer
        assert "пояснение модели" not in record.explanation
        assert category in record.explanation
        assert record.explanation.index(synonyms[0]) < record.explanation.index(synonyms[1])
        assert record.tokens.prompt == 17
        assert record.tokens.completion == 3
        assert record.latency_ms == 6.0
        assert record.retrieval.candidates == 4
        assert record.retrieval.after_verify == 1

    def test_retries_below_one_is_config_error(self) -> None:
        """retries меньше 1 отвергается при загрузке конфига."""
        with pytest.raises(ValidationError):
            AppConfig(runtime=RuntimeConfig(retries=0))

    def test_found_true_with_no_relevant_chunks_rejected(self) -> None:
        """found=true и no_relevant_chunks вместе не проходят проверку записи."""
        with pytest.raises(ValidationError):
            AnswerRecord(
                doc_id="doc",
                question_id="Q001",
                category="Срок",
                found=True,
                answer="есть",
                explanation="есть",
                reason="no_relevant_chunks",
                citations=[
                    Citation(
                        chunk_ids=["c0"],
                        page_from=1,
                        page_to=1,
                        quote="цитата",
                    )
                ],
            )

    def _generator(
        self,
        llm: LlmGateway,
        *,
        require_citations: bool = True,
        retries: int = 2,
        pause: float = 0.0,
        model: str = "Qwen/Qwen3-8B",
    ) -> AnswerGenerator:
        """Собирает генератор с коротким конфигом.

        Args:
            llm: Поддельный шлюз.
            require_citations: Просьба о ссылке в промпте.
            retries: Число вызовов, включая первый.
            pause: Пауза между попытками.
            model: Имя модели в записи.

        Returns:
            AnswerGenerator.
        """
        return AnswerGenerator(
            answer_config=AnswerConfig(
                model=model,
                temperature=0,
                max_output_tokens=123,
                require_citations=require_citations,
            ),
            runtime_config=RuntimeConfig(retries=retries, retry_pause_sec=pause),
            llm=llm,
        )

    def _llm(self, steps: list[GenerateResponse | Exception]) -> ScriptedLlm:
        """Загружает шлюз с очередью шагов.

        Args:
            steps: Ответы или исключения.

        Returns:
            Загруженный ScriptedLlm.
        """
        gateway = ScriptedLlm(steps)
        gateway.load()
        return gateway

    def _question(self, category: str, synonyms: list[str]) -> Question:
        """Вопрос пары.

        Args:
            category: Канонический вопрос.
            synonyms: Синонимы поиска.

        Returns:
            Question с id Q001.
        """
        return Question(id="Q001", category=category, synonyms=synonyms)

    def _fragment(
        self,
        *,
        rank: int,
        text: str,
        pages: list[int],
        path: list[str] | None,
        chunks: list[str],
    ) -> ContextCandidate:
        """Один фрагмент бандла.

        Args:
            rank: rank_after_rerank.
            text: Текст без префикса.
            pages: Страницы.
            path: Путь секции или None.
            chunks: chunk_id фрагмента.

        Returns:
            ContextCandidate.
        """
        return ContextCandidate(
            rank_after_rerank=rank,
            rerank_score=0.9,
            section_path=path,
            pages=pages,
            text=text,
            chunks_included=chunks,
        )

    def _bundle(
        self,
        *,
        category: str,
        synonyms: list[str],
        empty: bool = False,
        fragments: list[ContextCandidate] | None = None,
        top_k: int = 20,
        after_rerank: int | None = None,
        sections_used: list[str] | None = None,
    ) -> ContextBundle:
        """Бандл пары документ × вопрос.

        Args:
            category: Канонический вопрос.
            synonyms: Синонимы.
            empty: Пустой поиск.
            fragments: Фрагменты непустого поиска.
            top_k: retrieval_meta.top_k.
            after_rerank: retrieval_meta.after_rerank.
            sections_used: Заголовки секций.

        Returns:
            ContextBundle.
        """
        chosen = [] if empty else list(fragments or [])
        rerank_count = 0 if empty else (
            len(chosen) if after_rerank is None else after_rerank
        )
        return ContextBundle(
            doc_id="reglament.docx",
            question_id="Q001",
            category=category,
            used_synonyms=list(synonyms),
            empty_search=empty,
            reason="no_relevant_chunks" if empty else None,
            candidates=chosen,
            retrieval_meta={
                "top_k": top_k,
                "after_rerank": rerank_count,
                "sections_used": list(sections_used or []),
                "embedding_model": "emb",
                "reranker_model": "rerank",
            },
        )

    def _nonempty(self, category: str) -> ContextBundle:
        """Короткий непустой бандл из одного фрагмента.

        Args:
            category: Вопрос.

        Returns:
            ContextBundle.
        """
        fragment = self._fragment(
            rank=1,
            text="один фрагмент",
            pages=[1],
            path=["Раздел"],
            chunks=["c0"],
        )
        return self._bundle(category=category, synonyms=[], fragments=[fragment])

    def _valid_json(
        self,
        *,
        found: bool = True,
        answer: str = "Пять лет",
        explanation: str = "Срок назван во фрагменте",
        reason: str | None = None,
        n: int = 1,
        quote: str = "цитата",
    ) -> str:
        """Собирает JSON модели.

        Args:
            found: Флаг найденного ответа.
            answer: Текст ответа.
            explanation: Пояснение.
            reason: Код причины или None.
            n: Номер фрагмента в цитате.
            quote: Текст цитаты.

        Returns:
            Строка JSON.
        """
        citations: list[dict[str, object]] = []
        if found and reason is None:
            citations = [{"n": n, "quote": quote}]
        if found and reason is not None:
            citations = [{"n": n, "quote": quote}]
        payload = {
            "found": found,
            "answer": answer,
            "explanation": explanation,
            "reason": reason,
            "citations": citations,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _response(
        self,
        text: str,
        *,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        latency_ms: float = 1.0,
    ) -> GenerateResponse:
        """Ответ шлюза с заданным текстом.

        Args:
            text: Тело ответа модели.
            prompt_tokens: Токены промпта.
            completion_tokens: Токены ответа.
            latency_ms: Задержка.

        Returns:
            GenerateResponse.
        """
        return GenerateResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
