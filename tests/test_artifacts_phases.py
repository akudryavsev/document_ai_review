"""T-ART: артефакты, хеши и empty_search."""

from __future__ import annotations

from models.artifacts import ArtifactMeta
from tests.helpers import ServiceFactory
from tests.support import build_harness


class TestArtifactsPhases:
    """Каталог T-ART-01…06."""

    def test_t_art_01_matching_hashes_skip_gateway(self) -> None:
        """Совпавшие хеши — фаза skipped; шлюз не вызывается."""
        harness = build_harness()
        service = ServiceFactory().build(harness)
        service.run(harness.run_request(phases=["A1"]))
        loads_after_first = harness.embedding.load_calls
        infer_after_first = len(harness.embedding.infer_calls)
        second = service.run(harness.run_request(phases=["A1"]))
        assert second.phases["A1"].skipped >= 1
        assert harness.embedding.load_calls == loads_after_first
        assert len(harness.embedding.infer_calls) == infer_after_first

    def test_t_art_02_chunking_hash_change_invalidates_dependents(self) -> None:
        """Смена chunking_hash — инвалидация A1 и зависимых A2/B."""
        harness = build_harness()
        service = ServiceFactory().build(harness)
        service.run(harness.run_request())
        harness.config.chunking.max_tokens = 256
        third = service.run(harness.run_request())
        assert third.phases["A1"].computed >= 1
        assert third.phases["A2"].computed >= 1
        assert third.phases["B"].computed >= 1

    def test_t_art_03_force_phase_recomputes(self) -> None:
        """--force-phase при валидных хешах — фаза пересчитывается."""
        harness = build_harness()
        service = ServiceFactory().build(harness)
        service.run(harness.run_request(phases=["A1"]))
        forced = service.run(
            harness.run_request(phases=["A1"], force_phases={"A1"})
        )
        assert forced.phases["A1"].computed >= 1

    def test_t_art_04_phase_b_without_contexts_errors(self) -> None:
        """Только B без контекстов — явная ошибка по парам."""
        harness = build_harness()
        service = ServiceFactory().build(harness)
        summary = service.run(harness.run_request(phases=["B"]))
        assert summary.phases["B"].failed >= 1
        assert harness.result_sink.answers
        assert all(answer.status == "failed" for answer in harness.result_sink.answers)
        assert all(
            answer.reason == "missing_context" for answer in harness.result_sink.answers
        )

    def test_t_art_05_empty_search_calls_llm_skips_verify(self) -> None:
        """empty_search=true — LLM вызывается, verify пропускается, found=false."""
        harness = build_harness(empty_pairs={("doc_a.docx", "Q001")})
        service = ServiceFactory().build(harness)
        service.run(harness.run_request())
        empty_answers = [
            answer
            for answer in harness.result_sink.answers
            if answer.doc_id == "doc_a.docx" and answer.question_id == "Q001"
        ]
        assert empty_answers
        answer = empty_answers[0]
        assert answer.found is False
        assert answer.reason == "no_relevant_chunks"
        assert harness.llm.infer_calls
        # Для пустой пары verify не вызывался.
        assert ("doc_a.docx", "Q001") not in harness.verifier.calls

    def test_t_art_06_non_empty_search_uses_category_not_synonyms(self) -> None:
        """empty_search=false — LLM с category; synonyms не подставляются как вопрос."""
        harness = build_harness()
        service = ServiceFactory().build(harness)
        service.run(harness.run_request())
        category_calls = [
            call
            for call in harness.generator.calls
            if call["question_id"] == "Q001" and call["doc_id"] == "doc_a.docx"
        ]
        assert category_calls
        assert category_calls[0]["category"] == "Срок хранения"
        assert category_calls[0]["empty_search"] is False
        llm_prompts = [
            request.prompt
            for request in harness.llm.infer_calls
            if request.question_id == "Q001" and request.doc_id == "doc_a.docx"
        ]
        assert llm_prompts
        assert llm_prompts[0] == "Срок хранения"
        assert "retention" not in (llm_prompts[0] or "")

    def test_preseeded_meta_with_upstream_hashes(self) -> None:
        """Проверка записи upstream-хешей в meta A2 (вспомогательный инвариант)."""
        harness = build_harness()
        service = ServiceFactory().build(harness)
        service.run(harness.run_request(phases=["A1", "A2"]))
        meta = harness.store.load_candidates_meta("doc_a.docx")
        assert isinstance(meta, ArtifactMeta)
        assert meta.chunking_hash == harness.config.chunking_hash
        assert meta.retrieval_config_hash == harness.config.retrieval_config_hash
        assert meta.reranker_config_hash == harness.config.reranker_config_hash
