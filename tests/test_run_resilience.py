"""T-RUN: устойчивость пакетного прогона."""

from __future__ import annotations

from tests.helpers import ServiceFactory
from tests.support import RunHarness, build_harness


class TestRunResilience:
    """Каталог T-RUN-01…05."""

    def test_t_run_01_pdf_without_text_layer(self) -> None:
        """PDF без текстового слоя — failed; остальные документы продолжаются."""
        harness = build_harness(no_text_layer_docs={"doc_b.pdf"})
        service = ServiceFactory().build(harness)
        summary = service.run(harness.run_request())
        assert summary.phases["A1"].failed >= 1
        assert summary.phases["A1"].computed >= 1
        assert any(
            answer.doc_id == "doc_a.docx" and answer.status == "ok"
            for answer in harness.result_sink.answers
        )
        layer_failures = [
            event
            for event in harness.trace.events
            if event.stage == "A1"
            and event.status == "failed"
            and event.doc_id == "doc_b.pdf"
        ]
        assert layer_failures
        assert layer_failures[0].error == "no_text_layer"

    def test_t_run_02_zero_chunks_after_clean(self) -> None:
        """Ноль чанков — прогон не падает целиком; в сводке есть причина."""
        harness = build_harness(empty_docs={"doc_a.docx"})
        service = ServiceFactory().build(harness)
        summary = service.run(harness.run_request())
        assert summary.details.get("zero_chunks") or summary.phases["A1"].failed >= 1
        assert any(
            answer.doc_id == "doc_b.pdf" for answer in harness.result_sink.answers
        )
        zero_chunks = [
            event
            for event in harness.trace.events
            if event.stage == "A1"
            and event.status == "failed"
            and event.doc_id == "doc_a.docx"
            and event.error == "zero_chunks"
        ]
        assert zero_chunks

    def test_t_run_03_llm_retry_exhausted(self) -> None:
        """Исчерпание retry на одной паре — пара failed, следующие идут."""
        harness = build_harness(fail_pairs={("doc_a.docx", "Q001")})
        harness.generator.retries = 0
        service = ServiceFactory().build(harness)
        summary = service.run(harness.run_request())
        failed = [
            answer
            for answer in harness.result_sink.answers
            if answer.doc_id == "doc_a.docx" and answer.question_id == "Q001"
        ]
        assert failed and failed[0].status == "failed"
        ok_same_doc = [
            answer
            for answer in harness.result_sink.answers
            if answer.doc_id == "doc_a.docx" and answer.question_id == "Q002"
        ]
        assert ok_same_doc and ok_same_doc[0].status == "ok"
        assert summary.phases["B"].computed >= 1

    def test_t_run_04_verification_not_in_first_run(self) -> None:
        """Первый запуск не вызывает верификатор; verification_skipped остаётся false."""
        harness = build_harness(verifier_fail=True)
        service = ServiceFactory().build(harness)
        service.run(harness.run_request())
        assert harness.verifier.calls == []
        assert harness.result_sink.answers
        assert all(
            answer.verification_skipped is False
            for answer in harness.result_sink.answers
        )
        assert harness.llm.infer_calls

    def test_t_run_06_embedding_error_stays_in_trace(self) -> None:
        """Исключение эмбеддера не роняет фазу и пишется в трассу."""
        harness = build_harness()
        harness.embedding.raise_error = RuntimeError("embed broke")
        service = ServiceFactory().build(harness)
        summary = service.run(harness.run_request(phases=["A1"]))
        assert summary.phases["A1"].failed == 2
        assert summary.phases["A1"].computed == 0
        failures = [
            event
            for event in harness.trace.events
            if event.stage == "A1" and event.status == "failed"
        ]
        assert {event.doc_id for event in failures} == {"doc_a.docx", "doc_b.pdf"}
        assert all(event.error == "embed broke" for event in failures)
        execute_ok = [
            event
            for event in harness.trace.events
            if event.stage == "indexing_execute" and event.status == "ok"
        ]
        assert execute_ok

    def test_t_run_05_rerun_skips_all_phases(self, harness: RunHarness) -> None:
        """Повторный run с теми же параметрами — skipped по фазам."""
        service = ServiceFactory().build(harness)
        first = service.run(harness.run_request())
        embedding_loads = harness.embedding.load_calls
        second = service.run(harness.run_request())
        assert first.phases["A1"].computed >= 1
        assert second.phases["A1"].skipped >= 1
        assert second.phases["A2"].skipped >= 1
        assert second.phases["B"].skipped >= 1
        assert harness.embedding.load_calls == embedding_loads
