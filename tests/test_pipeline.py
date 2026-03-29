"""End-to-end behaviour of the assembled pipeline, entirely offline."""

from __future__ import annotations

import pytest

from rag.chunking import FixedSizeChunker, StructuralChunker
from rag.config import AppConfig, build_config
from rag.generate import AnswerComplete, AnswerDelta
from rag.pipeline import RagPipeline, RetrievalEvent
from rag.types import Document
from tests.conftest import MARKDOWN_DOCUMENT


def test_indexing_reports_what_it_did(pipeline: RagPipeline) -> None:
    stats = pipeline.stats()
    assert stats["chunks"] == pipeline.store.count() > 0
    assert stats["embedder"] == "hashing-512"
    assert stats["chunker"] == "structural"


def test_a_question_is_answered_from_the_indexed_text(pipeline: RagPipeline) -> None:
    result = pipeline.query("Чему равен retention.hours по умолчанию?")
    assert not result.answer.refused
    assert "168" in result.answer.text
    assert result.answer.citations
    assert all(
        citation.chunk_id in {item.chunk.chunk_id for item in result.retrieved}
        for citation in result.answer.citations
    )


def test_every_stage_is_timed(pipeline: RagPipeline) -> None:
    result = pipeline.query("порт брокера")
    stages = {timing.stage for timing in result.timings}
    assert stages == {"dense", "lexical", "fusion", "generation"}
    assert result.total_millis() > 0


def test_a_question_answered_only_by_an_exact_token_needs_the_lexical_half(
    offline_config: AppConfig, documents: list[Document]
) -> None:
    """The reason BM25 is in the pipeline at all: identifiers the embedder blurs."""
    dense_only = build_config({**offline_config.model_dump(), "lexical": {"enabled": False}})
    hybrid = RagPipeline.from_config(offline_config)
    dense = RagPipeline.from_config(dense_only)
    for built in (hybrid, dense):
        built.index(documents)

    question = "X-Kestrel-Quota-Remaining"
    hybrid_ids = [item.chunk.chunk_id for item in hybrid.retrieve(question)]
    dense_ids = [item.chunk.chunk_id for item in dense.retrieve(question)]
    target = next(
        chunk.chunk_id
        for chunk in hybrid.store.iter_chunks()
        if "X-Kestrel-Quota-Remaining" in chunk.text
    )
    assert hybrid_ids.index(target) <= dense_ids.index(target)


def test_metadata_filters_reach_both_retrievers(pipeline: RagPipeline) -> None:
    russian = pipeline.retrieve("порт", filters={"lang": "ru"})
    assert russian
    assert all(item.chunk.metadata["lang"] == "ru" for item in russian)


def test_a_filter_matching_nothing_produces_a_refusal(pipeline: RagPipeline) -> None:
    """No candidates means no markers, so the generator has nothing it may cite."""
    result = pipeline.query("порт", filters={"lang": "de"})
    assert result.retrieved == ()
    assert result.answer.refused
    assert result.answer.reason == "retrieval returned no chunks"


def test_reindexing_replaces_the_old_chunks(offline_config: AppConfig) -> None:
    """Chunks from a previous chunk size must not linger and keep scoring."""
    built = RagPipeline.from_config(offline_config)
    document = Document(doc_id="doc", text=MARKDOWN_DOCUMENT)
    built.chunker = StructuralChunker(max_chars=2000)
    first = built.index([document]).chunks
    built.chunker = FixedSizeChunker(size=120, overlap=0)
    second = built.index([document]).chunks
    assert second > first
    assert built.store.count() == second
    assert len({chunk.chunk_id for chunk in built.store.iter_chunks()}) == second


def test_the_lexical_index_is_rebuilt_with_the_store(offline_config: AppConfig) -> None:
    built = RagPipeline.from_config(offline_config)
    built.index([Document(doc_id="doc", text=MARKDOWN_DOCUMENT)])
    assert len(built.lexical_index) == built.store.count()


def test_reranking_is_applied_only_when_configured(
    offline_config: AppConfig, documents: list[Document]
) -> None:
    with_rerank = build_config(
        {**offline_config.model_dump(), "rerank": {"provider": "heuristic", "top_n": 20}}
    )
    built = RagPipeline.from_config(with_rerank)
    built.index(documents)
    result = built.query("срок хранения событий")
    assert all("rerank" in item.ranks for item in result.retrieved)
    assert "rerank" in {timing.stage for timing in result.timings}


def test_indexing_nothing_is_not_an_error(offline_config: AppConfig) -> None:
    built = RagPipeline.from_config(offline_config)
    report = built.index([])
    assert (report.documents, report.chunks) == (0, 0)
    assert built.query("что угодно").answer.refused


def test_streaming_emits_sources_then_text_then_the_final_answer(pipeline: RagPipeline) -> None:
    events = list(pipeline.stream("Чему равен retention.hours?"))
    assert isinstance(events[0], RetrievalEvent)
    assert events[0].retrieved
    assert isinstance(events[-1], AnswerComplete)
    assert any(isinstance(event, AnswerDelta) for event in events)


def test_result_serialises_to_a_json_friendly_shape(pipeline: RagPipeline) -> None:
    payload = pipeline.query("порт брокера").to_dict()
    assert set(payload) == {
        "query",
        "answer",
        "refused",
        "reason",
        "citations",
        "retrieved",
        "timings_ms",
    }
    assert payload["retrieved"][0]["chunk_id"]


def test_retrieval_is_reproducible(pipeline: RagPipeline) -> None:
    first = [item.chunk.chunk_id for item in pipeline.retrieve("срок хранения")]
    second = [item.chunk.chunk_id for item in pipeline.retrieve("срок хранения")]
    assert first == second


@pytest.mark.parametrize("question", ["", "   "])
def test_a_blank_question_returns_nothing_rather_than_everything(
    pipeline: RagPipeline, question: str
) -> None:
    assert pipeline.query(question).answer.refused
