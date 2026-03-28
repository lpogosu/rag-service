"""Citation enforcement, refusal, and the streaming JSON extractor.

These are the tests that decide whether the service can be trusted: an answer that
cites a source it was never shown is worse than no answer, because it is convincing.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from rag.generate import (
    DEFAULT_REFUSAL,
    AnswerComplete,
    AnswerDelta,
    ExtractiveGenerator,
    GenerationSettings,
    JsonStringFieldStreamer,
    OllamaGenerator,
    build_context,
    build_generator,
    enforce_citations,
)
from rag.ollama import OllamaClient, OllamaSettings
from rag.text import Analyzer
from rag.types import Chunk, FusedChunk
from tests.conftest import make_chunk

BASE_URL = "http://ollama.test:11434"

CONTEXT = [
    FusedChunk(
        chunk=make_chunk(
            "cfg::0001",
            "Параметр retention.hours задаёт срок хранения событий и по умолчанию равен 168 часам.",
            doc_id="broker-config",
        ),
        score=0.9,
        ranks={"dense": 1},
    ),
    FusedChunk(
        chunk=make_chunk(
            "net::0002",
            "Брокер принимает клиентские подключения на порту 7420.",
            doc_id="overview",
        ),
        score=0.5,
        ranks={"lexical": 1},
    ),
]

SETTINGS = GenerationSettings()


def markers() -> dict[int, Chunk]:
    return build_context(CONTEXT, SETTINGS).markers


def test_context_numbers_chunks_from_one() -> None:
    block = build_context(CONTEXT, SETTINGS)
    assert list(block.markers) == [1, 2]
    assert block.markers[1].chunk_id == "cfg::0001"
    assert block.text.startswith("[1] ")


def test_context_drops_whole_chunks_when_the_budget_runs_out() -> None:
    """A truncated chunk can be cited for a claim its missing half contradicts."""
    settings = GenerationSettings(max_context_chars=100)
    block = build_context(CONTEXT, settings)
    assert list(block.markers) == [1]
    assert "7420" not in block.text


def test_empty_retrieval_refuses_without_calling_a_model() -> None:
    answer = enforce_citations("что угодно", [1, 2], {}, SETTINGS)
    assert answer.refused
    assert answer.text == DEFAULT_REFUSAL
    assert answer.citations == ()
    assert answer.reason == "retrieval returned no chunks"


def test_a_citation_that_does_not_resolve_is_dropped() -> None:
    answer = enforce_citations("Ответ [1] и выдумка [9].", [1, 9], markers(), SETTINGS)
    assert not answer.refused
    assert [citation.marker for citation in answer.citations] == [1]
    assert "[9]" not in answer.text
    assert "[1]" in answer.text


def test_an_answer_whose_every_citation_is_invented_is_refused() -> None:
    answer = enforce_citations("Уверенный ответ [7].", [7], markers(), SETTINGS)
    assert answer.refused
    assert answer.reason == "no citation resolved to a retrieved chunk"
    assert answer.text == DEFAULT_REFUSAL


def test_an_answer_with_no_citations_at_all_is_refused() -> None:
    answer = enforce_citations("Просто утверждение без источника.", [], markers(), SETTINGS)
    assert answer.refused


def test_markers_found_only_in_the_text_still_count() -> None:
    """Models frequently write [2] in the prose and forget the citations array."""
    answer = enforce_citations("Порт 7420 [2].", [], markers(), SETTINGS)
    assert not answer.refused
    assert [citation.marker for citation in answer.citations] == [2]


def test_citations_resolve_to_the_chunk_and_document_that_were_retrieved() -> None:
    answer = enforce_citations("Срок хранения 168 часов [1].", [1], markers(), SETTINGS)
    citation = answer.citations[0]
    assert citation.chunk_id == "cfg::0001"
    assert citation.doc_id == "broker-config"
    assert "retention.hours" in citation.quote


def test_an_empty_answer_body_is_refused_even_with_a_valid_citation() -> None:
    answer = enforce_citations("[1]  ", [1], markers(), SETTINGS)
    assert not answer.refused  # the marker itself is content the reader can follow
    stripped = enforce_citations("   ", [1], markers(), SETTINGS)
    assert stripped.refused
    assert stripped.reason == "model returned an empty answer"


def test_extractive_generator_answers_from_the_retrieved_text() -> None:
    generator = ExtractiveGenerator(Analyzer())
    answer = generator.generate("Чему равен retention.hours?", CONTEXT)
    assert not answer.refused
    assert "168" in answer.text
    assert answer.cited_chunk_ids == ("cfg::0001",)


def test_extractive_generator_refuses_on_empty_retrieval() -> None:
    answer = ExtractiveGenerator(Analyzer()).generate("что угодно", [])
    assert answer.refused
    assert answer.reason == "retrieval returned no chunks"


def test_the_support_threshold_turns_weak_matches_into_refusals() -> None:
    """The knob measured in the README's refusal table."""
    permissive = ExtractiveGenerator(Analyzer(), min_support=0.0)
    strict = ExtractiveGenerator(Analyzer(), min_support=0.9)
    question = "Какой срок хранения событий и какие скидки предусмотрены для партнёров?"
    assert not permissive.generate(question, CONTEXT).refused
    assert strict.generate(question, CONTEXT).refused


def test_extractive_stream_ends_with_the_authoritative_answer() -> None:
    events = list(ExtractiveGenerator(Analyzer()).stream("Чему равен retention.hours?", CONTEXT))
    assert isinstance(events[-1], AnswerComplete)
    assert any(isinstance(event, AnswerDelta) for event in events)
    assert "168" in "".join(
        event.text for event in events if isinstance(event, AnswerDelta)
    )


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (['{"answer": "hello", "citations": [1]}'], "hello"),
        (['{"answer": "he', 'llo world"}'], "hello world"),
        (['{"answer": "line', '\\', 'nbreak"}'], "line\nbreak"),
        (['{"answer": "quote \\"x\\" end"}'], 'quote "x" end'),
        (['{"citations": [2], "answer": "after the array"}'], "after the array"),
        (['{"answer": "\\u041f\\u043e\\u0440\\u0442"}'], "Порт"),
    ],
)
def test_streaming_extractor_rebuilds_the_answer_field(pieces: list[str], expected: str) -> None:
    streamer = JsonStringFieldStreamer("answer")
    assert "".join(streamer.feed(piece) for piece in pieces) == expected


def test_streaming_extractor_stops_at_the_closing_quote() -> None:
    streamer = JsonStringFieldStreamer("answer")
    streamer.feed('{"answer": "done", "citations": [1, 2]}')
    assert streamer.feed("") == ""
    assert streamer.raw.endswith("]}")


def test_streaming_extractor_ignores_a_split_escape_until_it_is_complete() -> None:
    streamer = JsonStringFieldStreamer("answer")
    assert streamer.feed('{"answer": "a\\u04') == "a"
    assert streamer.feed('1f"') == "П"


def make_client() -> OllamaClient:
    return OllamaClient(OllamaSettings(base_url=BASE_URL, max_attempts=1), sleep=lambda _: None)


@respx.mock
def test_ollama_generator_validates_what_the_model_returns() -> None:
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "content": '{"answer": "Ответ [1] и подделка [5].", "citations": [1, 5]}'
                }
            },
        )
    )
    answer = OllamaGenerator(make_client(), "m").generate("вопрос", CONTEXT)
    assert [citation.marker for citation in answer.citations] == [1]
    assert "[5]" not in answer.text


@respx.mock
def test_a_malformed_payload_becomes_a_refusal_not_an_exception() -> None:
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "not json at all"}})
    )
    answer = OllamaGenerator(make_client(), "m").generate("вопрос", CONTEXT)
    assert answer.refused


@respx.mock
def test_a_transport_failure_becomes_a_refusal() -> None:
    respx.post(f"{BASE_URL}/api/chat").mock(side_effect=httpx.ConnectError("down"))
    answer = OllamaGenerator(make_client(), "m").generate("вопрос", CONTEXT)
    assert answer.refused
    assert "generation failed" in answer.reason


@respx.mock
def test_streamed_text_can_be_retracted_by_the_final_event() -> None:
    """Deltas are provisional; the last event is what the client must believe."""
    body = "\n".join(
        [
            '{"message": {"content": "{\\"answer\\": \\"Выдумка "}, "done": false}',
            '{"message": {"content": "[9].\\", \\"citations\\": [9]}"}, "done": true}',
        ]
    )
    respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(200, text=body))
    events = list(OllamaGenerator(make_client(), "m").stream("вопрос", CONTEXT))
    deltas = "".join(event.text for event in events if isinstance(event, AnswerDelta))
    final = events[-1]
    assert "Выдумка" in deltas
    assert isinstance(final, AnswerComplete)
    assert final.answer.refused


def test_build_generator_rejects_an_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown generation provider"):
        build_generator("gpt", model="m", analyzer=Analyzer(), settings=SETTINGS)
