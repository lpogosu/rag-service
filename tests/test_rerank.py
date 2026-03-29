"""Reranking: the heuristic scorer's behaviour and the LLM judge's fallback."""

from __future__ import annotations

import httpx
import pytest
import respx

from rag.ollama import OllamaClient, OllamaSettings
from rag.rerank import HeuristicReranker, LlmJudgeReranker, NoOpReranker, build_reranker
from rag.text import Analyzer
from rag.types import FusedChunk
from tests.conftest import make_chunk

BASE_URL = "http://ollama.test:11434"


def candidate(chunk_id: str, text: str, score: float = 0.0) -> FusedChunk:
    return FusedChunk(chunk=make_chunk(chunk_id, text), score=score, ranks={"dense": 1})


def test_noop_reranker_preserves_the_fused_order() -> None:
    candidates = [candidate("a", "alpha"), candidate("b", "beta")]
    assert [item.chunk.chunk_id for item in NoOpReranker().rerank("q", candidates, 2)] == ["a", "b"]


def test_full_term_coverage_beats_partial_coverage() -> None:
    reranker = HeuristicReranker(Analyzer())
    both = reranker.score("срок хранения событий", "Срок хранения событий равен 168 часам.")
    one = reranker.score("срок хранения событий", "Срок действия токена — 24 часа.")
    assert both > one


def test_terms_close_together_beat_terms_scattered_apart() -> None:
    """Coverage alone would score these equally; proximity is what separates them."""
    reranker = HeuristicReranker(Analyzer())
    filler = " ".join(f"слово{index}" for index in range(60))
    together = reranker.score("порт брокера", f"Порт брокера равен 7420. {filler}")
    apart = reranker.score(
        "порт брокера", f"Порт указан ниже. {filler} Брокера настраивают отдельно."
    )
    assert together > apart


def test_a_chunk_without_any_query_term_scores_zero() -> None:
    assert HeuristicReranker(Analyzer()).score("репликация реплик", "Совсем другой текст.") == 0.0


def test_an_empty_query_scores_zero() -> None:
    assert HeuristicReranker(Analyzer()).score("", "любой текст") == 0.0


def test_reranking_reorders_and_truncates() -> None:
    candidates = [
        candidate("noise", "Совершенно посторонний текст про сертификаты."),
        candidate("hit", "Срок хранения событий равен 168 часам."),
        candidate("weak", "Хранение снапшотов занимает 14 дней."),
    ]
    ranked = HeuristicReranker(Analyzer()).rerank("срок хранения событий", candidates, 2)
    assert [item.chunk.chunk_id for item in ranked] == ["hit", "weak"]
    assert [item.ranks["rerank"] for item in ranked] == [1, 2]
    assert ranked[0].ranks["dense"] == 1


def make_client() -> OllamaClient:
    return OllamaClient(OllamaSettings(base_url=BASE_URL, max_attempts=1), sleep=lambda _: None)


@respx.mock
def test_llm_judge_scores_each_candidate_and_sorts_by_score() -> None:
    scores = iter(['{"score": 2}', '{"score": 9}'])
    respx.post(f"{BASE_URL}/api/chat").mock(
        side_effect=lambda _: httpx.Response(200, json={"message": {"content": next(scores)}})
    )
    candidates = [candidate("low", "текст один"), candidate("high", "текст два")]
    ranked = LlmJudgeReranker(make_client(), "m").rerank("вопрос", candidates, 2)
    assert [item.chunk.chunk_id for item in ranked] == ["high", "low"]
    assert ranked[0].score == pytest.approx(0.9)


@respx.mock
def test_a_failing_judge_falls_back_instead_of_taking_the_answer_down() -> None:
    respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(503))
    candidates = [
        candidate("noise", "Посторонний текст."),
        candidate("hit", "Срок хранения событий равен 168 часам."),
    ]
    reranker = LlmJudgeReranker(make_client(), "m", fallback=HeuristicReranker(Analyzer()))
    ranked = reranker.rerank("срок хранения событий", candidates, 2)
    assert [item.chunk.chunk_id for item in ranked] == ["hit", "noise"]


@respx.mock
def test_a_judge_returning_nonsense_also_falls_back() -> None:
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "девять из десяти"}})
    )
    candidates = [candidate("hit", "Срок хранения событий равен 168 часам.")]
    ranked = LlmJudgeReranker(make_client(), "m").rerank("срок хранения", candidates, 1)
    assert ranked[0].chunk.chunk_id == "hit"


def test_reranking_nothing_returns_nothing() -> None:
    assert LlmJudgeReranker(make_client(), "m").rerank("вопрос", [], 5) == []


def test_build_reranker_selects_the_implementation() -> None:
    analyzer = Analyzer()
    assert build_reranker("none", model="m", analyzer=analyzer).name == "none"
    assert build_reranker("heuristic", model="m", analyzer=analyzer).name == "heuristic"
    with pytest.raises(ValueError, match="needs an OllamaClient"):
        build_reranker("ollama", model="m", analyzer=analyzer)
    with pytest.raises(ValueError, match="unknown reranker provider"):
        build_reranker("cohere", model="m", analyzer=analyzer)
