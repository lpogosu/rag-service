"""Reranking: a second, more expensive look at a short candidate list.

Fusion produces an ordering from two retrievers that never saw the query and the chunk
together — a bi-encoder compares two independent vectors, BM25 compares term
statistics. A reranker scores the pair jointly, which is the only stage in the pipeline
that can notice that a chunk contains all the query's words and still answers a
different question.

The joint scorer here is an instruction model over Ollama rather than a dedicated
cross-encoder, because Ollama serves generation and embeddings but not sequence
classification. It is pointwise and sequential: N candidates cost N round trips, which
is why ``top_n`` is small and why the README argues about when this stage is worth its
latency at all.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from rag.ollama import OllamaClient, OllamaError
from rag.text import Analyzer
from rag.types import FusedChunk

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 10}},
    "required": ["score"],
}

_SYSTEM_PROMPT = (
    "Ты оцениваешь релевантность фрагмента документации вопросу пользователя. "
    "Верни JSON {\"score\": N}, где N — целое от 0 до 10. "
    "10 — фрагмент прямо содержит ответ, 5 — относится к теме, но ответа нет, "
    "0 — фрагмент про другое. Не объясняй."
)


class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: Sequence[FusedChunk], top_k: int
    ) -> list[FusedChunk]: ...


class NoOpReranker:
    """Passes the fused ordering through. The honest baseline for the benchmark."""

    name = "none"

    def rerank(self, query: str, candidates: Sequence[FusedChunk], top_k: int) -> list[FusedChunk]:
        del query
        return list(candidates[:top_k])


class HeuristicReranker:
    """Query-term coverage, proximity and density. No model, no network.

    It measures three things a bi-encoder is bad at: whether *every* query term is
    present (coverage), whether they occur close together rather than scattered across
    unrelated paragraphs (proximity), and whether the chunk is mostly about them or
    mentions them once in passing (density).
    """

    name = "heuristic"

    def __init__(
        self,
        analyzer: Analyzer | None = None,
        *,
        coverage_weight: float = 0.6,
        proximity_weight: float = 0.25,
        density_weight: float = 0.15,
    ) -> None:
        self.analyzer = analyzer or Analyzer()
        self.coverage_weight = coverage_weight
        self.proximity_weight = proximity_weight
        self.density_weight = density_weight

    def score(self, query: str, text: str) -> float:
        query_terms = list(dict.fromkeys(self.analyzer.analyze(query)))
        if not query_terms:
            return 0.0
        chunk_terms = self.analyzer.analyze(text)
        if not chunk_terms:
            return 0.0
        positions: dict[str, list[int]] = {term: [] for term in query_terms}
        for index, term in enumerate(chunk_terms):
            if term in positions:
                positions[term].append(index)
        matched = [term for term in query_terms if positions[term]]
        coverage = len(matched) / len(query_terms)
        if not matched:
            return 0.0
        occurrences = sum(len(positions[term]) for term in matched)
        density = min(1.0, occurrences * 10.0 / len(chunk_terms))
        window = _minimal_window(positions, matched)
        proximity = len(matched) / window if window else 0.0
        return (
            self.coverage_weight * coverage
            + self.proximity_weight * proximity
            + self.density_weight * density
        )

    def rerank(self, query: str, candidates: Sequence[FusedChunk], top_k: int) -> list[FusedChunk]:
        scored = [
            FusedChunk(
                chunk=candidate.chunk,
                score=self.score(query, candidate.chunk.embedding_text()),
                ranks=dict(candidate.ranks),
            )
            for candidate in candidates
        ]
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return _with_rerank_ranks(scored[:top_k])


def _minimal_window(positions: dict[str, list[int]], matched: list[str]) -> int:
    """Smallest number of tokens containing one occurrence of every matched term."""
    pointers = dict.fromkeys(matched, 0)
    best = 0
    while True:
        current = [(positions[term][pointers[term]], term) for term in matched]
        low, low_term = min(current)
        high = max(index for index, _ in current)
        width = high - low + 1
        if best == 0 or width < best:
            best = width
        pointers[low_term] += 1
        if pointers[low_term] >= len(positions[low_term]):
            return best


class LlmJudgeReranker:
    """Joint query-document scoring by an instruction model, with a fallback.

    The fallback is not decoration. A reranker that raises takes the whole answer down,
    and a rerank failure is the one failure in this pipeline that degrades gracefully:
    the fused ordering is already a usable ranking.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        *,
        fallback: Reranker | None = None,
        max_chunk_chars: int = 1200,
    ) -> None:
        self.name = f"llm:{model}"
        self.model = model
        self.max_chunk_chars = max_chunk_chars
        self._client = client
        self._fallback = fallback or HeuristicReranker()

    def rerank(self, query: str, candidates: Sequence[FusedChunk], top_k: int) -> list[FusedChunk]:
        if not candidates:
            return []
        scored: list[FusedChunk] = []
        for candidate in candidates:
            try:
                score = self._score_one(query, candidate.chunk.embedding_text())
            except (OllamaError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                return self._fallback.rerank(query, candidates, top_k)
            scored.append(
                FusedChunk(chunk=candidate.chunk, score=score, ranks=dict(candidate.ranks))
            )
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return _with_rerank_ranks(scored[:top_k])

    def _score_one(self, query: str, text: str) -> float:
        content = self._client.chat(
            self.model,
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Вопрос: {query}\n\nФрагмент:\n{text[: self.max_chunk_chars]}",
                },
            ],
            response_format=_SCORE_SCHEMA,
            options={"temperature": 0.0},
        )
        payload = json.loads(content)
        return float(payload["score"]) / 10.0


def _with_rerank_ranks(items: list[FusedChunk]) -> list[FusedChunk]:
    return [
        FusedChunk(chunk=item.chunk, score=item.score, ranks={**item.ranks, "rerank": position})
        for position, item in enumerate(items, start=1)
    ]


def build_reranker(
    provider: str,
    *,
    model: str,
    analyzer: Analyzer,
    client: OllamaClient | None = None,
) -> Reranker:
    if provider == "none":
        return NoOpReranker()
    if provider == "heuristic":
        return HeuristicReranker(analyzer)
    if provider == "ollama":
        if client is None:
            raise ValueError("the ollama reranker needs an OllamaClient")
        return LlmJudgeReranker(client, model, fallback=HeuristicReranker(analyzer))
    raise ValueError(f"unknown reranker provider: {provider!r}")
