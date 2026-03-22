"""Fusion of dense and lexical result lists.

Two strategies are implemented, and the harness can measure both, because "RRF is
better" is an assertion until someone runs it on their own corpus.

Reciprocal Rank Fusion combines *positions*; weighted fusion combines *scores*. The
argument for positions is that the two scales are not comparable and cannot be made
comparable by rescaling:

* cosine similarity from a normalised embedder lives in a narrow band — a good match
  might be 0.71 and an unrelated chunk 0.58 — while BM25 is unbounded above and starts
  at zero. Min–max normalisation therefore stretches whatever spread the dense list
  happens to have on this query into the full [0, 1] range, and a query where every
  dense candidate is mediocre produces a confident-looking 1.0.
* the normalisation constants depend on the candidate set, so the fused score of a
  chunk changes when an unrelated chunk enters or leaves the list. Ranks are stable
  under that.
* BM25 returns nothing at all when no query term matches. Under score fusion the dense
  side is then normalised against an empty list; under RRF the lexical contribution is
  simply absent and the dense ranking passes through unchanged.

The cost of RRF is that it throws away margin: a dense hit at 0.95 and one at 0.71 are
"rank 1" and "rank 2", and the gap is lost. That is why reranking exists downstream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rag.types import Chunk, FusedChunk, ScoredChunk

DEFAULT_RRF_K = 60


class FusionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FusionParams:
    """Fusion configuration.

    ``k`` damps the top of the reciprocal curve: with k = 60 the difference between
    rank 1 and rank 2 is about 1.6 %, so a single retriever cannot dominate the fused
    list on its own; with k = 1 it very nearly can. 60 is the value from the original
    paper and a reasonable default, not a law.
    """

    method: str = "rrf"
    k: int = DEFAULT_RRF_K
    weights: Mapping[str, float] = field(default_factory=lambda: {"dense": 0.5, "lexical": 0.5})

    def __post_init__(self) -> None:
        if self.method not in {"rrf", "weighted"}:
            raise FusionError(f"unknown fusion method: {self.method!r}")
        if self.k <= 0:
            raise FusionError("k must be positive")
        if any(weight < 0 for weight in self.weights.values()):
            raise FusionError("fusion weights must be non-negative")

    def weight(self, source: str) -> float:
        return self.weights.get(source, 1.0)


def _index_by_chunk(rankings: Mapping[str, Sequence[ScoredChunk]]) -> dict[str, Chunk]:
    chunks: dict[str, Chunk] = {}
    for results in rankings.values():
        for scored in results:
            chunks.setdefault(scored.chunk.chunk_id, scored.chunk)
    return chunks


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[ScoredChunk]],
    params: FusionParams,
    limit: int,
) -> list[FusedChunk]:
    chunks = _index_by_chunk(rankings)
    scores: dict[str, float] = dict.fromkeys(chunks, 0.0)
    ranks: dict[str, dict[str, int]] = {chunk_id: {} for chunk_id in chunks}
    for source, results in rankings.items():
        weight = params.weight(source)
        for position, scored in enumerate(results, start=1):
            chunk_id = scored.chunk.chunk_id
            scores[chunk_id] += weight / (params.k + position)
            ranks[chunk_id][source] = position
    return _top(chunks, scores, ranks, limit)


def weighted_score_fusion(
    rankings: Mapping[str, Sequence[ScoredChunk]],
    params: FusionParams,
    limit: int,
) -> list[FusedChunk]:
    """Min–max normalise each list, then take the weighted sum. Kept for comparison."""
    chunks = _index_by_chunk(rankings)
    scores: dict[str, float] = dict.fromkeys(chunks, 0.0)
    ranks: dict[str, dict[str, int]] = {chunk_id: {} for chunk_id in chunks}
    for source, results in rankings.items():
        if not results:
            continue
        weight = params.weight(source)
        values = [scored.score for scored in results]
        low, high = min(values), max(values)
        spread = high - low
        for position, scored in enumerate(results, start=1):
            chunk_id = scored.chunk.chunk_id
            # A degenerate spread means the retriever cannot separate its own
            # candidates; giving them all 1.0 would let it outvote a retriever that can.
            normalised = 1.0 if spread == 0.0 else (scored.score - low) / spread
            scores[chunk_id] += weight * normalised
            ranks[chunk_id][source] = position
    return _top(chunks, scores, ranks, limit)


def _top(
    chunks: Mapping[str, Chunk],
    scores: Mapping[str, float],
    ranks: Mapping[str, dict[str, int]],
    limit: int,
) -> list[FusedChunk]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        FusedChunk(chunk=chunks[chunk_id], score=score, ranks=dict(ranks[chunk_id]))
        for chunk_id, score in ordered[: max(limit, 0)]
    ]


def fuse(
    rankings: Mapping[str, Sequence[ScoredChunk]],
    params: FusionParams,
    limit: int,
) -> list[FusedChunk]:
    if params.method == "weighted":
        return weighted_score_fusion(rankings, params, limit)
    return reciprocal_rank_fusion(rankings, params, limit)
