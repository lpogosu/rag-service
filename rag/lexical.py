"""BM25, implemented directly.

A dense retriever cannot find what it has never seen: exact identifiers, error codes,
flag names, version strings and rare proper nouns are precisely the tokens an embedding
model compresses away. BM25 finds them and costs a dictionary lookup.

There is no external search engine here on purpose. BM25 is one formula and an inverted
index; adding Elasticsearch to the deployment for it would be more moving parts than
the whole rest of this service.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from rag.filtering import Filters, matches_filters
from rag.text import Analyzer
from rag.types import Chunk, ScoredChunk


@dataclass(frozen=True, slots=True)
class Bm25Params:
    """Okapi BM25 parameters.

    ``k1`` controls how fast term frequency saturates; ``b`` how strongly long documents
    are penalised. The defaults are the usual 1.2–2.0 / 0.75 compromise. With chunks of
    a few hundred characters the length normalisation matters less than it does over
    whole documents, which is one more reason chunk size and BM25 tuning interact.
    """

    k1: float = 1.5
    b: float = 0.75


class Bm25Index:
    """In-memory inverted index over chunks.

    The index is rebuilt from the vector store rather than persisted. At the scale this
    service targets (up to a few hundred thousand chunks) a rebuild is seconds and
    removes a whole class of "the two indexes disagree" bugs. Past that scale the right
    move is Postgres full-text search or a dedicated engine, not a bigger dict.
    """

    def __init__(self, analyzer: Analyzer | None = None, params: Bm25Params | None = None) -> None:
        self.analyzer = analyzer or Analyzer()
        self.params = params or Bm25Params()
        self._chunks: list[Chunk] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._average_length = 0.0

    @classmethod
    def build(
        cls,
        chunks: Iterable[Chunk],
        analyzer: Analyzer | None = None,
        params: Bm25Params | None = None,
    ) -> Bm25Index:
        index = cls(analyzer=analyzer, params=params)
        index.add_all(chunks)
        return index

    def add_all(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            terms = self.analyzer.analyze(chunk.embedding_text())
            position = len(self._chunks)
            self._chunks.append(chunk)
            self._lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self._postings[term].append((position, frequency))
        total = sum(self._lengths)
        self._average_length = total / len(self._lengths) if self._lengths else 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def average_length(self) -> float:
        return self._average_length

    def document_frequency(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    def idf(self, term: str) -> float:
        """Robertson–Sparck Jones inverse document frequency with the +0.5 smoothing.

        The ``1 +`` inside the logarithm is what keeps the value positive for terms that
        appear in more than half the corpus. Without it a common term contributes a
        negative score and a document is punished for containing a word the user asked
        for, which is indefensible however defensible the probabilistic derivation is.
        """
        total = len(self._chunks)
        df = self.document_frequency(term)
        if total == 0 or df == 0:
            return 0.0
        return math.log(1.0 + (total - df + 0.5) / (df + 0.5))

    def score_terms(self, terms: list[str]) -> dict[int, float]:
        """Accumulate BM25 over the postings of the query terms only."""
        if not self._chunks or self._average_length == 0.0:
            return {}
        k1 = self.params.k1
        b = self.params.b
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for position, frequency in postings:
                norm = 1.0 - b + b * (self._lengths[position] / self._average_length)
                scores[position] += idf * frequency * (k1 + 1.0) / (frequency + k1 * norm)
        return scores

    def search(self, query: str, k: int, *, filters: Filters | None = None) -> list[ScoredChunk]:
        if k <= 0:
            return []
        terms = self.analyzer.analyze(query)
        scores = self.score_terms(terms)
        if not scores:
            return []
        candidates = (
            scores.items()
            if filters is None
            else [
                (position, score)
                for position, score in scores.items()
                if matches_filters(self._chunks[position].metadata, filters)
            ]
        )
        # Ties are broken by index position so that the ranking is deterministic; two
        # chunks with identical BM25 scores must not swap places between runs, or the
        # evaluation numbers stop being reproducible.
        top = heapq.nsmallest(k, candidates, key=lambda item: (-item[1], item[0]))
        return [
            ScoredChunk(chunk=self._chunks[position], score=score, source="lexical", rank=rank)
            for rank, (position, score) in enumerate(top, start=1)
        ]
