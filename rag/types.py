"""Value objects shared by every stage of the pipeline.

Chunks carry byte-exact offsets into their source document. That single decision is
what makes the whole repository auditable: any chunking strategy can be checked for
lost or duplicated text, and evaluation gold spans can be expressed once at the
document level instead of being re-annotated for every chunk size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]


@dataclass(frozen=True, slots=True)
class Document:
    """A source document as ingested, before chunking."""

    doc_id: str
    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A slice of a document.

    ``text`` is always exactly ``document.text[start:end]``. Chunkers may not trim,
    normalise or re-join the slice, because the reconstruction invariant in
    ``tests/test_chunking.py`` depends on it. Presentation-level cleanup (heading
    prefixes, whitespace collapsing) happens later, in :meth:`embedding_text`.
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    ordinal: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"empty span for chunk {self.chunk_id}: [{self.start}, {self.end})")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                f"chunk {self.chunk_id} text length {len(self.text)} "
                f"does not match span [{self.start}, {self.end})"
            )

    @property
    def heading(self) -> str:
        value = self.metadata.get("heading", "")
        return value if isinstance(value, str) else ""

    def embedding_text(self) -> str:
        """Text handed to the embedder.

        The heading path is prepended so that a chunk taken from the middle of a
        section still knows what it is about. Without it, "по умолчанию 30 секунд"
        is a sentence about nothing.
        """
        body = " ".join(self.text.split())
        heading = self.heading
        return f"{heading}\n{body}" if heading else body


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with the score and provenance of the stage that produced it."""

    chunk: Chunk
    score: float
    source: str
    rank: int


@dataclass(frozen=True, slots=True)
class FusedChunk:
    """A chunk after rank fusion, keeping the per-retriever ranks that produced it."""

    chunk: Chunk
    score: float
    ranks: dict[str, int]

    def explain(self) -> str:
        parts = [f"{source}#{rank}" for source, rank in sorted(self.ranks.items())]
        return f"{self.chunk.chunk_id} rrf={self.score:.5f} " + " ".join(parts)


@dataclass(frozen=True, slots=True)
class Citation:
    """A single citation resolved back to the chunk that justifies it."""

    marker: int
    chunk_id: str
    doc_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class Answer:
    """Generation result.

    ``refused`` is not an error state: refusing when retrieval came back empty is the
    designed behaviour, and the harness counts it separately from a wrong answer.
    """

    text: str
    citations: tuple[Citation, ...]
    refused: bool
    reason: str = ""

    @property
    def cited_chunk_ids(self) -> tuple[str, ...]:
        return tuple(citation.chunk_id for citation in self.citations)


@dataclass(frozen=True, slots=True)
class StageTiming:
    stage: str
    millis: float


@dataclass(frozen=True, slots=True)
class QueryResult:
    query: str
    answer: Answer
    retrieved: tuple[FusedChunk, ...]
    timings: tuple[StageTiming, ...]

    def total_millis(self) -> float:
        return sum(timing.millis for timing in self.timings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer.text,
            "refused": self.answer.refused,
            "reason": self.answer.reason,
            "citations": [
                {
                    "marker": citation.marker,
                    "chunk_id": citation.chunk_id,
                    "doc_id": citation.doc_id,
                    "quote": citation.quote,
                }
                for citation in self.answer.citations
            ],
            "retrieved": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "doc_id": item.chunk.doc_id,
                    "score": round(item.score, 6),
                    "ranks": item.ranks,
                }
                for item in self.retrieved
            ],
            "timings_ms": {timing.stage: round(timing.millis, 2) for timing in self.timings},
        }
