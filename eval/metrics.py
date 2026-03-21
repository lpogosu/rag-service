"""Retrieval and answer metrics.

Relevance labels are character spans in a document, not chunk ids. That is the only way
to compare chunking strategies honestly: a label attached to "chunk 7" means something
different at 400 characters per chunk than at 1200, whereas "the answer is in document
``auth`` between offsets 812 and 1004" means the same thing for every strategy. The
harness converts the span into the set of chunk ids that overlap it, for whatever
chunking the run under test happens to use.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from rag.text import Analyzer
from rag.types import Answer, Chunk


@dataclass(frozen=True, slots=True)
class GoldSpan:
    doc_id: str
    start: int
    end: int

    def overlaps(self, chunk: Chunk) -> bool:
        return chunk.doc_id == self.doc_id and chunk.start < self.end and self.start < chunk.end


def relevant_chunk_ids(gold: GoldSpan, chunks: Sequence[Chunk]) -> set[str]:
    """Every chunk that contains any part of the gold sentence."""
    return {chunk.chunk_id for chunk in chunks if gold.overlaps(chunk)}


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of relevant chunks that made it into the top k.

    With one gold sentence per question and chunks large enough to contain it whole,
    this is usually 1.0 or 0.0 — it answers "did the context contain the answer at all",
    which is the question that decides whether generation can possibly succeed.
    """
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the top k that is relevant.

    Bounded above by ``len(relevant) / k``, and ``len(relevant)`` is one or two here, so
    the absolute value is small by construction. It is still worth reporting: it is the
    context dilution figure — how much of the model's context window is noise.
    """
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for position, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG. Rewards putting the answer first, not merely including it."""
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(ranked[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass(frozen=True, slots=True)
class FaithfulnessReport:
    """Answer-level grounding, measured lexically.

    ``supported`` is the share of the answer's content terms that also occur in the
    chunks the answer cites. It is a proxy, and its limits are worth stating: a
    paraphrase that keeps the meaning and changes the words scores low, and a fluent
    lie assembled from words that do appear in the context scores high. What it does
    catch reliably is the common failure — a model that leaves the context entirely and
    answers from its weights, whose vocabulary then has nothing to do with the sources.

    ``citations_valid`` is structural rather than statistical: it is 1.0 unless
    :func:`rag.generate.enforce_citations` has a bug, and the harness reports it so that
    such a bug cannot pass silently.
    """

    supported: float
    unsupported_terms: tuple[str, ...]
    citations_valid: bool


def faithfulness(
    answer: Answer,
    retrieved_chunks: Sequence[Chunk],
    analyzer: Analyzer | None = None,
) -> FaithfulnessReport:
    analyzer = analyzer or Analyzer()
    retrieved_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    citations_valid = all(chunk_id in retrieved_by_id for chunk_id in answer.cited_chunk_ids)
    if answer.refused:
        return FaithfulnessReport(
            supported=1.0, unsupported_terms=(), citations_valid=citations_valid
        )

    cited_terms: set[str] = set()
    for chunk_id in answer.cited_chunk_ids:
        chunk = retrieved_by_id.get(chunk_id)
        if chunk is not None:
            cited_terms.update(analyzer.analyze(chunk.text))
    answer_terms = analyzer.analyze(answer.text)
    if not answer_terms:
        return FaithfulnessReport(
            supported=0.0, unsupported_terms=(), citations_valid=citations_valid
        )
    unsupported = tuple(dict.fromkeys(term for term in answer_terms if term not in cited_terms))
    supported = 1.0 - len(unsupported) / len(set(answer_terms))
    return FaithfulnessReport(
        supported=supported,
        unsupported_terms=unsupported,
        citations_valid=citations_valid,
    )


def answer_term_recall(
    answer_text: str, gold_answer: str, analyzer: Analyzer | None = None
) -> float:
    """Share of the gold answer's content terms that appear in the produced answer.

    Not a correctness measure — it cannot tell "порт 7420" from "не порт 7420". It is a
    cheap regression signal: when it drops between two configurations, something in
    retrieval stopped surfacing the sentence that carries the answer.
    """
    analyzer = analyzer or Analyzer()
    gold_terms = set(analyzer.analyze(gold_answer))
    if not gold_terms:
        return 0.0
    produced = set(analyzer.analyze(answer_text))
    return len(gold_terms & produced) / len(gold_terms)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
