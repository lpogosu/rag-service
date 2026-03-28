"""Metric arithmetic, with the expected values derived in comments."""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    GoldSpan,
    answer_term_recall,
    faithfulness,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_chunk_ids,
)
from rag.types import Answer, Chunk, Citation
from tests.conftest import make_chunk

RANKED = ["a", "b", "c", "d", "e"]


def span_chunk(chunk_id: str, doc_id: str, start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="x" * (end - start),
        start=start,
        end=end,
        ordinal=0,
    )


def test_gold_spans_select_every_overlapping_chunk() -> None:
    """Chunker-independent labelling: the gold sentence may straddle a boundary."""
    chunks = [
        span_chunk("c0", "doc", 0, 100),
        span_chunk("c1", "doc", 90, 200),
        span_chunk("c2", "doc", 200, 300),
        span_chunk("other", "elsewhere", 90, 200),
    ]
    gold = GoldSpan(doc_id="doc", start=95, end=120)
    assert relevant_chunk_ids(gold, chunks) == {"c0", "c1"}


def test_a_span_touching_a_boundary_does_not_count_as_overlap() -> None:
    chunks = [span_chunk("c0", "doc", 0, 100), span_chunk("c1", "doc", 100, 200)]
    assert relevant_chunk_ids(GoldSpan("doc", 100, 150), chunks) == {"c1"}


def test_recall_counts_relevant_chunks_inside_the_cut() -> None:
    assert recall_at_k(RANKED, {"c"}, 3) == 1.0
    assert recall_at_k(RANKED, {"c"}, 2) == 0.0
    assert recall_at_k(RANKED, {"a", "d"}, 3) == 0.5
    assert recall_at_k(RANKED, set(), 5) == 0.0


def test_precision_is_the_share_of_the_cut_that_is_relevant() -> None:
    assert precision_at_k(RANKED, {"a", "c"}, 5) == pytest.approx(0.4)
    assert precision_at_k(RANKED, {"a"}, 1) == 1.0
    assert precision_at_k(RANKED, {"a"}, 0) == 0.0


def test_reciprocal_rank_uses_the_first_hit_only() -> None:
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"c", "e"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"zzz"}) == 0.0


def test_ndcg_discounts_by_position() -> None:
    # one relevant chunk at rank 3: DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1
    assert ndcg_at_k(RANKED, {"c"}, 5) == pytest.approx(0.5)
    assert ndcg_at_k(RANKED, {"a"}, 5) == pytest.approx(1.0)
    # two relevant at ranks 1 and 3: DCG = 1 + 0.5, IDCG = 1 + 1/log2(3)
    assert ndcg_at_k(RANKED, {"a", "c"}, 5) == pytest.approx(1.5 / (1 + 1 / math.log2(3)))


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k(RANKED, {"zzz"}, 5) == 0.0


def answer_with(text: str, chunk_ids: list[str]) -> Answer:
    return Answer(
        text=text,
        citations=tuple(
            Citation(marker=index, chunk_id=chunk_id, doc_id="doc", quote="")
            for index, chunk_id in enumerate(chunk_ids, start=1)
        ),
        refused=False,
    )


def test_faithfulness_is_one_when_every_word_comes_from_a_cited_chunk() -> None:
    chunk = make_chunk("c1", "Срок хранения событий равен 168 часам.")
    report = faithfulness(answer_with("Срок хранения событий равен 168 часам", ["c1"]), [chunk])
    assert report.supported == pytest.approx(1.0)
    assert report.unsupported_terms == ()
    assert report.citations_valid


def test_faithfulness_reports_terms_absent_from_the_sources() -> None:
    chunk = make_chunk("c1", "Срок хранения событий равен 168 часам.")
    report = faithfulness(answer_with("Срок хранения равен 336 часам", ["c1"]), [chunk])
    assert report.supported < 1.0
    assert "336" in report.unsupported_terms


def test_an_unretrieved_citation_is_flagged() -> None:
    """The structural guarantee is asserted, not assumed."""
    chunk = make_chunk("c1", "текст")
    report = faithfulness(answer_with("ответ", ["ghost"]), [chunk])
    assert not report.citations_valid


def test_a_refusal_is_perfectly_faithful() -> None:
    refusal = Answer(text="нет ответа", citations=(), refused=True)
    assert faithfulness(refusal, []).supported == 1.0


def test_answer_term_recall_measures_overlap_with_the_gold_answer() -> None:
    assert answer_term_recall("Порт 7420 используется брокером", "7420") == 1.0
    assert answer_term_recall("совсем другое", "7420") == 0.0
    # The gold answer analyses to {debian, 12, rhel}: "и" is a stopword and the bare
    # "9" is below the two-character floor. Two of the three appear in the answer.
    assert answer_term_recall("Debian 12", "Debian 12 и RHEL 9") == pytest.approx(2 / 3)


def test_mean_of_nothing_is_zero() -> None:
    assert mean([]) == 0.0
    assert mean([1.0, 2.0]) == 1.5
