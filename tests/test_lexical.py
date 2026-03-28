"""BM25 scoring, checked against arithmetic done by hand.

Every constant in ``test_scores_match_the_formula_computed_by_hand`` is derived in the
comment above it. Asserting "the right document ranks first" would pass with a wrong
``b`` or a sign error in the IDF; asserting the exact score does not.
"""

from __future__ import annotations

import math

import pytest

from rag.lexical import Bm25Index, Bm25Params
from rag.text import Analyzer
from tests.conftest import make_chunk

# Three chunks, no stopwords, no stemming, so term counts are what they look like:
#   c1 "alpha beta"        -> dl = 2
#   c2 "alpha alpha gamma" -> dl = 3
#   c3 "delta"             -> dl = 1
# N = 3, avgdl = 2.0
CHUNKS = [
    make_chunk("c1", "alpha beta"),
    make_chunk("c2", "alpha alpha gamma"),
    make_chunk("c3", "delta"),
]


@pytest.fixture
def index(analyzer: Analyzer) -> Bm25Index:
    return Bm25Index.build(CHUNKS, analyzer=analyzer, params=Bm25Params(k1=1.5, b=0.75))


def test_document_lengths_and_average(index: Bm25Index) -> None:
    assert len(index) == 3
    assert index.average_length == pytest.approx(2.0)


def test_idf_uses_the_smoothed_robertson_formula(index: Bm25Index) -> None:
    # df(alpha) = 2, N = 3  ->  ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6)
    assert index.document_frequency("alpha") == 2
    assert index.idf("alpha") == pytest.approx(math.log(1.6))
    # df(delta) = 1  ->  ln(1 + 2.5 / 1.5)
    assert index.idf("delta") == pytest.approx(math.log(1 + 2.5 / 1.5))


def test_idf_of_an_unknown_term_is_zero(index: Bm25Index) -> None:
    assert index.idf("omega") == 0.0
    assert index.search("omega", 5) == []


def test_idf_stays_positive_for_a_term_in_every_document(analyzer: Analyzer) -> None:
    """Without the ``1 +`` inside the logarithm this would be negative."""
    chunks = [make_chunk(f"c{i}", "alpha") for i in range(5)]
    index = Bm25Index.build(chunks, analyzer=analyzer)
    assert index.idf("alpha") > 0


def test_scores_match_the_formula_computed_by_hand(index: Bm25Index) -> None:
    # idf = ln(1.6) = 0.4700036292
    # c1: tf = 1, dl = 2, norm = 1 - 0.75 + 0.75 * (2 / 2)   = 1.0
    #     score = idf * 1 * 2.5 / (1 + 1.5 * 1.0)             = idf * 1.0
    # c2: tf = 2, dl = 3, norm = 1 - 0.75 + 0.75 * (3 / 2)   = 1.375
    #     score = idf * 2 * 2.5 / (2 + 1.5 * 1.375)           = idf * 5 / 4.0625
    idf = math.log(1.6)
    results = index.search("alpha", 5)
    by_id = {result.chunk.chunk_id: result.score for result in results}
    assert by_id["c1"] == pytest.approx(idf * 1.0)
    assert by_id["c2"] == pytest.approx(idf * 5.0 / 4.0625)
    assert [result.chunk.chunk_id for result in results] == ["c2", "c1"]


def test_length_normalisation_penalises_the_longer_chunk(analyzer: Analyzer) -> None:
    """Same term frequency, different length: the shorter chunk must win."""
    chunks = [
        make_chunk("short", "alpha beta"),
        make_chunk("long", "alpha " + " ".join(f"w{i}" for i in range(40))),
    ]
    index = Bm25Index.build(chunks, analyzer=analyzer)
    ranked = [result.chunk.chunk_id for result in index.search("alpha", 5)]
    assert ranked == ["short", "long"]


def test_b_zero_disables_length_normalisation(analyzer: Analyzer) -> None:
    chunks = [
        make_chunk("short", "alpha beta"),
        make_chunk("long", "alpha " + " ".join(f"w{i}" for i in range(40))),
    ]
    index = Bm25Index.build(chunks, analyzer=analyzer, params=Bm25Params(k1=1.5, b=0.0))
    scores = {result.chunk.chunk_id: result.score for result in index.search("alpha", 5)}
    assert scores["short"] == pytest.approx(scores["long"])


def test_term_frequency_saturates(analyzer: Analyzer) -> None:
    """Doubling tf must add less than the first occurrence did — that is what k1 buys."""
    chunks = [make_chunk("one", "alpha"), make_chunk("many", "alpha " * 8)]
    index = Bm25Index.build(chunks, analyzer=analyzer, params=Bm25Params(k1=1.5, b=0.0))
    scores = {result.chunk.chunk_id: result.score for result in index.search("alpha", 5)}
    assert scores["many"] < 8 * scores["one"]
    assert scores["many"] > scores["one"]


def test_ranking_is_deterministic_for_tied_scores(analyzer: Analyzer) -> None:
    chunks = [make_chunk(f"c{i}", "alpha beta") for i in range(6)]
    index = Bm25Index.build(chunks, analyzer=analyzer)
    first = [result.chunk.chunk_id for result in index.search("alpha", 4)]
    second = [result.chunk.chunk_id for result in index.search("alpha", 4)]
    assert first == second == ["c0", "c1", "c2", "c3"]


def test_stemming_lets_a_query_form_match_a_document_form() -> None:
    chunks = [make_chunk("ru", "Параметр задаёт срок хранения событий в партициях.")]
    stemmed = Bm25Index.build(chunks, analyzer=Analyzer())
    plain = Bm25Index.build(chunks, analyzer=Analyzer(use_stemming=False))
    assert stemmed.search("партиция хранение", 3)
    assert plain.search("партиция хранение", 3) == []


def test_search_respects_metadata_filters(analyzer: Analyzer) -> None:
    chunks = [
        make_chunk("ru", "alpha", doc_id="d1", metadata={"lang": "ru"}),
        make_chunk("en", "alpha", doc_id="d2", metadata={"lang": "en"}),
    ]
    index = Bm25Index.build(chunks, analyzer=analyzer)
    assert [item.chunk.chunk_id for item in index.search("alpha", 5, filters={"lang": "ru"})] == [
        "ru"
    ]
    assert len(index.search("alpha", 5, filters={"lang": ["ru", "en"]})) == 2
    assert index.search("alpha", 5, filters={"lang": "de"}) == []


def test_k_of_zero_returns_nothing(index: Bm25Index) -> None:
    assert index.search("alpha", 0) == []


def test_empty_index_scores_nothing(analyzer: Analyzer) -> None:
    assert Bm25Index(analyzer=analyzer).search("alpha", 5) == []
