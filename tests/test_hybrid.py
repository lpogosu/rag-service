"""Rank fusion.

The RRF scores are written out by hand in the comments, because the whole argument for
RRF over score fusion is about *how* the numbers combine, and a test that only checks
the final order would pass with the wrong constant in the denominator.
"""

from __future__ import annotations

import pytest

from rag.hybrid import DEFAULT_RRF_K, FusionError, FusionParams, fuse, reciprocal_rank_fusion
from rag.types import ScoredChunk
from tests.conftest import make_chunk


def ranked(
    source: str, chunk_ids: list[str], scores: list[float] | None = None
) -> list[ScoredChunk]:
    values = scores or [1.0 / (position + 1) for position in range(len(chunk_ids))]
    return [
        ScoredChunk(
            chunk=make_chunk(chunk_id, f"text of {chunk_id}"),
            score=value,
            source=source,
            rank=position,
        )
        for position, (chunk_id, value) in enumerate(zip(chunk_ids, values, strict=True), start=1)
    ]


def test_rrf_scores_are_the_sum_of_weighted_reciprocal_ranks() -> None:
    # dense:   a(1) b(2) c(3)      lexical: c(1) a(2) d(3)      k = 60, weights 0.5
    # a: 0.5/61 + 0.5/62 = 0.0161637...
    # c: 0.5/63 + 0.5/61 = 0.0161335...
    # b: 0.5/62          = 0.0080645...
    # d: 0.5/63          = 0.0079365...
    params = FusionParams(method="rrf", k=60, weights={"dense": 0.5, "lexical": 0.5})
    fused = reciprocal_rank_fusion(
        {"dense": ranked("dense", ["a", "b", "c"]), "lexical": ranked("lexical", ["c", "a", "d"])},
        params,
        limit=10,
    )
    scores = {item.chunk.chunk_id: item.score for item in fused}
    assert scores["a"] == pytest.approx(0.5 / 61 + 0.5 / 62)
    assert scores["c"] == pytest.approx(0.5 / 63 + 0.5 / 61)
    assert scores["b"] == pytest.approx(0.5 / 62)
    assert scores["d"] == pytest.approx(0.5 / 63)
    assert [item.chunk.chunk_id for item in fused] == ["a", "c", "b", "d"]


def test_a_chunk_found_by_both_retrievers_outranks_a_chunk_found_by_one() -> None:
    """The property RRF exists for: agreement beats a single confident vote."""
    params = FusionParams()
    fused = fuse(
        {
            "dense": ranked("dense", ["solo", "shared"]),
            "lexical": ranked("lexical", ["shared", "other"]),
        },
        params,
        limit=5,
    )
    assert fused[0].chunk.chunk_id == "shared"


def test_fusion_records_the_rank_each_retriever_gave() -> None:
    fused = fuse(
        {"dense": ranked("dense", ["a", "b"]), "lexical": ranked("lexical", ["b"])},
        FusionParams(),
        limit=5,
    )
    by_id = {item.chunk.chunk_id: item.ranks for item in fused}
    assert by_id["a"] == {"dense": 1}
    assert by_id["b"] == {"dense": 2, "lexical": 1}
    assert "dense#2" in next(item for item in fused if item.chunk.chunk_id == "b").explain()


def test_weights_decide_which_retriever_wins_a_disagreement() -> None:
    rankings = {
        "dense": ranked("dense", ["d1", "shared"]),
        "lexical": ranked("lexical", ["l1", "shared"]),
    }
    dense_heavy = [
        item.chunk.chunk_id
        for item in fuse(rankings, FusionParams(weights={"dense": 0.9, "lexical": 0.1}), limit=5)
    ]
    lexical_heavy = [
        item.chunk.chunk_id
        for item in fuse(rankings, FusionParams(weights={"dense": 0.1, "lexical": 0.9}), limit=5)
    ]
    assert dense_heavy.index("d1") < dense_heavy.index("l1")
    assert lexical_heavy.index("l1") < lexical_heavy.index("d1")


def test_k_sets_how_much_rank_one_is_worth() -> None:
    """The ratio between consecutive ranks is (k + 2) / (k + 1) — 1.6 % at k = 60."""
    single = {"dense": ranked("dense", ["first", "second"])}
    damped = fuse(single, FusionParams(k=DEFAULT_RRF_K), limit=2)
    sharp = fuse(single, FusionParams(k=1), limit=2)
    assert damped[0].score / damped[1].score == pytest.approx(62 / 61)
    assert sharp[0].score / sharp[1].score == pytest.approx(3 / 2)


def test_damping_decides_whether_agreement_beats_a_single_top_hit() -> None:
    """Concretely why k = 60 is not an arbitrary constant.

    One retriever puts "top" first and the other has never heard of it; both put "both"
    fifth. At k = 60 the two fifth places outweigh the single first place, at k = 1 they
    do not.
    """
    rankings = {
        "dense": ranked("dense", ["top", "a", "b", "c", "both"]),
        "lexical": ranked("lexical", ["x", "y", "z", "w", "both"]),
    }
    assert fuse(rankings, FusionParams(k=DEFAULT_RRF_K), limit=1)[0].chunk.chunk_id == "both"
    assert fuse(rankings, FusionParams(k=1), limit=1)[0].chunk.chunk_id == "top"


def test_an_empty_lexical_list_leaves_the_dense_order_untouched() -> None:
    """BM25 returns nothing when no query term matches; that must not reorder anything."""
    dense = ranked("dense", ["a", "b", "c"])
    fused = fuse({"dense": dense, "lexical": []}, FusionParams(), limit=5)
    assert [item.chunk.chunk_id for item in fused] == ["a", "b", "c"]


def test_min_max_fusion_gives_a_flat_list_full_marks() -> None:
    """The failure mode RRF avoids: a retriever with no spread still votes at full weight."""
    flat = ranked("dense", ["a", "b"], scores=[0.5, 0.5])
    confident = ranked("lexical", ["c", "d"], scores=[9.0, 0.1])
    fused = fuse(
        {"dense": flat, "lexical": confident},
        FusionParams(method="weighted", weights={"dense": 0.5, "lexical": 0.5}),
        limit=4,
    )
    scores = {item.chunk.chunk_id: item.score for item in fused}
    assert scores["a"] == pytest.approx(0.5)
    assert scores["b"] == pytest.approx(0.5)
    assert scores["c"] == pytest.approx(0.5)
    assert scores["d"] == pytest.approx(0.0)


def test_limit_truncates_the_result() -> None:
    fused = fuse({"dense": ranked("dense", list("abcdef"))}, FusionParams(), limit=2)
    assert len(fused) == 2


def test_ties_break_deterministically() -> None:
    rankings = {"dense": ranked("dense", ["b", "a"]), "lexical": ranked("lexical", ["a", "b"])}
    first = [item.chunk.chunk_id for item in fuse(rankings, FusionParams(), limit=2)]
    second = [item.chunk.chunk_id for item in fuse(rankings, FusionParams(), limit=2)]
    assert first == second == ["a", "b"]


@pytest.mark.parametrize(
    "kwargs",
    [{"method": "borda"}, {"k": 0}, {"weights": {"dense": -1.0}}],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(FusionError):
        FusionParams(**kwargs)  # type: ignore[arg-type]
