"""In-memory store behaviour, plus the parts of the pgvector store testable offline."""

from __future__ import annotations

import numpy as np
import pytest

from rag.store import (
    ChunkRecord,
    HnswSettings,
    InMemoryVectorStore,
    StoreError,
    containment_filter,
    to_vector_literal,
)
from tests.conftest import make_chunk


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def store() -> InMemoryVectorStore:
    built = InMemoryVectorStore(dim=3, metric="cosine")
    built.upsert(
        [
            ChunkRecord(
                make_chunk("a", "alpha", doc_id="d1", metadata={"lang": "ru"}), unit(1, 0, 0)
            ),
            ChunkRecord(
                make_chunk("b", "beta", doc_id="d1", metadata={"lang": "en"}), unit(0, 1, 0)
            ),
            ChunkRecord(
                make_chunk("c", "gamma", doc_id="d2", metadata={"lang": "ru"}), unit(1, 1, 0)
            ),
        ]
    )
    return built


def test_cosine_search_orders_by_similarity(store: InMemoryVectorStore) -> None:
    results = store.search(unit(1, 0, 0), k=3)
    assert [item.chunk.chunk_id for item in results] == ["a", "c", "b"]
    assert results[0].score == pytest.approx(1.0, abs=1e-6)
    assert results[1].score == pytest.approx(0.70710678, abs=1e-6)
    assert results[2].score == pytest.approx(0.0, abs=1e-6)
    assert [item.rank for item in results] == [1, 2, 3]


def test_l2_search_orders_by_distance() -> None:
    store = InMemoryVectorStore(dim=2, metric="l2")
    store.upsert(
        [
            ChunkRecord(make_chunk("near", "n"), np.asarray([1.0, 0.0], dtype=np.float32)),
            ChunkRecord(make_chunk("far", "f"), np.asarray([5.0, 5.0], dtype=np.float32)),
        ]
    )
    results = store.search(np.asarray([1.0, 0.1], dtype=np.float32), k=2)
    assert [item.chunk.chunk_id for item in results] == ["near", "far"]
    # Scores are negated distances, so higher is still better.
    assert results[0].score == pytest.approx(-0.1, abs=1e-6)


def test_equality_filter_restricts_candidates(store: InMemoryVectorStore) -> None:
    results = store.search(unit(1, 0, 0), k=5, filters={"lang": "en"})
    assert [item.chunk.chunk_id for item in results] == ["b"]


def test_membership_filter_accepts_several_values(store: InMemoryVectorStore) -> None:
    results = store.search(unit(1, 0, 0), k=5, filters={"lang": ["ru", "en"]})
    assert len(results) == 3


def test_filter_matching_nothing_returns_nothing(store: InMemoryVectorStore) -> None:
    assert store.search(unit(1, 0, 0), k=5, filters={"lang": "de"}) == []


def test_filter_does_not_change_the_relative_order(store: InMemoryVectorStore) -> None:
    unfiltered = [
        item.chunk.chunk_id
        for item in store.search(unit(1, 1, 0), k=5)
        if item.chunk.metadata["lang"] == "ru"
    ]
    filtered = [
        item.chunk.chunk_id for item in store.search(unit(1, 1, 0), k=5, filters={"lang": "ru"})
    ]
    assert filtered == unfiltered


def test_upsert_replaces_a_chunk_in_place(store: InMemoryVectorStore) -> None:
    store.upsert([ChunkRecord(make_chunk("a", "alpha v2", doc_id="d1"), unit(0, 0, 1))])
    assert store.count() == 3
    assert store.search(unit(0, 0, 1), k=1)[0].chunk.text == "alpha v2"


def test_delete_document_removes_all_its_chunks(store: InMemoryVectorStore) -> None:
    assert store.delete_document("d1") == 2
    assert store.count() == 1
    assert [chunk.chunk_id for chunk in store.iter_chunks()] == ["c"]
    assert store.delete_document("d1") == 0


def test_search_after_delete_does_not_return_stale_vectors(store: InMemoryVectorStore) -> None:
    """The cached matrix must be invalidated, or deleted chunks keep scoring."""
    store.search(unit(1, 0, 0), k=3)
    store.delete_document("d1")
    assert [item.chunk.chunk_id for item in store.search(unit(1, 0, 0), k=3)] == ["c"]


def test_wrong_dimension_is_rejected_on_write() -> None:
    store = InMemoryVectorStore(dim=3)
    with pytest.raises(StoreError, match="does not match store dim"):
        store.upsert([ChunkRecord(make_chunk("a", "alpha"), unit(1, 0))])


def test_iteration_preserves_insertion_order(store: InMemoryVectorStore) -> None:
    assert [chunk.chunk_id for chunk in store.iter_chunks()] == ["a", "b", "c"]


def test_clear_empties_the_store(store: InMemoryVectorStore) -> None:
    store.clear()
    assert store.count() == 0
    assert store.search(unit(1, 0, 0), k=3) == []


def test_k_of_zero_returns_nothing(store: InMemoryVectorStore) -> None:
    assert store.search(unit(1, 0, 0), k=0) == []


def test_vector_literal_is_pgvector_syntax() -> None:
    literal = to_vector_literal(np.asarray([0.5, -0.25, 1.0], dtype=np.float32))
    assert literal == "[0.5,-0.25,1]"


def test_pgvector_rejects_multi_value_filters() -> None:
    """Better a loud error than a filter silently applied to only one of its values."""
    assert containment_filter({"lang": "ru"}) == {"lang": "ru"}
    with pytest.raises(StoreError, match="equality filters only"):
        containment_filter({"lang": ["ru", "en"]})


def test_hnsw_defaults_are_the_documented_ones() -> None:
    settings = HnswSettings()
    assert (settings.m, settings.ef_construction, settings.ef_search) == (16, 64, 64)
