"""Embedders: determinism offline, and correct HTTP behaviour against a mocked Ollama."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
import respx

from rag.embedding import (
    DimensionMismatchError,
    EmbeddingError,
    HashingEmbedder,
    OllamaEmbedder,
    build_embedder,
    l2_normalize,
)
from rag.ollama import OllamaClient, OllamaSettings

BASE_URL = "http://ollama.test:11434"


def test_hashing_embedder_is_deterministic_across_instances() -> None:
    """The property the whole offline evaluation rests on."""
    first = HashingEmbedder(dim=64, seed=7).embed_query("параметр retention.hours")
    second = HashingEmbedder(dim=64, seed=7).embed_query("параметр retention.hours")
    assert np.array_equal(first, second)


def test_a_different_seed_gives_a_different_projection() -> None:
    first = HashingEmbedder(dim=64, seed=1).embed_query("alpha")
    second = HashingEmbedder(dim=64, seed=2).embed_query("alpha")
    assert not np.array_equal(first, second)


def test_vectors_are_unit_length() -> None:
    matrix = HashingEmbedder(dim=64).embed_documents(["alpha beta", "совсем другой текст"])
    norms = np.linalg.norm(matrix, axis=1)
    assert norms == pytest.approx(np.ones(2), abs=1e-6)


def test_shared_vocabulary_scores_higher_than_unrelated_text() -> None:
    embedder = HashingEmbedder(dim=512)
    query = embedder.embed_query("срок хранения событий")
    related = embedder.embed_query("параметр задаёт срок хранения событий в брокере")
    unrelated = embedder.embed_query("клиентский сертификат содержит namespace в поле OU")
    assert float(query @ related) > float(query @ unrelated)


def test_character_ngrams_survive_a_morphological_change() -> None:
    """Stemming handles most inflection; trigrams catch what it misses."""
    embedder = HashingEmbedder(dim=512)
    query = embedder.embed_query("репликация")
    inflected = embedder.embed_query("репликации")
    other = embedder.embed_query("мониторинг")
    assert float(query @ inflected) > float(query @ other)


def test_empty_input_returns_an_empty_matrix_of_the_right_width() -> None:
    matrix = HashingEmbedder(dim=32).embed_documents([])
    assert matrix.shape == (0, 32)


def test_text_without_terms_does_not_produce_nan() -> None:
    """A punctuation-only chunk must stay in the index with zero similarity, not NaN."""
    matrix = HashingEmbedder(dim=16).embed_documents(["---", "alpha"])
    assert not np.isnan(matrix).any()


def test_l2_normalize_leaves_a_zero_row_at_zero() -> None:
    normalised = l2_normalize(np.zeros((1, 4), dtype=np.float32))
    assert np.array_equal(normalised, np.zeros((1, 4), dtype=np.float32))


@pytest.mark.parametrize("bad", [0, -5])
def test_invalid_dimension_is_rejected(bad: int) -> None:
    with pytest.raises(EmbeddingError, match="dim must be positive"):
        HashingEmbedder(dim=bad)


def make_client() -> OllamaClient:
    return OllamaClient(OllamaSettings(base_url=BASE_URL, max_attempts=1), sleep=lambda _: None)


@respx.mock
def test_ollama_embedder_batches_requests() -> None:
    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.read())["input"]
        batch_sizes.append(len(inputs))
        return httpx.Response(200, json={"embeddings": [[0.0, 1.0]] * len(inputs)})

    respx.post(f"{BASE_URL}/api/embed").mock(side_effect=handler)
    embedder = OllamaEmbedder(make_client(), model="m", dim=2, batch_size=2)
    matrix = embedder.embed_documents([f"text-{index}" for index in range(5)])
    assert matrix.shape == (5, 2)
    assert batch_sizes == [2, 2, 1]


@respx.mock
def test_ollama_embedder_validates_the_returned_width() -> None:
    respx.post(f"{BASE_URL}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})
    )
    embedder = OllamaEmbedder(make_client(), model="m", dim=768)
    with pytest.raises(DimensionMismatchError) as error:
        embedder.embed_documents(["alpha"])
    assert error.value.expected == 768
    assert error.value.actual == 3


@respx.mock
def test_ollama_embedder_normalises_what_the_model_returns() -> None:
    respx.post(f"{BASE_URL}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[3.0, 4.0]]})
    )
    vector = OllamaEmbedder(make_client(), model="m", dim=2).embed_query("alpha")
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)
    assert vector[0] == pytest.approx(0.6, abs=1e-6)


@respx.mock
def test_a_failing_provider_raises_an_embedding_error() -> None:
    respx.post(f"{BASE_URL}/api/embed").mock(return_value=httpx.Response(500, text="boom"))
    embedder = OllamaEmbedder(make_client(), model="m", dim=2)
    with pytest.raises(EmbeddingError, match="offset 0"):
        embedder.embed_documents(["alpha"])


def test_build_embedder_requires_a_client_for_ollama() -> None:
    with pytest.raises(EmbeddingError, match="needs an OllamaClient"):
        build_embedder("ollama", model="m", dim=8, batch_size=4, seed=1)


def test_build_embedder_rejects_an_unknown_provider() -> None:
    with pytest.raises(EmbeddingError, match="unknown embedding provider"):
        build_embedder("openai", model="m", dim=8, batch_size=4, seed=1)
