"""Embedders behind one protocol.

Two implementations ship: a real one that talks to Ollama, and a deterministic hashing
embedder. The hashing embedder is not a toy — it is what makes the test suite and the
evaluation harness run in CI with no model, no GPU and no network, and it produces the
same vectors on every machine and every Python version.

What it is not is *good*. It is a signed random projection of a bag of stemmed words and
character trigrams, so its notion of similarity is lexical overlap with noise. Numbers
measured with it are a floor, not a forecast of production quality; the README says so
next to the table.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from rag.ollama import OllamaClient, OllamaError
from rag.text import fold, stem, tokenize

Vector = npt.NDArray[np.float32]


class EmbeddingError(RuntimeError):
    pass


class DimensionMismatchError(EmbeddingError):
    """The provider returned vectors of a width the store was not built for.

    Worth its own class because it is the failure that silently corrupts an index:
    without the check, mixed-width vectors either crash deep inside pgvector or, in the
    in-memory store, get broadcast into nonsense.
    """

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"expected {expected}-dimensional vectors, provider returned {actual}")
        self.expected = expected
        self.actual = actual


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> Vector: ...

    def embed_query(self, text: str) -> Vector: ...


def l2_normalize(matrix: Vector) -> Vector:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Zero vectors happen for chunks that are pure punctuation; leaving the norm at 1
    # keeps them in the index with zero similarity to everything instead of producing
    # NaNs that poison every later comparison.
    norms[norms == 0.0] = 1.0
    return cast(Vector, (matrix / norms).astype(np.float32))


def _stable_hash(token: str, salt: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, salt=salt.to_bytes(8, "little"))
    return int.from_bytes(digest.digest(), "little")


class HashingEmbedder:
    """Deterministic offline embedder.

    Features are stemmed words plus character trigrams; trigrams matter because Russian
    inflection that survives stemming (and typos in queries) still shares most of its
    character n-grams. Each feature is hashed to a coordinate and to a sign, term
    frequency is damped with ``1 + log(tf)``, and the result is L2-normalised so that
    cosine similarity is a dot product.
    """

    def __init__(self, dim: int = 512, seed: int = 20260214, char_ngram: int = 3) -> None:
        if dim <= 0:
            raise EmbeddingError("dim must be positive")
        if char_ngram < 2:
            raise EmbeddingError("char_ngram must be at least 2")
        self.name = f"hashing-{dim}"
        self.dim = dim
        self.seed = seed
        self.char_ngram = char_ngram

    def _features(self, text: str) -> Counter[str]:
        features: Counter[str] = Counter()
        folded = fold(text)
        for token in tokenize(folded):
            features[f"w:{stem(token)}"] += 1
            if len(token) > self.char_ngram:
                padded = f"^{token}$"
                for index in range(len(padded) - self.char_ngram + 1):
                    features[f"c:{padded[index : index + self.char_ngram]}"] += 1
        return features

    def _vector(self, text: str) -> Vector:
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature, count in self._features(text).items():
            digest = _stable_hash(feature, self.seed)
            index = digest % self.dim
            sign = 1.0 if (digest >> 63) & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        return vector

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        stacked: Vector = np.vstack([self._vector(text) for text in texts])
        return l2_normalize(stacked)

    def embed_query(self, text: str) -> Vector:
        return cast(Vector, self.embed_documents([text])[0])


class OllamaEmbedder:
    """Embeddings from a local Ollama server.

    Batching is the whole reason this class exists: ``nomic-embed-text`` over HTTP costs
    roughly as much per request as per item at small sizes, so indexing chunk-by-chunk
    wastes most of the wall clock on round trips.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str = "nomic-embed-text",
        dim: int = 768,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be positive")
        self.name = f"ollama:{model}"
        self.dim = dim
        self.model = model
        self.batch_size = batch_size
        self._client = client

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                vectors.extend(self._client.embed(self.model, batch))
            except OllamaError as error:
                raise EmbeddingError(
                    f"embedding batch at offset {start} failed: {error}"
                ) from error
        widths = {len(vector) for vector in vectors}
        if widths != {self.dim}:
            raise DimensionMismatchError(self.dim, next(iter(sorted(widths))))
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


    def embed_query(self, text: str) -> Vector:
        return cast(Vector, self.embed_documents([text])[0])


def build_embedder(
    provider: str,
    *,
    model: str,
    dim: int,
    batch_size: int,
    seed: int,
    client: OllamaClient | None = None,
) -> Embedder:
    if provider == "hashing":
        return HashingEmbedder(dim=dim, seed=seed)
    if provider == "ollama":
        if client is None:
            raise EmbeddingError("the ollama embedder needs an OllamaClient")
        return OllamaEmbedder(client, model=model, dim=dim, batch_size=batch_size)
    raise EmbeddingError(f"unknown embedding provider: {provider!r}")
