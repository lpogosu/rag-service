"""The pipeline: the only place the stages know about each other.

Reading this file top to bottom is meant to be the fastest way to understand what
retrieval-augmented generation actually does. Each stage is a field, each field comes
from config, and every stage is timed — because "RAG is slow" is never actionable and
"reranking is 780 ms of an 900 ms request" is.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from rag.chunking import Chunker, build_chunker
from rag.config import AppConfig
from rag.embedding import Embedder, build_embedder
from rag.filtering import Filters
from rag.generate import (
    AnswerComplete,
    AnswerDelta,
    GenerationSettings,
    Generator,
    build_generator,
)
from rag.hybrid import FusionParams, fuse
from rag.lexical import Bm25Index, Bm25Params
from rag.ollama import OllamaClient, OllamaSettings
from rag.rerank import Reranker, build_reranker
from rag.store import (
    ChunkRecord,
    HnswSettings,
    InMemoryVectorStore,
    PgVectorStore,
    VectorStore,
)
from rag.text import Analyzer
from rag.types import Chunk, Document, FusedChunk, QueryResult, ScoredChunk, StageTiming


@dataclass(frozen=True, slots=True)
class IndexReport:
    documents: int
    chunks: int
    chunker: str
    millis: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "chunker": self.chunker,
            "elapsed_ms": round(self.millis, 2),
        }


@dataclass(frozen=True, slots=True)
class RetrievalEvent:
    """Emitted before generation starts so a UI can show sources while the answer runs."""

    retrieved: tuple[FusedChunk, ...]


PipelineEvent = RetrievalEvent | AnswerDelta | AnswerComplete


@dataclass
class _Timer:
    timings: list[StageTiming] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings.append(StageTiming(name, (time.perf_counter() - started) * 1000.0))


class RagPipeline:
    def __init__(
        self,
        config: AppConfig,
        *,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker,
        generator: Generator,
        analyzer: Analyzer,
        bm25_params: Bm25Params,
        client: OllamaClient | None = None,
    ) -> None:
        self.config = config
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.generator = generator
        self.analyzer = analyzer
        self.bm25_params = bm25_params
        self._client = client
        self._lexical = Bm25Index(analyzer=analyzer, params=bm25_params)
        self._fusion = FusionParams(
            method=config.retrieval.fusion.method,
            k=config.retrieval.fusion.k,
            weights={
                "dense": config.retrieval.fusion.dense_weight,
                "lexical": config.retrieval.fusion.lexical_weight,
            },
        )

    @classmethod
    def from_config(cls, config: AppConfig, *, client: OllamaClient | None = None) -> RagPipeline:
        if client is None and config.needs_ollama():
            client = OllamaClient(
                OllamaSettings(
                    base_url=config.ollama.base_url,
                    timeout_seconds=config.ollama.timeout_seconds,
                    max_attempts=config.ollama.max_attempts,
                    backoff_seconds=config.ollama.backoff_seconds,
                )
            )
        analyzer = Analyzer(
            use_stemming=config.lexical.use_stemming,
            min_token_length=config.lexical.min_token_length,
        )
        embedder = build_embedder(
            config.embedding.provider,
            model=config.embedding.model,
            dim=config.embedding.dim,
            batch_size=config.embedding.batch_size,
            seed=config.embedding.seed,
            client=client,
        )
        store = _build_store(config, embedder.dim)
        return cls(
            config,
            chunker=build_chunker(config.chunking.strategy, config.chunking.options),
            embedder=embedder,
            store=store,
            reranker=build_reranker(
                config.rerank.provider,
                model=config.rerank.model,
                analyzer=analyzer,
                client=client,
            ),
            generator=build_generator(
                config.generation.provider,
                model=config.generation.model,
                analyzer=analyzer,
                settings=GenerationSettings(
                    max_context_chars=config.generation.max_context_chars,
                    max_chunk_chars=config.generation.max_chunk_chars,
                    temperature=config.generation.temperature,
                ),
                max_sentences=config.generation.max_sentences,
                min_support=config.generation.min_support,
                client=client,
            ),
            analyzer=analyzer,
            bm25_params=Bm25Params(k1=config.lexical.k1, b=config.lexical.b),
            client=client,
        )

    def close(self) -> None:
        if isinstance(self.store, PgVectorStore):
            self.store.close()
        if self._client is not None:
            self._client.close()

    def index(self, documents: Sequence[Document]) -> IndexReport:
        started = time.perf_counter()
        chunks: list[Chunk] = []
        for document in documents:
            # Re-indexing a document must not leave chunks from the previous chunk size
            # behind; they would keep scoring and would never be cited correctly.
            self.store.delete_document(document.doc_id)
            chunks.extend(self.chunker.split(document))
        if chunks:
            vectors = self.embedder.embed_documents([chunk.embedding_text() for chunk in chunks])
            self.store.upsert(
                [ChunkRecord(chunk=chunk, vector=vectors[i]) for i, chunk in enumerate(chunks)]
            )
        self.rebuild_lexical_index()
        return IndexReport(
            documents=len(documents),
            chunks=len(chunks),
            chunker=self.chunker.name,
            millis=(time.perf_counter() - started) * 1000.0,
        )

    def rebuild_lexical_index(self) -> None:
        """Rebuild BM25 from the store, so the two indexes cannot drift apart."""
        self._lexical = Bm25Index.build(
            self.store.iter_chunks(), analyzer=self.analyzer, params=self.bm25_params
        )

    @property
    def lexical_index(self) -> Bm25Index:
        return self._lexical

    def retrieve(
        self,
        query: str,
        *,
        filters: Filters | None = None,
        timer: _Timer | None = None,
    ) -> list[FusedChunk]:
        timer = timer or _Timer()
        retrieval = self.config.retrieval
        rankings: dict[str, Sequence[ScoredChunk]] = {}

        if retrieval.dense_k > 0:
            with timer.stage("dense"):
                vector = self.embedder.embed_query(query)
                rankings["dense"] = self.store.search(vector, retrieval.dense_k, filters=filters)

        if self.config.lexical.enabled and retrieval.lexical_k > 0:
            with timer.stage("lexical"):
                rankings["lexical"] = self._lexical.search(
                    query, retrieval.lexical_k, filters=filters
                )

        with timer.stage("fusion"):
            candidate_count = max(retrieval.final_k, self.config.rerank.top_n)
            fused = fuse(rankings, self._fusion, candidate_count)

        if self.config.rerank.provider != "none":
            with timer.stage("rerank"):
                fused = self.reranker.rerank(
                    query, fused[: self.config.rerank.top_n], retrieval.final_k
                )
        else:
            fused = fused[: retrieval.final_k]
        return fused

    def query(self, question: str, *, filters: Filters | None = None) -> QueryResult:
        timer = _Timer()
        retrieved = self.retrieve(question, filters=filters, timer=timer)
        with timer.stage("generation"):
            answer = self.generator.generate(question, retrieved)
        return QueryResult(
            query=question,
            answer=answer,
            retrieved=tuple(retrieved),
            timings=tuple(timer.timings),
        )

    def stream(self, question: str, *, filters: Filters | None = None) -> Iterator[PipelineEvent]:
        retrieved = self.retrieve(question, filters=filters)
        yield RetrievalEvent(retrieved=tuple(retrieved))
        yield from self.generator.stream(question, retrieved)

    def stats(self) -> dict[str, object]:
        return {
            "config": self.config.name,
            "chunker": self.chunker.name,
            "embedder": self.embedder.name,
            "store": type(self.store).__name__,
            "chunks": self.store.count(),
            "lexical_chunks": len(self._lexical),
            "reranker": self.reranker.name,
            "generator": self.generator.name,
        }


def _build_store(config: AppConfig, dim: int) -> VectorStore:
    if config.store.backend == "memory":
        return InMemoryVectorStore(dim=dim, metric=config.store.metric)
    store = PgVectorStore(
        config.store.dsn,
        dim=dim,
        metric=config.store.metric,
        table=config.store.table,
        hnsw=HnswSettings(
            m=config.store.hnsw.m,
            ef_construction=config.store.hnsw.ef_construction,
            ef_search=config.store.hnsw.ef_search,
        ),
    )
    store.create_schema()
    return store
