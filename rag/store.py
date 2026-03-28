"""Vector stores: pgvector for real deployments, in-memory for tests and evaluation.

Both implement the same protocol, so the pipeline, the harness and the API cannot tell
them apart. That is what makes the evaluation numbers reproducible without a database
and the production path a one-line config change.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np
import psycopg
from psycopg import sql

from rag.embedding import Vector
from rag.filtering import Filters, matches_filters
from rag.types import Chunk, Metadata, ScoredChunk

Metric = Literal["cosine", "l2"]


class StoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk: Chunk
    vector: Vector


class VectorStore(Protocol):
    dim: int
    metric: Metric

    def upsert(self, records: Sequence[ChunkRecord]) -> None: ...

    def search(
        self, vector: Vector, k: int, *, filters: Filters | None = None
    ) -> list[ScoredChunk]: ...

    def delete_document(self, doc_id: str) -> int: ...

    def iter_chunks(self) -> Iterator[Chunk]: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


def _scores(matrix: Vector, query: Vector, metric: Metric) -> Vector:
    """Similarity of every row to the query. Higher is always better.

    For L2 the sign is flipped rather than inverted, because any monotone transform
    would do for ranking and a plain negation keeps the number readable as a distance.
    """
    if metric == "cosine":
        return cast(Vector, matrix @ query)
    return cast(Vector, -np.linalg.norm(matrix - query, axis=1))


class InMemoryVectorStore:
    """Exhaustive search over a dense matrix.

    No index, no approximation: this is the ground truth the HNSW recall figures in the
    README are measured against, and at eval-corpus scale a full scan is faster than
    building a graph anyway.
    """

    def __init__(self, dim: int, metric: Metric = "cosine") -> None:
        self.dim = dim
        self.metric = metric
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, Vector] = {}
        self._order: list[str] = []
        self._matrix: Vector | None = None

    def upsert(self, records: Sequence[ChunkRecord]) -> None:
        for record in records:
            if record.vector.shape != (self.dim,):
                raise StoreError(
                    f"chunk {record.chunk.chunk_id}: vector shape {record.vector.shape} "
                    f"does not match store dim {self.dim}"
                )
            if record.chunk.chunk_id not in self._chunks:
                self._order.append(record.chunk.chunk_id)
            self._chunks[record.chunk.chunk_id] = record.chunk
            self._vectors[record.chunk.chunk_id] = record.vector.astype(np.float32)
        self._matrix = None

    def search(
        self, vector: Vector, k: int, *, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        if k <= 0 or not self._order:
            return []
        if filters is None:
            candidates = self._order
            matrix = self._ensure_matrix()
        else:
            candidates = [
                cid for cid in self._order if matches_filters(self._chunks[cid].metadata, filters)
            ]
            if not candidates:
                return []
            matrix = np.vstack([self._vectors[cid] for cid in candidates])
        scores = _scores(matrix, vector.astype(np.float32), self.metric)
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            ScoredChunk(
                chunk=self._chunks[candidates[int(position)]],
                score=float(scores[int(position)]),
                source="dense",
                rank=rank,
            )
            for rank, position in enumerate(top, start=1)
        ]

    def delete_document(self, doc_id: str) -> int:
        removed = [cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id]
        for chunk_id in removed:
            del self._chunks[chunk_id]
            del self._vectors[chunk_id]
        if removed:
            self._order = [cid for cid in self._order if cid in self._chunks]
            self._matrix = None
        return len(removed)

    def iter_chunks(self) -> Iterator[Chunk]:
        for chunk_id in self._order:
            yield self._chunks[chunk_id]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._order.clear()
        self._matrix = None

    def _ensure_matrix(self) -> Vector:
        if self._matrix is None:
            self._matrix = np.vstack([self._vectors[cid] for cid in self._order])
        return self._matrix


@dataclass(frozen=True, slots=True)
class HnswSettings:
    """HNSW build and search parameters.

    ``m`` and ``ef_construction`` are fixed at build time and decide the graph's quality
    ceiling; ``ef_search`` is the per-query knob that trades latency for recall. Exposed
    in config precisely because it is the one worth tuning against your own data.
    """

    m: int = 16
    ef_construction: int = 64
    ef_search: int = 64


def to_vector_literal(vector: Vector) -> str:
    return "[" + ",".join(f"{float(value):.7g}" for value in vector) + "]"


def _row_to_chunk(row: tuple[Any, ...]) -> Chunk:
    metadata = row[6] if isinstance(row[6], dict) else json.loads(row[6])
    return Chunk(
        chunk_id=str(row[0]),
        doc_id=str(row[1]),
        ordinal=int(row[2]),
        start=int(row[3]),
        end=int(row[4]),
        text=str(row[5]),
        metadata=cast(Metadata, metadata),
    )


class PgVectorStore:
    """pgvector-backed store with an HNSW index.

    One table, one HNSW index for similarity and one GIN index for metadata. Metadata
    filtering uses jsonb containment (``@>``) rather than expression indexes on
    individual keys, so adding a new filterable field costs nothing at schema level —
    at the price of a filter that Postgres applies after the index scan, which is the
    documented recall trap discussed in the README.
    """

    def __init__(
        self,
        dsn: str,
        dim: int,
        *,
        metric: Metric = "cosine",
        table: str = "rag_chunks",
        hnsw: HnswSettings | None = None,
    ) -> None:
        self.dim = dim
        self.metric = metric
        self.table_name = table
        self.table = sql.Identifier(table)
        self.hnsw = hnsw or HnswSettings()
        self._connection: psycopg.Connection[tuple[Any, ...]] = psycopg.connect(
            dsn, autocommit=True
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PgVectorStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def _distance_operator(self) -> sql.SQL:
        return sql.SQL("<=>") if self.metric == "cosine" else sql.SQL("<->")

    @property
    def _index_opclass(self) -> sql.Identifier:
        return sql.Identifier("vector_cosine_ops" if self.metric == "cosine" else "vector_l2_ops")

    def create_schema(self) -> None:
        statements: list[sql.Composed | sql.SQL] = [
            sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"),
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ("
                "chunk_id text PRIMARY KEY,"
                "doc_id text NOT NULL,"
                "ordinal integer NOT NULL,"
                "start_char integer NOT NULL,"
                "end_char integer NOT NULL,"
                "content text NOT NULL,"
                "metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,"
                "embedding vector({dim}) NOT NULL,"
                "updated_at timestamptz NOT NULL DEFAULT now())"
            ).format(table=self.table, dim=sql.Literal(self.dim)),
            sql.SQL("CREATE INDEX IF NOT EXISTS {name} ON {table} (doc_id)").format(
                name=sql.Identifier(f"{self.table_name}_doc_id_idx"), table=self.table
            ),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {table} USING gin (metadata jsonb_path_ops)"
            ).format(name=sql.Identifier(f"{self.table_name}_metadata_idx"), table=self.table),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {table} "
                "USING hnsw (embedding {opclass}) WITH (m = {m}, ef_construction = {efc})"
            ).format(
                name=sql.Identifier(f"{self.table_name}_embedding_idx"),
                table=self.table,
                opclass=self._index_opclass,
                m=sql.Literal(self.hnsw.m),
                efc=sql.Literal(self.hnsw.ef_construction),
            ),
        ]
        with self._connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)

    def upsert(self, records: Sequence[ChunkRecord]) -> None:
        if not records:
            return
        for record in records:
            if record.vector.shape != (self.dim,):
                raise StoreError(
                    f"chunk {record.chunk.chunk_id}: vector shape {record.vector.shape} "
                    f"does not match store dim {self.dim}"
                )
        statement = sql.SQL(
            "INSERT INTO {table} "
            "(chunk_id, doc_id, ordinal, start_char, end_char, content, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector) "
            "ON CONFLICT (chunk_id) DO UPDATE SET "
            "doc_id = EXCLUDED.doc_id, ordinal = EXCLUDED.ordinal, "
            "start_char = EXCLUDED.start_char, end_char = EXCLUDED.end_char, "
            "content = EXCLUDED.content, metadata = EXCLUDED.metadata, "
            "embedding = EXCLUDED.embedding, updated_at = now()"
        ).format(table=self.table)
        rows = [
            (
                record.chunk.chunk_id,
                record.chunk.doc_id,
                record.chunk.ordinal,
                record.chunk.start,
                record.chunk.end,
                record.chunk.text,
                json.dumps(record.chunk.metadata, ensure_ascii=False),
                to_vector_literal(record.vector),
            )
            for record in records
        ]
        with self._connection.cursor() as cursor:
            cursor.executemany(statement, rows)

    def search(
        self, vector: Vector, k: int, *, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        literal = to_vector_literal(vector)
        where = sql.SQL("")
        params: list[Any] = [literal]
        if filters:
            where = sql.SQL(" WHERE metadata @> %s::jsonb")
            params.append(json.dumps(containment_filter(filters), ensure_ascii=False))
        params.extend([literal, k])
        statement = sql.SQL(
            "SELECT chunk_id, doc_id, ordinal, start_char, end_char, content, metadata, "
            "embedding {op} %s::vector AS distance "
            "FROM {table}{where} ORDER BY embedding {op} %s::vector LIMIT %s"
        ).format(table=self.table, where=where, op=self._distance_operator)
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(self.hnsw.ef_search))
            )
            cursor.execute(statement, params)
            rows = cursor.fetchall()
        return [
            ScoredChunk(
                chunk=_row_to_chunk(row),
                score=1.0 - float(row[7]) if self.metric == "cosine" else -float(row[7]),
                source="dense",
                rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def delete_document(self, doc_id: str) -> int:
        statement = sql.SQL("DELETE FROM {table} WHERE doc_id = %s").format(table=self.table)
        with self._connection.cursor() as cursor:
            cursor.execute(statement, (doc_id,))
            return cursor.rowcount

    def iter_chunks(self) -> Iterator[Chunk]:
        statement = sql.SQL(
            "SELECT chunk_id, doc_id, ordinal, start_char, end_char, content, metadata "
            "FROM {table} ORDER BY doc_id, ordinal"
        ).format(table=self.table)
        with self._connection.cursor(name="rag_chunks_scan") as cursor:
            cursor.execute(statement)
            for row in cursor:
                yield _row_to_chunk(row)

    def count(self) -> int:
        statement = sql.SQL("SELECT count(*) FROM {table}").format(table=self.table)
        with self._connection.cursor() as cursor:
            cursor.execute(statement)
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        statement = sql.SQL("TRUNCATE {table}").format(table=self.table)
        with self._connection.cursor() as cursor:
            cursor.execute(statement)


def containment_filter(filters: Filters) -> dict[str, Any]:
    """jsonb containment only expresses equality, so multi-value filters are rejected.

    Failing loudly beats silently dropping half a filter and returning results the
    caller believes were filtered.
    """
    payload: dict[str, Any] = {}
    for key, value in filters.items():
        if isinstance(value, list | tuple | set):
            raise StoreError(
                f"filter {key!r}: pgvector store supports equality filters only; "
                "issue one query per value or add a dedicated column"
            )
        payload[key] = value
    return payload
