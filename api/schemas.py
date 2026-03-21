"""Request and response models for the HTTP layer.

Kept separate from :mod:`rag.types` so that the wire format can change without the
pipeline noticing, and so that a rename inside the pipeline cannot silently break a
client.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentIn] = Field(min_length=1, max_length=500)


class IndexResponse(BaseModel):
    documents: int
    chunks: int
    chunker: str
    elapsed_ms: float


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    filters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class CitationOut(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    quote: str


class RetrievedOut(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    ranks: dict[str, int]


class QueryResponse(BaseModel):
    query: str
    answer: str
    refused: bool
    reason: str
    citations: list[CitationOut]
    retrieved: list[RetrievedOut]
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    detail: dict[str, Any]
