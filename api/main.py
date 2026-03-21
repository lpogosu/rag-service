"""FastAPI surface over the pipeline.

The endpoints are thin by design — all the interesting behaviour is in :mod:`rag`, and
that is where it is tested. What this layer adds is the streaming contract, which is the
one place where the HTTP shape affects semantics: citations can only be validated after
the last token, so the stream carries provisional text first and the authoritative
answer last.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas import (
    HealthResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
)
from rag.config import load_config
from rag.filtering import Filters
from rag.generate import AnswerComplete, AnswerDelta
from rag.pipeline import RagPipeline, RetrievalEvent
from rag.types import Document

DEFAULT_CONFIG = "config/offline.yaml"


def create_app(config_path: str | None = None) -> FastAPI:
    path = config_path or os.environ.get("RAG_CONFIG", DEFAULT_CONFIG)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        pipeline = RagPipeline.from_config(load_config(path))
        application.state.pipeline = pipeline
        try:
            yield
        finally:
            pipeline.close()

    application = FastAPI(
        title="rag-service",
        version="0.5.0",
        summary="Retrieval-augmented generation with explicit, measurable stages",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        pipeline = _pipeline(request)
        detail = dict(pipeline.stats())
        # An empty index is a valid state but not a usable one, and a load balancer
        # should know the difference between "the process is up" and "it can answer".
        status = "ok" if pipeline.store.count() > 0 else "empty"
        return HealthResponse(status=status, detail=detail)

    @application.post("/index", response_model=IndexResponse)
    def index(request: Request, payload: IndexRequest) -> IndexResponse:
        pipeline = _pipeline(request)
        report = pipeline.index(
            [
                Document(doc_id=item.doc_id, text=item.text, metadata=dict(item.metadata))
                for item in payload.documents
            ]
        )
        return IndexResponse(
            documents=report.documents,
            chunks=report.chunks,
            chunker=report.chunker,
            elapsed_ms=round(report.millis, 2),
        )

    @application.post("/query", response_model=QueryResponse)
    def query(request: Request, payload: QueryRequest) -> QueryResponse:
        pipeline = _pipeline(request)
        result = pipeline.query(payload.query, filters=_filters(payload))
        return QueryResponse.model_validate(result.to_dict())

    @application.post("/query/stream")
    def query_stream(request: Request, payload: QueryRequest) -> StreamingResponse:
        pipeline = _pipeline(request)
        return StreamingResponse(
            _sse(pipeline, payload),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return application


def _pipeline(request: Request) -> RagPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if not isinstance(pipeline, RagPipeline):
        raise HTTPException(status_code=503, detail="pipeline is not initialised")
    return pipeline


def _filters(payload: QueryRequest) -> Filters | None:
    return dict(payload.filters) if payload.filters else None


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse(pipeline: RagPipeline, payload: QueryRequest) -> Iterator[str]:
    """Server-sent events for one query.

    Three event types. ``sources`` arrives first so the client can render citations
    while the answer is still being written; ``delta`` carries provisional text; and
    ``answer`` is authoritative — if it says ``refused``, the deltas that preceded it
    did not survive citation checking and must be discarded.
    """
    try:
        for event in pipeline.stream(payload.query, filters=_filters(payload)):
            if isinstance(event, RetrievalEvent):
                yield _event(
                    "sources",
                    {
                        "retrieved": [
                            {
                                "chunk_id": item.chunk.chunk_id,
                                "doc_id": item.chunk.doc_id,
                                "score": round(item.score, 6),
                            }
                            for item in event.retrieved
                        ]
                    },
                )
            elif isinstance(event, AnswerDelta):
                yield _event("delta", {"text": event.text})
            elif isinstance(event, AnswerComplete):
                answer = event.answer
                yield _event(
                    "answer",
                    {
                        "answer": answer.text,
                        "refused": answer.refused,
                        "reason": answer.reason,
                        "citations": [
                            {
                                "marker": citation.marker,
                                "chunk_id": citation.chunk_id,
                                "doc_id": citation.doc_id,
                                "quote": citation.quote,
                            }
                            for citation in answer.citations
                        ],
                    },
                )
    # A stream that dies mid-flight must still end with a frame the client can parse;
    # an unhandled traceback would leave the connection open and the UI spinning.
    except Exception as error:
        yield _event("error", {"message": f"{type(error).__name__}: {error}"})


app = create_app()
