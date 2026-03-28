"""HTTP surface, exercised against the offline profile."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from tests.conftest import ENGLISH_DOCUMENT, MARKDOWN_DOCUMENT

PAYLOAD = {
    "documents": [
        {"doc_id": "broker-config", "text": MARKDOWN_DOCUMENT, "metadata": {"lang": "ru"}},
        {"doc_id": "http-api", "text": ENGLISH_DOCUMENT, "metadata": {"lang": "en"}},
    ]
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app("config/offline.yaml")) as test_client:
        yield test_client


@pytest.fixture
def indexed(client: TestClient) -> TestClient:
    client.post("/index", json=PAYLOAD)
    return client


def test_health_distinguishes_running_from_usable(client: TestClient) -> None:
    empty = client.get("/health").json()
    assert empty["status"] == "empty"
    client.post("/index", json=PAYLOAD)
    ready = client.get("/health").json()
    assert ready["status"] == "ok"
    assert ready["detail"]["chunks"] > 0


def test_index_reports_the_chunks_it_created(client: TestClient) -> None:
    response = client.post("/index", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == 2
    assert body["chunks"] > 2
    assert body["chunker"] == "structural"


def test_index_rejects_an_unknown_field(client: TestClient) -> None:
    response = client.post(
        "/index", json={"documents": [{"doc_id": "d", "text": "t", "tags": ["x"]}]}
    )
    assert response.status_code == 422


def test_index_rejects_an_empty_document(client: TestClient) -> None:
    response = client.post("/index", json={"documents": [{"doc_id": "d", "text": ""}]})
    assert response.status_code == 422


def test_query_returns_an_answer_with_resolvable_citations(indexed: TestClient) -> None:
    body = indexed.post("/query", json={"query": "Чему равен retention.hours?"}).json()
    assert not body["refused"]
    assert "168" in body["answer"]
    retrieved = {item["chunk_id"] for item in body["retrieved"]}
    assert {citation["chunk_id"] for citation in body["citations"]} <= retrieved
    assert set(body["timings_ms"]) >= {"dense", "lexical", "fusion", "generation"}


def test_query_honours_metadata_filters(indexed: TestClient) -> None:
    body = indexed.post("/query", json={"query": "topics", "filters": {"lang": "en"}}).json()
    assert all(item["doc_id"] == "http-api" for item in body["retrieved"])


def test_a_filter_that_matches_nothing_refuses(indexed: TestClient) -> None:
    body = indexed.post("/query", json={"query": "порт", "filters": {"lang": "de"}}).json()
    assert body["refused"]
    assert body["retrieved"] == []


def test_query_against_an_empty_index_refuses(client: TestClient) -> None:
    body = client.post("/query", json={"query": "что угодно"}).json()
    assert body["refused"]
    assert body["reason"] == "retrieval returned no chunks"


def test_query_validates_its_input(client: TestClient) -> None:
    assert client.post("/query", json={"query": ""}).status_code == 422
    assert client.post("/query", json={}).status_code == 422


def parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_sends_sources_then_deltas_then_the_final_answer(indexed: TestClient) -> None:
    response = indexed.post("/query/stream", json={"query": "Чему равен retention.hours?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-1] == "answer"
    assert "delta" in names
    final = events[-1][1]
    assert not final["refused"]
    assert final["citations"]


def test_stream_of_an_unanswerable_query_ends_with_a_refusal(client: TestClient) -> None:
    events = parse_sse(client.post("/query/stream", json={"query": "что угодно"}).text)
    assert events[-1][0] == "answer"
    assert events[-1][1]["refused"]
