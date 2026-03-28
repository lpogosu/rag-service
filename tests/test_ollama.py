"""Ollama client: what is retried, what is not, and what is raised."""

from __future__ import annotations

import httpx
import pytest
import respx

from rag.ollama import OllamaClient, OllamaError, OllamaSettings

BASE_URL = "http://ollama.test:11434"


def client(max_attempts: int = 3) -> tuple[OllamaClient, list[float]]:
    slept: list[float] = []
    return (
        OllamaClient(
            OllamaSettings(base_url=BASE_URL, max_attempts=max_attempts, backoff_seconds=0.5),
            sleep=slept.append,
        ),
        slept,
    )


@respx.mock
def test_a_transient_server_error_is_retried() -> None:
    route = respx.post(f"{BASE_URL}/api/embed").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"embeddings": [[1.0]]}),
        ]
    )
    connection, slept = client()
    assert connection.embed("m", ["x"]) == [[1.0]]
    assert route.call_count == 2
    assert slept == [0.5]


@respx.mock
def test_backoff_doubles_between_attempts() -> None:
    respx.post(f"{BASE_URL}/api/embed").mock(return_value=httpx.Response(503))
    connection, slept = client(max_attempts=3)
    with pytest.raises(OllamaError, match="after 3 attempts"):
        connection.embed("m", ["x"])
    assert slept == [0.5, 1.0]


@respx.mock
def test_a_client_error_is_not_retried() -> None:
    """A 400 means the request is wrong; repeating it only delays the error."""
    route = respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(400, text="bad model")
    )
    connection, slept = client()
    with pytest.raises(OllamaError, match="HTTP 400"):
        connection.chat("m", [{"role": "user", "content": "hi"}])
    assert route.call_count == 1
    assert slept == []


@respx.mock
def test_a_connection_failure_is_retried_then_reported() -> None:
    respx.post(f"{BASE_URL}/api/embed").mock(side_effect=httpx.ConnectError("refused"))
    connection, _ = client(max_attempts=2)
    with pytest.raises(OllamaError, match="ConnectError"):
        connection.embed("m", ["x"])


@respx.mock
def test_a_short_embedding_response_is_rejected() -> None:
    """Silently returning fewer vectors than inputs would misalign every chunk."""
    respx.post(f"{BASE_URL}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[1.0]]})
    )
    connection, _ = client(max_attempts=1)
    with pytest.raises(OllamaError, match="1 vectors for 2 inputs"):
        connection.embed("m", ["a", "b"])


@respx.mock
def test_chat_returns_the_message_content() -> None:
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "ответ"}})
    )
    connection, _ = client()
    assert connection.chat("m", [{"role": "user", "content": "?"}]) == "ответ"


@respx.mock
def test_chat_without_content_is_an_error() -> None:
    respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(200, json={"done": True}))
    connection, _ = client(max_attempts=1)
    with pytest.raises(OllamaError, match="no message content"):
        connection.chat("m", [{"role": "user", "content": "?"}])


@respx.mock
def test_streaming_yields_deltas_until_done() -> None:
    body = "\n".join(
        [
            '{"message": {"content": "раз "}, "done": false}',
            '{"message": {"content": "два"}, "done": false}',
            '{"message": {"content": ""}, "done": true}',
            '{"message": {"content": "после done"}, "done": false}',
        ]
    )
    respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(200, text=body))
    connection, _ = client()
    assert list(connection.chat_stream("m", [{"role": "user", "content": "?"}])) == ["раз ", "два"]


@respx.mock
def test_a_streaming_failure_is_wrapped() -> None:
    respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(500))
    connection, _ = client()
    with pytest.raises(OllamaError, match="streaming chat failed"):
        list(connection.chat_stream("m", [{"role": "user", "content": "?"}]))


@respx.mock
def test_availability_probe_never_raises() -> None:
    respx.get(f"{BASE_URL}/api/tags").mock(side_effect=httpx.ConnectError("down"))
    connection, _ = client()
    assert connection.is_available() is False


@respx.mock
def test_availability_probe_reports_a_live_server() -> None:
    respx.get(f"{BASE_URL}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))
    connection, _ = client()
    assert connection.is_available() is True
