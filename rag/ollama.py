"""Minimal Ollama HTTP client.

Only the three calls this service makes are implemented. A generated SDK would add a
dependency, a version to track and a lot of surface area for three endpoints whose
payloads fit on one screen.

The retry policy is deliberately narrow: connection failures, timeouts, 429 and 5xx are
retried, everything else is raised immediately. A 400 from Ollama means the request is
wrong, and repeating it five times only delays the error.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class OllamaError(RuntimeError):
    """Raised when Ollama cannot serve a request after the configured retries."""


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    backoff_seconds: float = 0.5


class OllamaClient:
    def __init__(
        self,
        settings: OllamaSettings | None = None,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or OllamaSettings()
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_seconds,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def is_available(self) -> bool:
        try:
            response = self._client.get("/api/tags")
        except httpx.HTTPError:
            return False
        return response.status_code == httpx.codes.OK

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        payload = {"model": model, "input": inputs}
        body = self._post_json("/api/embed", payload)
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise OllamaError(
                f"/api/embed returned {len(embeddings) if isinstance(embeddings, list) else 'no'} "
                f"vectors for {len(inputs)} inputs"
            )
        return [[float(value) for value in vector] for vector in embeddings]

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if response_format is not None:
            payload["format"] = response_format
        if options:
            payload["options"] = options
        body = self._post_json("/api/chat", payload)
        message = body.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaError("/api/chat returned no message content")
        return content

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | str | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield content deltas from a streaming chat completion.

        Not retried: once tokens have been handed to the caller a retry would duplicate
        them, and the caller is better placed to decide what to do with a half answer.
        """
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if response_format is not None:
            payload["format"] = response_format
        if options:
            payload["options"] = options
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    message = chunk.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content:
                            yield content
                    if chunk.get("done"):
                        return
        except httpx.HTTPError as error:
            raise OllamaError(f"streaming chat failed: {error}") from error

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error = ""
        for attempt in range(self.settings.max_attempts):
            try:
                response = self._client.post(path, json=payload)
            except httpx.HTTPError as error:
                last_error = f"{type(error).__name__}: {error}"
            else:
                if response.status_code == httpx.codes.OK:
                    parsed = response.json()
                    if not isinstance(parsed, dict):
                        raise OllamaError(
                            f"{path} returned {type(parsed).__name__}, expected an object"
                        )
                    return parsed
                if response.status_code not in RETRYABLE_STATUS:
                    raise OllamaError(
                        f"{path} failed with HTTP {response.status_code}: {response.text[:200]}"
                    )
                last_error = f"HTTP {response.status_code}"
            if attempt + 1 < self.settings.max_attempts:
                self._sleep(self.settings.backoff_seconds * (2**attempt))
        raise OllamaError(
            f"{path} failed after {self.settings.max_attempts} attempts ({last_error})"
        )
