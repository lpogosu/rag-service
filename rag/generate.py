"""Answer synthesis with structurally enforced citations.

The rule this module exists to implement: a citation is valid only if the pipeline can
resolve it to a chunk that was actually retrieved for this query. Nothing else counts.

Asking the model nicely for citations in the prompt produces citations that *look*
right — plausible file names, plausible section numbers — including for chunks that
were never in the context. So the model is never allowed to name a source. It gets
opaque integer markers assigned at request time, and every marker it returns is looked
up in that request's marker table; anything that does not resolve is deleted from the
answer text and from the citation list. If nothing survives, the answer is refused.

The same logic makes the empty-retrieval path trivial: no chunks means no markers means
no possible valid citation, so the model is not called at all.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from rag.ollama import OllamaClient, OllamaError
from rag.text import Analyzer, sentence_spans, snippet
from rag.types import Answer, Chunk, Citation, FusedChunk

DEFAULT_REFUSAL = (
    "В предоставленных документах нет ответа на этот вопрос. "
    "Уточните формулировку или проиндексируйте нужный документ."
)

_MARKER_RE = re.compile(r"\[(\d{1,3})\]")

_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "citations"],
}

_SYSTEM_PROMPT = """Ты отвечаешь на вопросы строго по приведённым фрагментам документации.

Правила:
- Используй только информацию из фрагментов. Ничего не додумывай.
- После каждого утверждения ставь номер фрагмента в квадратных скобках: [1], [2].
- В поле citations перечисли номера всех использованных фрагментов.
- Если во фрагментах нет ответа, верни пустой citations и напиши, что ответа нет.
- Отвечай на языке вопроса, кратко, без вступлений.

Формат ответа — JSON: {"answer": "...", "citations": [1, 2]}"""


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    max_context_chars: int = 6000
    max_chunk_chars: int = 1500
    refusal_text: str = DEFAULT_REFUSAL
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class AnswerDelta:
    """A piece of answer text as it is produced."""

    text: str


@dataclass(frozen=True, slots=True)
class AnswerComplete:
    """The validated answer.

    Streaming and citation enforcement genuinely conflict: text has already reached the
    client by the time the citation list can be checked. Rather than pretend otherwise,
    the stream ends with the authoritative object, and a consumer that receives
    ``refused=True`` after deltas must discard what it showed.
    """

    answer: Answer


GenerationEvent = AnswerDelta | AnswerComplete


class Generator(Protocol):
    name: str

    def generate(self, query: str, context: Sequence[FusedChunk]) -> Answer: ...

    def stream(self, query: str, context: Sequence[FusedChunk]) -> Iterator[GenerationEvent]: ...


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """The prompt-visible context and the marker table used to validate citations."""

    text: str
    markers: dict[int, Chunk]

    def __bool__(self) -> bool:
        return bool(self.markers)


def build_context(context: Sequence[FusedChunk], settings: GenerationSettings) -> ContextBlock:
    """Number the chunks and pack them into the context budget, in ranked order.

    Chunks are dropped from the tail rather than truncated in the middle: half a chunk
    is a chunk that can be cited for a claim its missing half contradicts.
    """
    parts: list[str] = []
    markers: dict[int, Chunk] = {}
    used = 0
    for index, item in enumerate(context, start=1):
        body = item.chunk.embedding_text()[: settings.max_chunk_chars]
        block = f"[{index}] {body}"
        if used + len(block) > settings.max_context_chars and markers:
            break
        parts.append(block)
        markers[index] = item.chunk
        used += len(block)
    return ContextBlock(text="\n\n".join(parts), markers=markers)


def enforce_citations(
    text: str,
    claimed: Sequence[int],
    markers: dict[int, Chunk],
    settings: GenerationSettings,
) -> Answer:
    """Drop every citation that does not resolve, then decide whether an answer remains.

    Markers inside the text are rewritten as well as filtered: leaving "[7]" in the
    prose while removing 7 from the citation list would show the reader a source the
    system does not stand behind.
    """
    if not markers:
        return Answer(
            text=settings.refusal_text,
            citations=(),
            refused=True,
            reason="retrieval returned no chunks",
        )

    in_text = {int(match.group(1)) for match in _MARKER_RE.finditer(text)}
    claimed_then_written = dict.fromkeys([*claimed, *sorted(in_text)])
    resolved = [marker for marker in claimed_then_written if marker in markers]
    cleaned = _MARKER_RE.sub(
        lambda match: match.group(0) if int(match.group(1)) in markers else "", text
    )
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    if not resolved:
        return Answer(
            text=settings.refusal_text,
            citations=(),
            refused=True,
            reason="no citation resolved to a retrieved chunk",
        )
    if not cleaned:
        return Answer(
            text=settings.refusal_text,
            citations=(),
            refused=True,
            reason="model returned an empty answer",
        )

    citations = tuple(
        Citation(
            marker=marker,
            chunk_id=markers[marker].chunk_id,
            doc_id=markers[marker].doc_id,
            quote=snippet(markers[marker].text, 240),
        )
        for marker in resolved
    )
    return Answer(text=cleaned, citations=citations, refused=False)


class ExtractiveGenerator:
    """Selects supporting sentences from the retrieved chunks. No model involved.

    This is what the evaluation harness and the tests use. It cannot paraphrase, so its
    answers read like quotes — but it is deterministic, free, and it makes every
    retrieval-quality number in the README a property of retrieval rather than a
    property of whichever model happened to be loaded.
    """

    name = "extractive"

    def __init__(
        self,
        analyzer: Analyzer | None = None,
        settings: GenerationSettings | None = None,
        max_sentences: int = 3,
        min_support: float = 0.0,
    ) -> None:
        self.analyzer = analyzer or Analyzer()
        self.settings = settings or GenerationSettings()
        self.max_sentences = max_sentences
        self.min_support = min_support

    def generate(self, query: str, context: Sequence[FusedChunk]) -> Answer:
        block = build_context(context, self.settings)
        if not block:
            return enforce_citations("", (), block.markers, self.settings)
        query_terms = set(self.analyzer.analyze(query))
        selected: list[tuple[float, int, int, str]] = []
        for marker, chunk in block.markers.items():
            text = chunk.text
            for order, span in enumerate(sentence_spans(text)):
                sentence = " ".join(text[span.start : span.end].split())
                if len(sentence) < 20:
                    continue
                terms = set(self.analyzer.analyze(sentence))
                if not terms:
                    continue
                overlap = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
                if overlap > 0 and overlap >= self.min_support:
                    selected.append((-overlap, marker, order, sentence))
        if not selected:
            return enforce_citations("", (), block.markers, self.settings)
        selected.sort()
        chosen = selected[: self.max_sentences]
        parts = [f"{sentence} [{marker}]" for _, marker, _, sentence in chosen]
        used = [marker for _, marker, _, _ in chosen]
        return enforce_citations(" ".join(parts), used, block.markers, self.settings)

    def stream(self, query: str, context: Sequence[FusedChunk]) -> Iterator[GenerationEvent]:
        answer = self.generate(query, context)
        if not answer.refused:
            for piece in answer.text.split(". "):
                yield AnswerDelta(text=piece + ". " if not piece.endswith(".") else piece)
        yield AnswerComplete(answer=answer)


class JsonStringFieldStreamer:
    """Extracts one top-level JSON string field from a stream, character by character.

    The model is asked for structured output, so the raw stream is JSON, not prose. To
    show text while it is being produced, the value of ``answer`` has to be unescaped
    incrementally — including across chunk boundaries that split an escape sequence.
    """

    _ESCAPES: ClassVar[dict[str, str]] = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }

    def __init__(self, field: str = "answer") -> None:
        self._pattern = re.compile(r'"' + re.escape(field) + r'"\s*:\s*"')
        self._buffer = ""
        self._cursor = -1
        self._done = False

    @property
    def raw(self) -> str:
        return self._buffer

    def feed(self, text: str) -> str:
        self._buffer += text
        if self._done:
            return ""
        if self._cursor < 0:
            match = self._pattern.search(self._buffer)
            if match is None:
                return ""
            self._cursor = match.end()

        out: list[str] = []
        index = self._cursor
        buffer = self._buffer
        size = len(buffer)
        while index < size:
            char = buffer[index]
            if char == "\\":
                if index + 1 >= size:
                    break
                escape = buffer[index + 1]
                if escape == "u":
                    if index + 6 > size:
                        break
                    out.append(chr(int(buffer[index + 2 : index + 6], 16)))
                    index += 6
                    continue
                out.append(self._ESCAPES.get(escape, escape))
                index += 2
                continue
            if char == '"':
                self._done = True
                index += 1
                break
            out.append(char)
            index += 1
        self._cursor = index
        return "".join(out)


class OllamaGenerator:
    """Answer synthesis by a local instruction model."""

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        settings: GenerationSettings | None = None,
    ) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.settings = settings or GenerationSettings()
        self._client = client

    def _messages(self, query: str, block: ContextBlock) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Фрагменты:\n{block.text}\n\nВопрос: {query}"},
        ]

    def generate(self, query: str, context: Sequence[FusedChunk]) -> Answer:
        block = build_context(context, self.settings)
        if not block:
            return enforce_citations("", (), block.markers, self.settings)
        try:
            raw = self._client.chat(
                self.model,
                self._messages(query, block),
                response_format=_ANSWER_SCHEMA,
                options={"temperature": self.settings.temperature},
            )
        except OllamaError as error:
            return Answer(
                text=self.settings.refusal_text,
                citations=(),
                refused=True,
                reason=f"generation failed: {error}",
            )
        text, claimed = _parse_payload(raw)
        return enforce_citations(text, claimed, block.markers, self.settings)

    def stream(self, query: str, context: Sequence[FusedChunk]) -> Iterator[GenerationEvent]:
        block = build_context(context, self.settings)
        if not block:
            yield AnswerComplete(answer=enforce_citations("", (), block.markers, self.settings))
            return
        streamer = JsonStringFieldStreamer("answer")
        try:
            for delta in self._client.chat_stream(
                self.model,
                self._messages(query, block),
                response_format=_ANSWER_SCHEMA,
                options={"temperature": self.settings.temperature},
            ):
                visible = streamer.feed(delta)
                if visible:
                    yield AnswerDelta(text=visible)
        except OllamaError as error:
            yield AnswerComplete(
                answer=Answer(
                    text=self.settings.refusal_text,
                    citations=(),
                    refused=True,
                    reason=f"generation failed: {error}",
                )
            )
            return
        text, claimed = _parse_payload(streamer.raw)
        yield AnswerComplete(answer=enforce_citations(text, claimed, block.markers, self.settings))


def _parse_payload(raw: str) -> tuple[str, list[int]]:
    """Read the model's JSON. A malformed payload yields no citations, so it refuses."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "", []
    if not isinstance(payload, dict):
        return "", []
    text = payload.get("answer")
    citations = payload.get("citations")
    markers: list[int] = []
    if isinstance(citations, list):
        markers = [int(value) for value in citations if isinstance(value, int | float)]
    return (text if isinstance(text, str) else ""), markers


def build_generator(
    provider: str,
    *,
    model: str,
    analyzer: Analyzer,
    settings: GenerationSettings,
    max_sentences: int = 3,
    min_support: float = 0.0,
    client: OllamaClient | None = None,
) -> Generator:
    if provider == "extractive":
        return ExtractiveGenerator(
            analyzer, settings, max_sentences=max_sentences, min_support=min_support
        )
    if provider == "ollama":
        if client is None:
            raise ValueError("the ollama generator needs an OllamaClient")
        return OllamaGenerator(client, model, settings)
    raise ValueError(f"unknown generation provider: {provider!r}")
