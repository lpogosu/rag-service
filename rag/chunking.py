"""Chunking strategies behind a single interface.

Every strategy produces chunks that satisfy the same three invariants:

1. ``chunks[0].start == 0`` and ``chunks[-1].end == len(document.text)``;
2. ``chunks[i + 1].start <= chunks[i].end`` — consecutive chunks touch or overlap,
   never leave a gap;
3. both ``start`` and ``end`` increase strictly.

Together they mean the document can be reconstructed byte-for-byte from the chunks
alone, which ``tests/test_chunking.py`` asserts for every strategy on every fixture.
That is the difference between a chunker you can trust and one that silently eats the
last paragraph of a document at a certain size.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import pairwise

from rag.text import Span, sentence_spans, word_spans
from rag.types import Chunk, Document, Metadata

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n")


class ChunkingError(ValueError):
    """Raised when a strategy is configured in a way that cannot produce valid chunks."""


@dataclass(frozen=True, slots=True)
class _Heading:
    offset: int
    path: str


def heading_index(text: str) -> list[_Heading]:
    """Markdown heading offsets with their full path ("Раздел > Подраздел")."""
    stack: list[tuple[int, str]] = []
    headings: list[_Heading] = []
    for match in _HEADING_RE.finditer(text):
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        headings.append(_Heading(match.start(), " > ".join(title for _, title in stack)))
    return headings


def heading_at(headings: list[_Heading], offset: int) -> str:
    """The heading path in effect at ``offset``, or "" above the first heading."""
    current = ""
    for heading in headings:
        if heading.offset > offset:
            break
        current = heading.path
    return current


class Chunker(ABC):
    """A chunking strategy.

    Subclasses implement :meth:`boundaries`, returning the character spans of the
    chunks. The base class handles heading annotation, id assignment and invariant
    checking, so a new strategy cannot accidentally break reconstruction.
    """

    name: str

    @abstractmethod
    def boundaries(self, text: str) -> list[Span]:
        """Chunk spans, ordered, contiguous or overlapping, covering the whole text."""

    def split(self, document: Document) -> list[Chunk]:
        if not document.text.strip():
            return []
        spans = self.boundaries(document.text)
        spans = _merge_undersized(spans, self.min_chars)
        _check_coverage(spans, len(document.text), self.name)
        headings = heading_index(document.text)
        chunks: list[Chunk] = []
        for ordinal, span in enumerate(spans):
            metadata: Metadata = dict(document.metadata)
            heading = heading_at(headings, span.start)
            if heading:
                metadata["heading"] = heading
            metadata["chunker"] = self.name
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{ordinal:04d}",
                    doc_id=document.doc_id,
                    text=document.text[span.start : span.end],
                    start=span.start,
                    end=span.end,
                    ordinal=ordinal,
                    metadata=metadata,
                )
            )
        return chunks

    @property
    def min_chars(self) -> int:
        """Chunks shorter than this are merged into their predecessor."""
        return 0


def _merge_undersized(spans: list[Span], min_chars: int) -> list[Span]:
    if min_chars <= 0 or len(spans) < 2:
        return spans
    merged = list(spans)
    if len(merged[-1]) < min_chars:
        tail = merged.pop()
        previous = merged.pop()
        merged.append(Span(previous.start, tail.end))
    return merged


def _check_coverage(spans: list[Span], length: int, name: str) -> None:
    if not spans:
        raise ChunkingError(f"{name}: produced no chunks for a non-empty document")
    if spans[0].start != 0:
        raise ChunkingError(f"{name}: first chunk starts at {spans[0].start}, not 0")
    if spans[-1].end != length:
        raise ChunkingError(f"{name}: last chunk ends at {spans[-1].end}, not {length}")
    for previous, current in pairwise(spans):
        if current.start > previous.end:
            raise ChunkingError(
                f"{name}: gap between [{previous.start}, {previous.end}) "
                f"and [{current.start}, {current.end})"
            )
        if current.start <= previous.start or current.end <= previous.end:
            raise ChunkingError(f"{name}: chunk spans do not advance at offset {current.start}")


class FixedSizeChunker(Chunker):
    """Sliding window over characters.

    The baseline. It knows nothing about the text, which makes it fast, perfectly
    predictable in cost, and prone to cutting a sentence — and therefore a fact — in
    half. Useful as a control in the benchmark table.
    """

    name = "fixed"

    def __init__(self, size: int = 800, overlap: int = 120) -> None:
        if size <= 0:
            raise ChunkingError("size must be positive")
        if not 0 <= overlap < size:
            raise ChunkingError("overlap must be in [0, size)")
        self.size = size
        self.overlap = overlap

    def boundaries(self, text: str) -> list[Span]:
        step = self.size - self.overlap
        spans: list[Span] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.size, length)
            spans.append(Span(start, end))
            if end == length:
                break
            start += step
        return spans

    @property
    def min_chars(self) -> int:
        return min(self.size // 4, 120)


class SentenceChunker(Chunker):
    """Packs whole sentences up to a character budget.

    Overlap is expressed in sentences rather than characters: repeating half a sentence
    adds tokens without adding retrievable meaning.
    """

    name = "sentence"

    def __init__(self, max_chars: int = 800, overlap_sentences: int = 1) -> None:
        if max_chars <= 0:
            raise ChunkingError("max_chars must be positive")
        if overlap_sentences < 0:
            raise ChunkingError("overlap_sentences must be non-negative")
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences

    def boundaries(self, text: str) -> list[Span]:
        sentences = sentence_spans(text)
        if not sentences:
            return [Span(0, len(text))]

        spans: list[Span] = []
        index = 0
        while index < len(sentences):
            start = sentences[index].start
            last = index
            for lookahead in range(index + 1, len(sentences)):
                if sentences[lookahead].end - start > self.max_chars:
                    break
                last = lookahead
            end = sentences[last].end
            if spans and end <= spans[-1].end:
                # A sentence longer than the budget followed the overlap window, so the
                # new chunk would repeat the previous one and add nothing. Drop the
                # overlap and continue from the first sentence not yet covered.
                index = last + 1
                continue
            spans.append(Span(start, end))
            if last + 1 >= len(sentences):
                break
            index = max(last + 1 - self.overlap_sentences, index + 1)
        spans[-1] = Span(spans[-1].start, len(text))
        return spans

    @property
    def min_chars(self) -> int:
        return min(self.max_chars // 4, 120)


class StructuralChunker(Chunker):
    """Recursive split along the document's own structure.

    Order of separators: markdown headings, then blank-line paragraphs, then sentences,
    then a hard character cut. A section that already fits the budget is kept whole,
    which is the point — a heading and its body answer a question together.
    """

    name = "structural"

    def __init__(self, max_chars: int = 900, min_section_chars: int = 200) -> None:
        if max_chars <= 0:
            raise ChunkingError("max_chars must be positive")
        self.max_chars = max_chars
        self.min_section_chars = min_section_chars

    def boundaries(self, text: str) -> list[Span]:
        sections = self._sections(text)
        spans: list[Span] = []
        for section in sections:
            spans.extend(self._split_span(text, section))
        return spans

    def _sections(self, text: str) -> list[Span]:
        offsets = [heading.offset for heading in heading_index(text)]
        if not offsets or offsets[0] != 0:
            offsets.insert(0, 0)
        merged: list[Span] = []
        for index, offset in enumerate(offsets):
            end = offsets[index + 1] if index + 1 < len(offsets) else len(text)
            if end <= offset:
                continue
            # Fold a heading with almost no body into the next section: a lone "## API"
            # chunk retrieves for every API question and answers none of them.
            too_small = end - offset < min(self.min_section_chars, self.max_chars)
            if merged and too_small and len(merged[-1]) + (end - offset) <= self.max_chars:
                merged[-1] = Span(merged[-1].start, end)
                continue
            merged.append(Span(offset, end))
        return merged or [Span(0, len(text))]

    def _split_span(self, text: str, span: Span) -> list[Span]:
        if len(span) <= self.max_chars:
            return [span]
        for candidate in (self._paragraph_starts, self._sentence_starts):
            groups = self._pack(span, candidate(text, span))
            if len(groups) > 1:
                result: list[Span] = []
                for group in groups:
                    result.extend(
                        [group] if len(group) <= self.max_chars else self._hard_split(group)
                    )
                return result
        return self._hard_split(span)

    def _paragraph_starts(self, text: str, span: Span) -> list[int]:
        starts = [span.start]
        for match in _PARAGRAPH_RE.finditer(text, span.start, span.end):
            if starts[-1] < match.end() < span.end:
                starts.append(match.end())
        return starts

    def _sentence_starts(self, text: str, span: Span) -> list[int]:
        body = text[span.start : span.end]
        return [span.start + sentence.start for sentence in sentence_spans(body)]

    def _pack(self, span: Span, starts: list[int]) -> list[Span]:
        """Greedily group consecutive separator-delimited blocks under the budget."""
        if len(starts) < 2:
            return [span]
        bounds = [*starts, span.end]
        spans: list[Span] = []
        current_start = bounds[0]
        for index in range(1, len(bounds)):
            if bounds[index] - current_start > self.max_chars and bounds[index - 1] > current_start:
                spans.append(Span(current_start, bounds[index - 1]))
                current_start = bounds[index - 1]
        spans.append(Span(current_start, span.end))
        return spans

    def _hard_split(self, span: Span) -> list[Span]:
        spans: list[Span] = []
        start = span.start
        while start < span.end:
            end = min(start + self.max_chars, span.end)
            spans.append(Span(start, end))
            start = end
        return spans

    @property
    def min_chars(self) -> int:
        return 0


class TokenChunker(Chunker):
    """Packs a fixed number of tokens with a token-expressed overlap.

    The tokenizer here is the regex word tokenizer from :mod:`rag.text`, not a model's
    BPE. That is deliberate: the repository must run offline and byte-identically in
    CI, and a BPE vocabulary is a download. The counts are therefore a proxy — plug a
    real tokenizer in if you need to respect a hard context limit exactly.
    """

    name = "token"

    def __init__(self, max_tokens: int = 180, overlap_tokens: int = 30) -> None:
        if max_tokens <= 0:
            raise ChunkingError("max_tokens must be positive")
        if not 0 <= overlap_tokens < max_tokens:
            raise ChunkingError("overlap_tokens must be in [0, max_tokens)")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def boundaries(self, text: str) -> list[Span]:
        tokens = word_spans(text)
        if len(tokens) <= self.max_tokens:
            return [Span(0, len(text))]

        step = self.max_tokens - self.overlap_tokens
        spans: list[Span] = []
        index = 0
        while index < len(tokens):
            last = min(index + self.max_tokens, len(tokens)) - 1
            start = 0 if index == 0 else tokens[index].start
            end = len(text) if last == len(tokens) - 1 else tokens[last].end
            spans.append(Span(start, end))
            if last == len(tokens) - 1:
                break
            index += step
        return spans

    @property
    def min_chars(self) -> int:
        return 0


def build_chunker(strategy: str, options: dict[str, int]) -> Chunker:
    """Factory used by the config loader and the benchmark sweep."""
    match strategy:
        case "fixed":
            return FixedSizeChunker(
                size=options.get("size", 800), overlap=options.get("overlap", 120)
            )
        case "sentence":
            return SentenceChunker(
                max_chars=options.get("max_chars", 800),
                overlap_sentences=options.get("overlap_sentences", 1),
            )
        case "structural":
            return StructuralChunker(
                max_chars=options.get("max_chars", 900),
                min_section_chars=options.get("min_section_chars", 200),
            )
        case "token":
            return TokenChunker(
                max_tokens=options.get("max_tokens", 180),
                overlap_tokens=options.get("overlap_tokens", 30),
            )
        case _:
            raise ChunkingError(f"unknown chunking strategy: {strategy!r}")


def reconstruct(chunks: list[Chunk]) -> str:
    """Rebuild the source text from chunks alone, removing overlap once.

    Used by the tests as the strongest available statement of chunker correctness: if
    this returns the original document for every strategy and every configuration, no
    text was dropped and none was duplicated into the index.
    """
    if not chunks:
        return ""
    ordered = sorted(chunks, key=lambda chunk: chunk.start)
    text = ordered[0].text
    consumed = ordered[0].end
    for chunk in ordered[1:]:
        if chunk.end <= consumed:
            continue
        text += chunk.text[consumed - chunk.start :]
        consumed = chunk.end
    return text
