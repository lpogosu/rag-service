"""Chunker correctness.

The central test is reconstruction: rebuilding the document from the chunks alone and
comparing it byte-for-byte with the source. It catches the two failures that matter and
are otherwise invisible — text that never made it into any chunk (the answer is
unreachable) and text duplicated across chunks (it wins every retrieval by repetition).
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import pytest

from rag.chunking import (
    Chunker,
    ChunkingError,
    FixedSizeChunker,
    SentenceChunker,
    StructuralChunker,
    TokenChunker,
    build_chunker,
    heading_index,
    reconstruct,
)
from rag.types import Document
from tests.conftest import ENGLISH_DOCUMENT, MARKDOWN_DOCUMENT

TEXTS = [
    MARKDOWN_DOCUMENT,
    ENGLISH_DOCUMENT,
    "Одно предложение без заголовков и переводов строки.",
    "# Только заголовок\n",
    "a" * 2500,
    "Строка.\n\nВторая.\n\n\nТретья с хвостом  \n",
]

CHUNKERS: list[Chunker] = [
    FixedSizeChunker(size=200, overlap=50),
    FixedSizeChunker(size=64, overlap=0),
    FixedSizeChunker(size=1000, overlap=300),
    SentenceChunker(max_chars=180, overlap_sentences=1),
    SentenceChunker(max_chars=90, overlap_sentences=0),
    SentenceChunker(max_chars=5, overlap_sentences=2),
    StructuralChunker(max_chars=300, min_section_chars=100),
    StructuralChunker(max_chars=40, min_section_chars=0),
    TokenChunker(max_tokens=20, overlap_tokens=5),
    TokenChunker(max_tokens=3, overlap_tokens=2),
]


@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: f"{c.name}-{id(c) % 1000}")
@pytest.mark.parametrize("text", TEXTS, ids=lambda t: f"text{len(t)}")
def test_chunks_reconstruct_the_document(chunker: Chunker, text: str) -> None:
    document = Document(doc_id="d", text=text)
    chunks = chunker.split(document)
    assert chunks, "a non-empty document must produce at least one chunk"
    assert reconstruct(chunks) == text


@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: f"{c.name}-{id(c) % 1000}")
def test_spans_cover_the_document_without_gaps(chunker: Chunker) -> None:
    document = Document(doc_id="d", text=MARKDOWN_DOCUMENT)
    chunks = chunker.split(document)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(MARKDOWN_DOCUMENT)
    for previous, current in pairwise(chunks):
        assert current.start <= previous.end, "gap between chunks"
        assert current.start > previous.start
        assert current.end > previous.end
        assert current.text == MARKDOWN_DOCUMENT[current.start : current.end]


def test_fixed_overlap_repeats_exactly_the_requested_characters() -> None:
    text = "".join(str(index % 10) for index in range(1000))
    chunks = FixedSizeChunker(size=300, overlap=60).split(Document(doc_id="d", text=text))
    first, second = chunks[0], chunks[1]
    assert first.end - second.start == 60
    assert first.text[-60:] == second.text[:60]


def test_zero_overlap_produces_disjoint_chunks() -> None:
    text = "".join(str(index % 10) for index in range(500))
    chunks = FixedSizeChunker(size=100, overlap=0).split(Document(doc_id="d", text=text))
    for previous, current in pairwise(chunks):
        assert current.start == previous.end
    assert "".join(chunk.text for chunk in chunks) == text


def test_sentence_chunker_does_not_split_mid_sentence() -> None:
    text = (
        "Первое предложение достаточно длинное. Второе предложение тоже. "
        "Третье предложение завершает абзац."
    )
    chunks = SentenceChunker(max_chars=70, overlap_sentences=0).split(
        Document(doc_id="d", text=text)
    )
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.text.rstrip().endswith(".")


def test_sentence_overlap_repeats_the_previous_sentence() -> None:
    text = "Раз один. Два два. Три три. Четыре четыре. Пять пять."
    overlapping = SentenceChunker(max_chars=30, overlap_sentences=1).split(
        Document(doc_id="d", text=text)
    )
    disjoint = SentenceChunker(max_chars=30, overlap_sentences=0).split(
        Document(doc_id="d", text=text)
    )
    assert len(overlapping) > 2
    # With one sentence of overlap every chunk after the first starts inside its
    # predecessor, and the shared text is a whole sentence.
    for previous, current in pairwise(overlapping):
        assert current.start < previous.end
        assert previous.text[current.start - previous.start :].strip().endswith(".")
    assert all(current.start == previous.end for previous, current in pairwise(disjoint))


def test_structural_chunker_keeps_a_small_section_whole() -> None:
    chunks = StructuralChunker(max_chars=2000).split(
        Document(doc_id="d", text=MARKDOWN_DOCUMENT)
    )
    storage = [chunk for chunk in chunks if "retention.hours" in chunk.text]
    assert len(storage) == 1
    assert "segment.size.mb" in storage[0].text


def test_chunks_carry_the_heading_in_effect_at_their_start() -> None:
    chunks = StructuralChunker(max_chars=300).split(Document(doc_id="d", text=MARKDOWN_DOCUMENT))
    with_retention = next(chunk for chunk in chunks if "retention.hours" in chunk.text)
    assert with_retention.heading == "Конфигурация брокера > Хранение"
    assert with_retention.embedding_text().startswith("Конфигурация брокера > Хранение\n")


def test_heading_index_builds_the_full_path() -> None:
    text = "# A\ntext\n## B\ntext\n### C\ntext\n## D\ntext\n"
    paths = [heading.path for heading in heading_index(text)]
    assert paths == ["A", "A > B", "A > B > C", "A > D"]


def test_token_chunker_respects_the_token_budget() -> None:
    text = " ".join(f"слово{index}" for index in range(200))
    chunks = TokenChunker(max_tokens=25, overlap_tokens=5).split(Document(doc_id="d", text=text))
    assert len(chunks) > 5
    for chunk in chunks:
        assert len(chunk.text.split()) <= 25


def test_empty_document_produces_no_chunks() -> None:
    assert FixedSizeChunker().split(Document(doc_id="d", text="   \n\n  ")) == []


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: FixedSizeChunker(size=100, overlap=100), "overlap"),
        (lambda: FixedSizeChunker(size=0), "size"),
        (lambda: TokenChunker(max_tokens=10, overlap_tokens=10), "overlap_tokens"),
        (lambda: SentenceChunker(max_chars=0), "max_chars"),
    ],
)
def test_invalid_configuration_is_rejected(factory: Callable[[], Chunker], message: str) -> None:
    with pytest.raises(ChunkingError, match=message):
        factory()


def test_build_chunker_rejects_an_unknown_strategy() -> None:
    with pytest.raises(ChunkingError, match="unknown chunking strategy"):
        build_chunker("semantic", {})


def test_build_chunker_applies_options() -> None:
    chunker = build_chunker("fixed", {"size": 123, "overlap": 7})
    assert isinstance(chunker, FixedSizeChunker)
    assert (chunker.size, chunker.overlap) == (123, 7)


def test_chunk_ids_are_stable_and_ordered() -> None:
    document = Document(doc_id="broker", text=MARKDOWN_DOCUMENT)
    first = StructuralChunker(max_chars=300).split(document)
    second = StructuralChunker(max_chars=300).split(document)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == "broker::0000"
    assert [chunk.ordinal for chunk in first] == list(range(len(first)))
