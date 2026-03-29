"""Tokenisation, sentence segmentation and the analyzer."""

from __future__ import annotations

import pytest

from rag.text import Analyzer, fold, sentence_spans, snippet, tokenize, word_spans


def sentences(text: str) -> list[str]:
    return [text[span.start : span.end] for span in sentence_spans(text)]


def test_sentence_spans_tile_the_text_without_gaps() -> None:
    """The chunkers' reconstruction guarantee starts here."""
    text = "Первое. Второе! Третье?  Хвост без точки"
    spans = sentence_spans(text)
    assert spans[0].start == 0
    assert spans[-1].end == len(text)
    assert "".join(sentences(text)) == text


def test_sentences_split_on_terminal_punctuation() -> None:
    assert sentences("Раз. Два! Три?") == ["Раз. ", "Два! ", "Три?"]


def test_a_decimal_number_does_not_end_a_sentence() -> None:
    assert len(sentences("Версия 2.6 вышла в июне. Следующая позже.")) == 2


def test_a_known_abbreviation_does_not_end_a_sentence() -> None:
    assert len(sentences("Параметры и т.д. описаны ниже. Конец.")) == 2
    assert len(sentences("Use POST, e.g. /v1/topics for creation. Done.")) == 2


def test_a_lowercase_continuation_does_not_end_a_sentence() -> None:
    assert len(sentences("Файл broker.yaml читается при старте. Затем всё.")) == 2


def test_a_blank_line_ends_a_sentence_even_without_punctuation() -> None:
    assert len(sentences("## Заголовок\n\nТекст после заголовка")) == 2


def test_empty_text_has_no_sentences() -> None:
    assert sentence_spans("") == []


def test_folding_is_case_and_yo_insensitive() -> None:
    assert fold("Ещё РАЗ") == "еще раз"


def test_tokenize_keeps_numbers_and_drops_punctuation() -> None:
    assert tokenize("Порт 7420, версия 2.6 — TLS!") == ["порт", "7420", "версия", "2.6", "tls"]


def test_word_spans_point_into_the_original_text() -> None:
    text = "Порт 7420."
    assert [text[span.start : span.end] for span in word_spans(text)] == ["Порт", "7420"]


def test_analyzer_removes_stopwords_and_stems() -> None:
    terms = Analyzer().analyze("Какой срок хранения событий в брокере?")
    assert "как" not in terms
    assert "хранен" in terms
    assert "брокер" in terms


def test_analyzer_can_be_told_not_to_stem() -> None:
    terms = Analyzer(use_stemming=False).analyze("хранения событий")
    assert terms == ["хранения", "событий"]


def test_analyzer_drops_tokens_below_the_length_floor() -> None:
    assert Analyzer(min_token_length=4).analyze("порт 74 сервер") == ["порт", "сервер"]


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        ("  много   пробелов  ", 50, "много пробелов"),
        ("а" * 30, 10, "а" * 9 + "…"),
    ],
)
def test_snippet_collapses_whitespace_and_truncates(text: str, limit: int, expected: str) -> None:
    assert snippet(text, limit) == expected
