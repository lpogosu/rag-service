"""Stemmer conformance.

The expected values below are the output of the reference Snowball implementations
(``snowballstemmer``, algorithms "russian" and "porter"). That library is not a
dependency of this project — it was used once, during development, to check these
implementations against 245 739 Russian word forms and 23 662 English words, with no
divergence. What is pinned here is a readable subset of that comparison, so a
regression in the suffix tables is caught without adding a dependency to run the tests.
"""

from __future__ import annotations

import pytest

from rag.stemming import stem_english, stem_russian
from rag.text import stem

RUSSIAN_CASES = {
    "настройка": "настройк",
    "настройки": "настройк",
    "настройке": "настройк",
    "настройкой": "настройк",
    "конфигурации": "конфигурац",
    "конфигурация": "конфигурац",
    "сервисов": "сервис",
    "сервисами": "сервис",
    "работающий": "работа",
    "работающая": "работа",
    "читаемость": "читаем",
    "читаемости": "читаем",
    "брокером": "брокер",
    "брокеру": "брокер",
    "красивейший": "красив",
    "данные": "дан",
    "данных": "дан",
    "включенный": "включен",
    "развертывание": "развертыван",
    "партиции": "партиц",
    "партициями": "партиц",
    "хранения": "хранен",
    "хранение": "хранен",
    "дал": "дал",
    "дать": "дат",
    "прав": "прав",
}

ENGLISH_CASES = {
    "documents": "document",
    "document": "document",
    "retrieval": "retriev",
    "retrieved": "retriev",
    "retrieving": "retriev",
    "chunking": "chunk",
    "embeddings": "embed",
    "queries": "queri",
    "generation": "gener",
    "relational": "relat",
    "conditional": "condit",
    "rational": "ration",
    "technology": "technologi",
    "possibly": "possibli",
    "happy": "happi",
    "sky": "sky",
    "running": "run",
    "indexes": "index",
    "caresses": "caress",
    "ponies": "poni",
    "agreed": "agre",
    "conflated": "conflat",
}


@pytest.mark.parametrize(("word", "expected"), sorted(RUSSIAN_CASES.items()))
def test_russian_stemmer_matches_snowball(word: str, expected: str) -> None:
    assert stem_russian(word) == expected


@pytest.mark.parametrize(("word", "expected"), sorted(ENGLISH_CASES.items()))
def test_english_stemmer_matches_porter(word: str, expected: str) -> None:
    assert stem_english(word) == expected


def test_inflections_of_one_lemma_collapse_to_one_term() -> None:
    """The property BM25 actually depends on: a query form matches a document form."""
    forms = ["настройка", "настройки", "настройке", "настройкой", "настройками"]
    assert len({stem_russian(form) for form in forms}) == 1


def test_stem_routes_by_script() -> None:
    assert stem("брокеров") == "брокер"
    assert stem("brokers") == "broker"
    assert stem("7420") == "7420"


def test_short_words_are_left_alone_in_english() -> None:
    for word in ("io", "os", "id"):
        assert stem_english(word) == word
