"""Tokenisation, sentence segmentation and query/document analysis.

Everything here works on character offsets rather than on copies of the text, because
the chunkers and the evaluation gold spans both need to point back into the original
document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.stemming import stem_english, stem_russian

_WORD_RE = re.compile(r"[a-z]+(?:['`][a-z]+)*|[а-я]+|\d+(?:[.,]\d+)*")
_CYRILLIC_RE = re.compile(r"[а-я]")

_SENTENCE_END = ".!?…"

# Abbreviations that end in a period without ending a sentence. Kept short on purpose:
# a long list is a maintenance burden and the chunkers degrade gracefully when a
# sentence boundary is missed — the chunk simply gets a little longer.
_ABBREVIATIONS = frozenset(
    {
        "т.е",
        "т.д",
        "т.п",
        "т.к",
        "др",
        "рис",
        "см",
        "напр",
        "гг",
        "стр",
        "п",
        "пп",
        "e.g",
        "i.e",
        "etc",
        "vs",
        "fig",
        "no",
        "cf",
        "approx",
    }
)

RUSSIAN_STOPWORDS = frozenset(
    [
        "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она",
        "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее",
        "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда",
        "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до",
        "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
        "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем",
        "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет",
        "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним",
        "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были",
        "куда", "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой",
        "хоть", "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них",
        "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
        "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда",
        "конечно", "всю", "между"
    ]
)

ENGLISH_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is",
        "it", "its", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
        "there", "these", "they", "this", "to", "was", "will", "with", "from", "your", "you",
        "we", "our", "have", "has", "had", "can", "could", "should", "would", "about", "which",
        "when", "where", "who", "whom", "how", "what"
    ]
)

STOPWORDS = RUSSIAN_STOPWORDS | ENGLISH_STOPWORDS


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end


def fold(text: str) -> str:
    """Lowercase and fold "ё" to "е".

    Search must not care which of the two spellings the author used, and the Snowball
    Russian stemmer is defined over the folded alphabet.
    """
    return text.lower().replace("ё", "е").replace("Ё", "е")


def word_spans(text: str) -> list[Span]:
    """Word offsets in the original text, in order."""
    folded = fold(text)
    return [Span(match.start(), match.end()) for match in _WORD_RE.finditer(folded)]


def tokenize(text: str) -> list[str]:
    """Surface tokens: lowercase words and numbers, no stemming, no stopword removal."""
    return _WORD_RE.findall(fold(text))


def stem(token: str) -> str:
    """Route a token to the stemmer matching its script."""
    if _CYRILLIC_RE.search(token):
        return stem_russian(token)
    if token.isdigit():
        return token
    return stem_english(token)


@dataclass(frozen=True, slots=True)
class Analyzer:
    """Turns raw text into the term list an index or a scorer works with.

    Kept as data rather than behaviour so that an index can be rebuilt with a different
    analyzer and the change is visible in the config, not buried in a constructor.
    """

    stopwords: frozenset[str] = STOPWORDS
    min_token_length: int = 2
    use_stemming: bool = True

    def analyze(self, text: str) -> list[str]:
        terms: list[str] = []
        for token in tokenize(text):
            if token in self.stopwords or len(token) < self.min_token_length:
                continue
            terms.append(stem(token) if self.use_stemming else token)
        return terms


def _is_abbreviation(text: str, period_index: int) -> bool:
    start = period_index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    candidate = fold(text[start:period_index])
    return candidate in _ABBREVIATIONS


def sentence_spans(text: str) -> list[Span]:
    """Split into sentences, returning offsets that tile the text with no gaps.

    Consecutive spans are contiguous (``spans[i].end == spans[i + 1].start``) and the
    last one ends at ``len(text)``, so trailing whitespace and newlines belong to the
    sentence that precedes them. The chunkers rely on that to keep documents
    reconstructible.
    """
    if not text:
        return []

    boundaries: list[int] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in _SENTENCE_END:
            if char == "." and _is_abbreviation(text, index):
                index += 1
                continue
            if char == "." and index + 1 < length and text[index + 1].isdigit():
                index += 1
                continue
            end = index + 1
            while end < length and text[end] in _SENTENCE_END + '"»)\'':
                end += 1
            while end < length and text[end] in " \t":
                end += 1
            if end >= length:
                break
            if text[end] == "\n":
                index = end
                continue
            if text[end].islower():
                index = end
                continue
            boundaries.append(end)
            index = end
            continue
        if char == "\n":
            end = index
            while end < length and text[end] in " \t\n":
                end += 1
            # A blank line always ends a sentence, even without punctuation: headings
            # and list items in documentation frequently have none.
            if text.count("\n", index, end) >= 2 and end < length:
                boundaries.append(end)
            index = end if end > index else index + 1
            continue
        index += 1

    spans: list[Span] = []
    previous = 0
    for boundary in boundaries:
        if boundary > previous:
            spans.append(Span(previous, boundary))
            previous = boundary
    spans.append(Span(previous, length))
    return spans


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def snippet(text: str, limit: int = 200) -> str:
    collapsed = collapse_whitespace(text)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"
