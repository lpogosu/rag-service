"""Porter (English) and Snowball (Russian) stemmers, implemented here on purpose.

BM25 over Russian text without morphological normalisation is close to useless:
"настройка", "настройки" and "настройке" are three unrelated terms to an exact-match
index, so a query almost never hits the form the document happens to use. The usual
fix is to pull in a full morphological analyser, which is a 15 MB dictionary and a
noticeable per-token cost.

Suffix stripping gets most of the recall for none of that. The two algorithms below
are the published ones (Porter 1980 for English, the Snowball Russian stemmer), and
``tests/test_stemming.py`` pins them against reference outputs. See the README section
"Решения и компромиссы" for what stemming loses compared to lemmatisation.
"""

from __future__ import annotations

_ENGLISH_VOWELS = frozenset("aeiou")

_STEP2_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
)

_STEP3_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)

_STEP4_SUFFIXES: tuple[str, ...] = (
    "ement",
    "ance",
    "ence",
    "able",
    "ible",
    "ment",
    "ant",
    "ent",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
    "ion",
    "al",
    "er",
    "ic",
    "ou",
)


class _PorterWord:
    """Mutable cursor over a word, mirroring the structure of the published algorithm.

    The algorithm is defined in terms of "the stem before the suffix", so carrying the
    suffix start index (``j``) explicitly is clearer than re-slicing at every rule.
    """

    __slots__ = ("chars", "j")

    def __init__(self, word: str) -> None:
        self.chars = list(word)
        self.j = 0

    def __len__(self) -> int:
        return len(self.chars)

    def word(self) -> str:
        return "".join(self.chars)

    def is_consonant(self, index: int) -> bool:
        letter = self.chars[index]
        if letter in _ENGLISH_VOWELS:
            return False
        if letter == "y":
            return index == 0 or not self.is_consonant(index - 1)
        return True

    def measure(self) -> int:
        """Number of vowel-consonant sequences in ``chars[:j]``."""
        index = 0
        limit = self.j
        while index < limit and self.is_consonant(index):
            index += 1
        count = 0
        while index < limit:
            while index < limit and not self.is_consonant(index):
                index += 1
            if index >= limit:
                break
            count += 1
            index += 1
            while index < limit and self.is_consonant(index):
                index += 1
        return count

    def stem_has_vowel(self) -> bool:
        return any(not self.is_consonant(index) for index in range(self.j))

    def ends_double_consonant(self) -> bool:
        size = len(self.chars)
        if size < 2:
            return False
        return self.chars[-1] == self.chars[-2] and self.is_consonant(size - 1)

    def ends_cvc(self) -> bool:
        """True when the word ends consonant-vowel-consonant, last letter not w/x/y."""
        size = len(self.chars)
        if size < 3:
            return False
        if not (
            self.is_consonant(size - 3)
            and not self.is_consonant(size - 2)
            and self.is_consonant(size - 1)
        ):
            return False
        return self.chars[-1] not in "wxy"

    def ends_with(self, suffix: str) -> bool:
        if len(suffix) > len(self.chars):
            return False
        if self.word().endswith(suffix):
            self.j = len(self.chars) - len(suffix)
            return True
        return False

    def replace_suffix(self, replacement: str) -> None:
        self.chars = self.chars[: self.j] + list(replacement)


def _porter_step1ab(word: _PorterWord) -> None:
    if word.chars[-1] == "s":
        if word.ends_with("sses") or word.ends_with("ies"):
            word.chars = word.chars[:-2]
        elif len(word.chars) >= 2 and word.chars[-2] != "s":
            word.chars = word.chars[:-1]

    if word.ends_with("eed"):
        if word.measure() > 0:
            word.chars = word.chars[:-1]
        return

    stripped = False
    for suffix in ("ed", "ing"):
        if word.ends_with(suffix) and word.stem_has_vowel():
            word.chars = word.chars[: word.j]
            stripped = True
            break
    if not stripped:
        return

    if word.ends_with("at") or word.ends_with("bl") or word.ends_with("iz"):
        word.chars.append("e")
    elif word.ends_double_consonant() and word.chars[-1] not in "lsz":
        word.chars = word.chars[:-1]
    else:
        word.j = len(word.chars)
        if word.measure() == 1 and word.ends_cvc():
            word.chars.append("e")


def _porter_step1c(word: _PorterWord) -> None:
    if word.ends_with("y") and word.stem_has_vowel():
        word.chars[-1] = "i"


def _porter_step2(word: _PorterWord) -> None:
    for suffix, replacement in _STEP2_SUFFIXES:
        if word.ends_with(suffix):
            if word.measure() > 0:
                word.replace_suffix(replacement)
            return


def _porter_step3(word: _PorterWord) -> None:
    for suffix, replacement in _STEP3_SUFFIXES:
        if word.ends_with(suffix):
            if word.measure() > 0:
                word.replace_suffix(replacement)
            return


def _porter_step4(word: _PorterWord) -> None:
    for suffix in _STEP4_SUFFIXES:
        if not word.ends_with(suffix):
            continue
        if suffix == "ion" and (word.j == 0 or word.chars[word.j - 1] not in "st"):
            return
        if word.measure() > 1:
            word.replace_suffix("")
        return


def _porter_step5(word: _PorterWord) -> None:
    if word.chars[-1] == "e":
        word.j = len(word.chars) - 1
        measure = word.measure()
        if measure > 1:
            word.chars = word.chars[:-1]
        elif measure == 1:
            word.chars = word.chars[:-1]
            if word.ends_cvc():
                word.chars.append("e")
    if word.chars[-1] == "l" and word.ends_double_consonant():
        word.j = len(word.chars)
        if word.measure() > 1:
            word.chars = word.chars[:-1]


def stem_english(word: str) -> str:
    """Porter (1980) stemmer. Input must already be lowercased."""
    if len(word) <= 2:
        return word
    state = _PorterWord(word)
    _porter_step1ab(state)
    _porter_step1c(state)
    _porter_step2(state)
    _porter_step3(state)
    _porter_step4(state)
    _porter_step5(state)
    return state.word()


_RUSSIAN_VOWELS = frozenset("аеиоуыэюя")

_PERFECTIVE_GERUND_1: tuple[str, ...] = ("вшись", "вши", "в")
_PERFECTIVE_GERUND_2: tuple[str, ...] = ("ывшись", "ившись", "ывши", "ивши", "ыв", "ив")
_ADJECTIVE: tuple[str, ...] = (
    "иями",
    "ими",
    "ыми",
    "его",
    "ого",
    "ему",
    "ому",
    "ее",
    "ие",
    "ые",
    "ое",
    "ей",
    "ий",
    "ый",
    "ой",
    "ем",
    "им",
    "ым",
    "ом",
    "их",
    "ых",
    "ую",
    "юю",
    "ая",
    "яя",
    "ою",
    "ею",
)
_PARTICIPLE_1: tuple[str, ...] = ("ющ", "вш", "ем", "нн", "щ")
_PARTICIPLE_2: tuple[str, ...] = ("ующ", "ывш", "ивш")
_REFLEXIVE: tuple[str, ...] = ("ся", "сь")
_VERB_1: tuple[str, ...] = (
    "ешь",
    "нно",
    "ете",
    "йте",
    "ла",
    "на",
    "ли",
    "ем",
    "ло",
    "но",
    "ет",
    "ют",
    "ны",
    "ть",
    "й",
    "л",
    "н",
)
_VERB_2: tuple[str, ...] = (
    "ейте",
    "уйте",
    "ила",
    "ыла",
    "ена",
    "ите",
    "или",
    "ыли",
    "ило",
    "ыло",
    "ено",
    "ует",
    "уют",
    "ены",
    "ить",
    "ыть",
    "ишь",
    "ей",
    "уй",
    "ил",
    "ыл",
    "им",
    "ым",
    "ен",
    "ят",
    "ит",
    "ыт",
    "ую",
    "ю",
)
_NOUN: tuple[str, ...] = (
    "иями",
    "ями",
    "ами",
    "иях",
    "иям",
    "ием",
    "ией",
    "ях",
    "ям",
    "ах",
    "ов",
    "ев",
    "ие",
    "ье",
    "еи",
    "ии",
    "ей",
    "ой",
    "ий",
    "ем",
    "ом",
    "ам",
    "ию",
    "ью",
    "ия",
    "ья",
    "а",
    "е",
    "и",
    "й",
    "о",
    "у",
    "ы",
    "ь",
    "ю",
    "я",
)
_SUPERLATIVE: tuple[str, ...] = ("ейше", "ейш")
_DERIVATIONAL: tuple[str, ...] = ("ость", "ост")


def _russian_regions(word: str) -> tuple[int, int]:
    """Return the start offsets of RV and R2.

    RV starts after the first vowel; R1 after the first vowel-consonant pair; R2 is R1
    applied again inside R1. Only RV and R2 are used by the algorithm.
    """
    length = len(word)
    rv = length
    for index, letter in enumerate(word):
        if letter in _RUSSIAN_VOWELS:
            rv = index + 1
            break

    def after_vowel_consonant(start: int) -> int:
        index = start
        while index < length and word[index] not in _RUSSIAN_VOWELS:
            index += 1
        while index < length and word[index] in _RUSSIAN_VOWELS:
            index += 1
        return min(index + 1, length) if index < length else length

    r1 = after_vowel_consonant(0)
    r2 = after_vowel_consonant(r1) if r1 < length else length
    return rv, r2


def _strip_in_region(word: str, region_start: int, endings: tuple[str, ...]) -> str | None:
    for ending in endings:
        if word.endswith(ending) and len(word) - len(ending) >= region_start:
            return word[: len(word) - len(ending)]
    return None


def _strip_group1(word: str, rv: int, endings: tuple[str, ...]) -> str | None:
    """Endings that are only valid when preceded by "а" or "я".

    The preceding vowel must itself lie inside RV: Snowball narrows the search
    window to RV before matching, so the whole "а"+ending sequence has to fit. Dropping
    that detail turns "дал" into "да".
    """
    for ending in endings:
        if not word.endswith(ending):
            continue
        cut = len(word) - len(ending)
        if cut - 1 < rv:
            continue
        if word[cut - 1] in "ая":
            return word[:cut]
    return None


def _russian_step1(word: str, rv: int) -> str:
    gerund = _strip_group1(word, rv, _PERFECTIVE_GERUND_1)
    if gerund is None:
        gerund = _strip_in_region(word, rv, _PERFECTIVE_GERUND_2)
    if gerund is not None:
        return gerund

    reflexive = _strip_in_region(word, rv, _REFLEXIVE)
    if reflexive is not None:
        word = reflexive

    adjectival = _strip_in_region(word, rv, _ADJECTIVE)
    if adjectival is not None:
        participle = _strip_group1(adjectival, rv, _PARTICIPLE_1)
        if participle is None:
            participle = _strip_in_region(adjectival, rv, _PARTICIPLE_2)
        return participle if participle is not None else adjectival

    verb = _strip_group1(word, rv, _VERB_1)
    if verb is None:
        verb = _strip_in_region(word, rv, _VERB_2)
    if verb is not None:
        return verb

    noun = _strip_in_region(word, rv, _NOUN)
    return noun if noun is not None else word


def stem_russian(word: str) -> str:
    """Snowball Russian stemmer. Input must be lowercased, with "ё" folded to "е"."""
    rv, r2 = _russian_regions(word)

    word = _russian_step1(word, rv)

    if word.endswith("и") and len(word) - 1 >= rv:
        word = word[:-1]

    derivational = _strip_in_region(word, r2, _DERIVATIONAL)
    if derivational is not None:
        word = derivational

    if word.endswith("нн") and len(word) - 1 >= rv:
        return word[:-1]
    superlative = _strip_in_region(word, rv, _SUPERLATIVE)
    if superlative is not None:
        word = superlative
        if word.endswith("нн") and len(word) - 1 >= rv:
            word = word[:-1]
        return word
    if word.endswith("ь") and len(word) - 1 >= rv:
        return word[:-1]
    return word
