"""The evaluation corpus must stay reproducible and its labels must stay correct.

If the committed JSONL and the generator ever disagree, every number in the README
becomes unverifiable. This is the test that keeps them honest.
"""

from __future__ import annotations

from eval.datasets.generate import DEFAULT_SEED, build
from eval.harness import load_documents, load_questions, load_unanswerable


def test_generation_is_deterministic_for_a_given_seed() -> None:
    first, _ = build(DEFAULT_SEED)
    second, _ = build(DEFAULT_SEED)
    assert [document.text for document in first.documents] == [
        document.text for document in second.documents
    ]


def test_committed_files_match_what_the_generator_produces() -> None:
    """Regenerating must be a no-op; otherwise the published numbers drift."""
    corpus, unanswerable = build(DEFAULT_SEED)
    on_disk = {document.doc_id: document.text for document in load_documents()}
    assert on_disk == {document.doc_id: document.text for document in corpus.documents}
    assert len(load_questions()) == len(corpus.questions)
    assert len(load_unanswerable()) == len(unanswerable)


def test_every_gold_span_points_at_the_sentence_that_answers_the_question() -> None:
    documents = {document.doc_id: document.text for document in load_documents()}
    for question in load_questions():
        text = documents[question.gold.doc_id]
        span = text[question.gold.start : question.gold.end]
        assert span.strip()
        assert text.count(span) == 1, f"{question.qid}: gold sentence is ambiguous"


def test_the_corpus_carries_enough_distractors_to_be_a_test() -> None:
    documents = load_documents()
    neighbours = [d for d in documents if d.metadata["kind"] == "neighbour"]
    assert len(neighbours) >= 10
    assert sum(len(document.text) for document in neighbours) > sum(
        len(document.text) for document in documents if document.metadata["kind"] != "neighbour"
    ) / 2


def test_both_languages_are_present() -> None:
    languages = {document.metadata["lang"] for document in load_documents()}
    assert languages == {"ru", "en"}


def test_unanswerable_questions_have_no_gold_document() -> None:
    answerable = {question.question for question in load_questions()}
    assert not answerable & set(load_unanswerable())
