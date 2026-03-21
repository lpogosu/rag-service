"""Runs a configured pipeline over the evaluation corpus and prints a results table.

Two sweeps are built in, because they answer the two questions the README makes claims
about: does the chunking strategy matter, and does hybrid retrieval beat either half of
it. Both run entirely offline with the hashing embedder and the in-memory store, so the
numbers are reproducible on any machine with no model and no database.

    python -m eval.harness --sweep --json eval/results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eval.metrics import (
    GoldSpan,
    answer_term_recall,
    faithfulness,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_chunk_ids,
)
from rag.config import AppConfig, build_config, load_config
from rag.pipeline import RagPipeline
from rag.text import Analyzer
from rag.types import Document

DATA_DIR = Path(__file__).resolve().parent / "datasets"
REPORT_K = (1, 3, 5)


@dataclass(frozen=True, slots=True)
class Question:
    qid: str
    question: str
    answer: str
    gold: GoldSpan


@dataclass(frozen=True, slots=True)
class RunResult:
    """One row of the results table."""

    run: str
    chunker: str
    retrieval: str
    chunks: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float
    faithfulness: float
    answer_terms: float
    refused_answerable: float
    refused_unanswerable: float
    index_ms: float
    query_ms_p50: float
    query_ms_p95: float


def load_documents(directory: Path = DATA_DIR) -> list[Document]:
    return [
        Document(doc_id=row["doc_id"], text=row["text"], metadata=row["metadata"])
        for row in _read_jsonl(directory / "corpus.jsonl")
    ]


def load_questions(directory: Path = DATA_DIR) -> list[Question]:
    return [
        Question(
            qid=row["qid"],
            question=row["question"],
            answer=row["answer"],
            gold=GoldSpan(doc_id=row["doc_id"], start=row["gold_start"], end=row["gold_end"]),
        )
        for row in _read_jsonl(directory / "questions.jsonl")
    ]


def load_unanswerable(directory: Path = DATA_DIR) -> list[str]:
    return [row["question"] for row in _read_jsonl(directory / "unanswerable.jsonl")]


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def variant(base: AppConfig, name: str, **overrides: dict[str, Any]) -> AppConfig:
    """Produce a configuration that differs from ``base`` in the given sections."""
    payload = base.model_dump()
    payload["name"] = name
    for section, values in overrides.items():
        current = payload.get(section)
        payload[section] = {**current, **values} if isinstance(current, dict) else values
    return build_config(payload)


def evaluate(
    config: AppConfig,
    documents: Sequence[Document],
    questions: Sequence[Question],
    unanswerable: Sequence[str],
    *,
    retrieval_label: str,
) -> RunResult:
    pipeline = RagPipeline.from_config(config)
    analyzer = Analyzer()
    report = pipeline.index(documents)
    all_chunks = list(pipeline.store.iter_chunks())

    recalls: dict[int, list[float]] = {k: [] for k in REPORT_K}
    precisions: list[float] = []
    reciprocal: list[float] = []
    ndcg: list[float] = []
    grounding: list[float] = []
    answer_terms: list[float] = []
    refusals: list[float] = []
    latencies: list[float] = []
    invalid_citations = 0

    for question in questions:
        relevant = relevant_chunk_ids(question.gold, all_chunks)
        started = time.perf_counter()
        result = pipeline.query(question.question)
        latencies.append((time.perf_counter() - started) * 1000.0)
        ranked = [item.chunk.chunk_id for item in result.retrieved]
        for k in REPORT_K:
            recalls[k].append(recall_at_k(ranked, relevant, k))
        precisions.append(precision_at_k(ranked, relevant, 5))
        reciprocal.append(reciprocal_rank(ranked, relevant))
        ndcg.append(ndcg_at_k(ranked, relevant, 5))
        grounded = faithfulness(result.answer, [item.chunk for item in result.retrieved], analyzer)
        grounding.append(grounded.supported)
        if not grounded.citations_valid:
            invalid_citations += 1
        answer_terms.append(answer_term_recall(result.answer.text, question.answer, analyzer))
        refusals.append(1.0 if result.answer.refused else 0.0)

    if invalid_citations:
        raise AssertionError(
            f"{invalid_citations} answers cited a chunk that was not retrieved; "
            "citation enforcement is broken"
        )

    unanswerable_refusals = [
        1.0 if pipeline.query(question).answer.refused else 0.0 for question in unanswerable
    ]
    pipeline.close()

    return RunResult(
        run=config.name,
        chunker=report.chunker,
        retrieval=retrieval_label,
        chunks=report.chunks,
        recall_at_1=mean(recalls[1]),
        recall_at_3=mean(recalls[3]),
        recall_at_5=mean(recalls[5]),
        precision_at_5=mean(precisions),
        mrr=mean(reciprocal),
        ndcg_at_5=mean(ndcg),
        faithfulness=mean(grounding),
        answer_terms=mean(answer_terms),
        refused_answerable=mean(refusals),
        refused_unanswerable=mean(unanswerable_refusals),
        index_ms=report.millis,
        query_ms_p50=statistics.median(latencies) if latencies else 0.0,
        query_ms_p95=_percentile(latencies, 95),
    )


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100.0) * (len(ordered) - 1)))
    return ordered[index]


CHUNKING_PRESETS: dict[str, dict[str, Any]] = {
    "fixed-800": {"strategy": "fixed", "options": {"size": 800, "overlap": 120}},
    "sentence-800": {"strategy": "sentence", "options": {"max_chars": 800, "overlap_sentences": 1}},
    "structural-900": {
        "strategy": "structural",
        "options": {"max_chars": 900, "min_section_chars": 200},
    },
    "token-180": {"strategy": "token", "options": {"max_tokens": 180, "overlap_tokens": 30}},
}


def chunking_sweep(base: AppConfig) -> list[tuple[AppConfig, str]]:
    """Same retrieval, four chunkers with comparable budgets."""
    return [
        (variant(base, name, chunking=preset), "hybrid rrf")
        for name, preset in CHUNKING_PRESETS.items()
    ]


def retrieval_sweep(base: AppConfig) -> list[tuple[AppConfig, str]]:
    """Same chunker, five retrieval configurations."""
    return [
        (variant(base, "dense-only", lexical={"enabled": False}), "dense"),
        (variant(base, "bm25-only", retrieval={"dense_k": 0, "lexical_k": 60}), "bm25"),
        (variant(base, "hybrid-rrf"), "hybrid rrf"),
        (
            variant(
                base,
                "hybrid-weighted",
                retrieval={
                    "dense_k": base.retrieval.dense_k,
                    "lexical_k": base.retrieval.lexical_k,
                    "final_k": base.retrieval.final_k,
                    "fusion": {**base.retrieval.fusion.model_dump(), "method": "weighted"},
                },
            ),
            "hybrid min-max",
        ),
        (
            variant(base, "hybrid-rrf-rerank", rerank={"provider": "heuristic", "top_n": 20}),
            "hybrid rrf + rerank",
        ),
    ]


def markdown_table(rows: Sequence[RunResult], first_column: str) -> str:
    header = (
        f"| {first_column} | чанков | recall@1 | recall@3 | recall@5 | "
        "precision@5 | MRR | nDCG@5 | p50 мс | p95 мс |"
    )
    separator = "|" + "|".join(["---"] * 10) + "|"
    lines = [header, separator]
    for row in rows:
        label = row.retrieval if first_column == "режим поиска" else row.run
        lines.append(
            f"| {label} | {row.chunks} | {row.recall_at_1:.3f} | {row.recall_at_3:.3f} | "
            f"{row.recall_at_5:.3f} | {row.precision_at_5:.3f} | {row.mrr:.3f} | "
            f"{row.ndcg_at_5:.3f} | {row.query_ms_p50:.1f} | {row.query_ms_p95:.1f} |"
        )
    return "\n".join(lines)


def refusal_table(rows: Sequence[tuple[float, RunResult]]) -> str:
    header = (
        "| min_support | ложный отказ (ответ есть) | верный отказ (ответа нет) | "
        "recall@5 | термины ответа |"
    )
    lines = [header, "|" + "|".join(["---"] * 5) + "|"]
    for threshold, row in rows:
        lines.append(
            f"| {threshold:.2f} | {row.refused_answerable:.3f} | "
            f"{row.refused_unanswerable:.3f} | {row.recall_at_5:.3f} | {row.answer_terms:.3f} |"
        )
    return "\n".join(lines)


SUPPORT_THRESHOLDS = (0.0, 0.2, 0.34, 0.5, 0.67)


def refusal_sweep(base: AppConfig) -> list[tuple[float, AppConfig]]:
    """Same retrieval, a rising evidence requirement before an answer is allowed."""
    return [
        (
            threshold,
            variant(base, f"support-{threshold:.2f}", generation={"min_support": threshold}),
        )
        for threshold in SUPPORT_THRESHOLDS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness")
    parser.add_argument("--config", default="config/offline.yaml", help="base configuration")
    parser.add_argument("--sweep", action="store_true", help="run both comparison sweeps")
    parser.add_argument("--json", dest="json_path", default="", help="write raw results here")
    arguments = parser.parse_args()

    base = load_config(arguments.config)
    documents = load_documents()
    questions = load_questions()
    unanswerable = load_unanswerable()

    print(
        f"corpus: {len(documents)} документов, "
        f"{sum(len(document.text) for document in documents)} символов, "
        f"{len(questions)} вопросов с ответом, {len(unanswerable)} без ответа"
    )

    if not arguments.sweep:
        result = evaluate(base, documents, questions, unanswerable, retrieval_label="hybrid rrf")
        print()
        print(markdown_table([result], "конфигурация"))
        _write_json(arguments.json_path, {"single": [asdict(result)]})
        return

    chunking_results = [
        evaluate(config, documents, questions, unanswerable, retrieval_label=label)
        for config, label in chunking_sweep(base)
    ]
    best = max(chunking_results, key=lambda row: (row.ndcg_at_5, row.recall_at_5))
    tuned = variant(base, "retrieval-sweep", chunking=CHUNKING_PRESETS[best.run])
    retrieval_results = [
        evaluate(config, documents, questions, unanswerable, retrieval_label=label)
        for config, label in retrieval_sweep(tuned)
    ]

    refusal_results = [
        (
            threshold,
            evaluate(config, documents, questions, unanswerable, retrieval_label="hybrid rrf"),
        )
        for threshold, config in refusal_sweep(tuned)
    ]

    print()
    print("### Стратегии чанкинга (гибридный поиск, RRF, без переранжирования)")
    print(markdown_table(chunking_results, "стратегия"))
    print()
    print(f"### Режимы поиска (чанкер {best.run})")
    print(markdown_table(retrieval_results, "режим поиска"))
    print()
    print(f"### Порог поддержки и отказы (чанкер {best.run}, гибрид RRF)")
    print(refusal_table(refusal_results))
    _write_json(
        arguments.json_path,
        {
            "chunking": [asdict(row) for row in chunking_results],
            "retrieval": [asdict(row) for row in retrieval_results],
            "refusal": [
                {"min_support": threshold, **asdict(row)} for threshold, row in refusal_results
            ],
            "best_chunker": best.run,
        },
    )


def _write_json(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nraw results: {target}")


if __name__ == "__main__":
    main()
