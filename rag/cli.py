"""Command line entry point: index documents, then ask questions.

Exists so that the pipeline can be exercised without starting the API, which is what
you want when you are debugging retrieval rather than HTTP.

    python -m rag.cli index --config config/offline.yaml --corpus
    python -m rag.cli query --config config/offline.yaml "какой порт у брокера?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.config import load_config
from rag.pipeline import RagPipeline
from rag.types import Document

CORPUS_PATH = Path(__file__).resolve().parent.parent / "eval" / "datasets" / "corpus.jsonl"
TEXT_SUFFIXES = frozenset({".md", ".txt"})


def read_directory(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        documents.append(
            Document(
                doc_id=path.relative_to(directory).as_posix(),
                text=text,
                metadata={"source": str(directory), "kind": path.suffix.lstrip(".")},
            )
        )
    return documents


def read_corpus(path: Path = CORPUS_PATH) -> list[Document]:
    """Load the evaluation corpus, so the CLI has something to index out of the box."""
    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            documents.append(
                Document(doc_id=row["doc_id"], text=row["text"], metadata=row["metadata"])
            )
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(prog="rag.cli", description="rag-service command line")
    parser.add_argument("--config", default="config/offline.yaml")
    subcommands = parser.add_subparsers(dest="command", required=True)

    index_parser = subcommands.add_parser("index", help="index documents")
    source = index_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", type=Path, help="directory with .md/.txt files")
    source.add_argument("--corpus", action="store_true", help="index the evaluation corpus")

    query_parser = subcommands.add_parser("query", help="index, then ask one question")
    query_parser.add_argument("question")
    query_parser.add_argument("--path", type=Path, help="directory to index first")

    arguments = parser.parse_args()
    pipeline = RagPipeline.from_config(load_config(arguments.config))
    try:
        if arguments.command == "index":
            documents = read_corpus() if arguments.corpus else read_directory(arguments.path)
            if not documents:
                print("nothing to index", file=sys.stderr)
                return 1
            print(json.dumps(pipeline.index(documents).to_dict(), ensure_ascii=False))
            return 0

        documents = read_directory(arguments.path) if arguments.path else read_corpus()
        pipeline.index(documents)
        result = pipeline.query(arguments.question)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
