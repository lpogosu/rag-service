from __future__ import annotations

import pytest

from rag.config import AppConfig, load_config
from rag.pipeline import RagPipeline
from rag.text import Analyzer
from rag.types import Chunk, Document, Metadata

MARKDOWN_DOCUMENT = """# Конфигурация брокера

Справочник параметров broker.yaml.

## Хранение

Параметр retention.hours задаёт срок хранения событий и по умолчанию равен 168 часам.
Параметр segment.size.mb определяет размер сегмента и равен 512 мегабайтам.

Уменьшение segment.size.mb ускоряет удаление старых данных.

## Сеть

Брокер принимает клиентские подключения на порту 7420. Метрики отдаются на порту 9420.

Незашифрованные соединения не принимаются: поддерживается только TLS 1.3.
"""

ENGLISH_DOCUMENT = """# Kestrel HTTP API

All administrative endpoints live under the /v1 prefix.

## Endpoints

POST /v1/topics creates a topic. GET /v1/topics/{name}/offsets returns partition offsets.

Every response carries the X-Kestrel-Quota-Remaining header.
"""


@pytest.fixture
def analyzer() -> Analyzer:
    """Analyzer with no stopwords and no stemming, so test arithmetic stays checkable."""
    return Analyzer(stopwords=frozenset(), min_token_length=1, use_stemming=False)


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(doc_id="broker-config", text=MARKDOWN_DOCUMENT, metadata={"lang": "ru"}),
        Document(doc_id="http-api", text=ENGLISH_DOCUMENT, metadata={"lang": "en"}),
    ]


@pytest.fixture
def offline_config() -> AppConfig:
    return load_config("config/offline.yaml")


@pytest.fixture
def pipeline(offline_config: AppConfig, documents: list[Document]) -> RagPipeline:
    built = RagPipeline.from_config(offline_config)
    built.index(documents)
    return built


def make_chunk(
    chunk_id: str,
    text: str,
    doc_id: str = "doc",
    ordinal: int = 0,
    metadata: Metadata | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        start=0,
        end=len(text),
        ordinal=ordinal,
        metadata=dict(metadata or {}),
    )
