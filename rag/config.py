"""Configuration: one YAML file describes the whole pipeline.

Every stage is selected by name, so swapping the chunker or turning reranking off is a
config edit and a restart, not a code change. That is the property the repository is
built to demonstrate, and the benchmark sweep depends on it: the harness builds each
row of the table by mutating this object, not by writing a variant pipeline.

``extra="forbid"`` everywhere is deliberate. A typo in a YAML key that is silently
ignored is how a service ends up running with defaults nobody chose.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")


class ConfigError(ValueError):
    pass


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkingConfig(_Base):
    strategy: Literal["fixed", "sentence", "structural", "token"] = "structural"
    options: dict[str, int] = Field(default_factory=dict)


class EmbeddingConfig(_Base):
    provider: Literal["hashing", "ollama"] = "hashing"
    model: str = "nomic-embed-text"
    dim: int = Field(default=512, gt=0)
    batch_size: int = Field(default=32, gt=0)
    seed: int = 20260214


class HnswConfig(_Base):
    """pgvector HNSW parameters. Ranges are the ones pgvector itself accepts."""

    m: int = Field(default=16, ge=2, le=100)
    ef_construction: int = Field(default=64, ge=4, le=1000)
    ef_search: int = Field(default=64, ge=1, le=1000)


class StoreConfig(_Base):
    backend: Literal["memory", "pgvector"] = "memory"
    metric: Literal["cosine", "l2"] = "cosine"
    dsn: str = ""
    table: str = "rag_chunks"
    hnsw: HnswConfig = Field(default_factory=HnswConfig)

    @model_validator(mode="after")
    def _check_dsn(self) -> StoreConfig:
        if self.backend == "pgvector" and not self.dsn:
            raise ConfigError("store.dsn is required when backend is pgvector")
        return self


class LexicalConfig(_Base):
    enabled: bool = True
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    use_stemming: bool = True
    min_token_length: int = Field(default=2, gt=0)


class FusionConfig(_Base):
    method: Literal["rrf", "weighted"] = "rrf"
    k: int = Field(default=60, gt=0)
    dense_weight: float = Field(default=0.5, ge=0)
    lexical_weight: float = Field(default=0.5, ge=0)


class RetrievalConfig(_Base):
    """Candidate counts. Setting one of the two ``k`` values to zero disables that
    retriever entirely, which is how the harness measures dense-only and BM25-only."""

    dense_k: int = Field(default=30, ge=0)
    lexical_k: int = Field(default=30, ge=0)
    final_k: int = Field(default=6, gt=0)
    fusion: FusionConfig = Field(default_factory=FusionConfig)

    @model_validator(mode="after")
    def _check_at_least_one_retriever(self) -> RetrievalConfig:
        if self.dense_k == 0 and self.lexical_k == 0:
            raise ConfigError("dense_k and lexical_k cannot both be zero")
        return self


class RerankConfig(_Base):
    provider: Literal["none", "heuristic", "ollama"] = "none"
    model: str = "qwen2.5:3b"
    top_n: int = Field(default=20, gt=0)


class GenerationConfig(_Base):
    provider: Literal["extractive", "ollama"] = "extractive"
    model: str = "qwen2.5:7b-instruct"
    max_context_chars: int = Field(default=6000, gt=0)
    max_chunk_chars: int = Field(default=1500, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    max_sentences: int = Field(default=3, gt=0)
    # Minimum share of the question's content terms a sentence must share with the
    # context before it may be used as an answer. Zero answers everything; see the
    # refusal table in the README for what raising it costs and buys.
    min_support: float = Field(default=0.0, ge=0.0, le=1.0)


class OllamaConfig(_Base):
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_attempts: int = Field(default=3, gt=0)
    backoff_seconds: float = Field(default=0.5, ge=0)


class AppConfig(_Base):
    name: str = "default"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    lexical: LexicalConfig = Field(default_factory=LexicalConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)

    @model_validator(mode="after")
    def _check_dimensions(self) -> AppConfig:
        if self.rerank.top_n < self.retrieval.final_k:
            raise ConfigError("rerank.top_n must be at least retrieval.final_k")
        # An HNSW scan visits at most ef_search nodes, so a LIMIT above it quietly
        # returns fewer rows than asked for — which reads as a recall problem, not a
        # configuration one.
        if self.store.backend == "pgvector" and self.store.hnsw.ef_search < self.retrieval.dense_k:
            raise ConfigError(
                f"store.hnsw.ef_search ({self.store.hnsw.ef_search}) is below "
                f"retrieval.dense_k ({self.retrieval.dense_k}); the index cannot return that many"
            )
        return self

    def needs_ollama(self) -> bool:
        return "ollama" in {
            self.embedding.provider,
            self.rerank.provider,
            self.generation.provider,
        }


def _expand(value: Any) -> Any:
    """Replace ``${VAR}`` and ``${VAR:default}`` with environment values.

    Secrets belong in the environment, not in a file that gets committed; the DSN in
    ``config/pgvector.yaml`` is a reference, not a value, and resolves at runtime.
    """
    if isinstance(value, str):
        def substitute(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name, default)
            if resolved is None:
                raise ConfigError(f"environment variable {name} is not set and has no default")
            return resolved

        return _ENV_PATTERN.sub(substitute, value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def build_config(payload: dict[str, Any]) -> AppConfig:
    """Validate a configuration mapping, reporting every failure as :class:`ConfigError`.

    Pydantic wraps validator errors in ``ValidationError``; callers of this module
    should only ever have to catch one exception type.
    """
    try:
        return AppConfig.model_validate(_expand(payload))
    except ValidationError as error:
        raise ConfigError(str(error)) from error


def load_config(path: str | Path) -> AppConfig:
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"config file not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: expected a mapping at the top level")
    try:
        return build_config(raw)
    except ConfigError as error:
        raise ConfigError(f"{source}: {error}") from error
