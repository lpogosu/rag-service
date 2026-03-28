"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import ConfigError, build_config, load_config


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["offline", "default", "pgvector"])
def test_shipped_profiles_load(name: str) -> None:
    config = load_config(f"config/{name}.yaml")
    assert config.name == name


def test_offline_profile_needs_nothing_external() -> None:
    config = load_config("config/offline.yaml")
    assert not config.needs_ollama()
    assert config.store.backend == "memory"


def test_a_typo_in_a_key_is_an_error_not_a_silent_default(tmp_path: Path) -> None:
    """Without extra="forbid" this runs happily with a chunk size nobody chose."""
    path = write(tmp_path, "chunking:\n  strategy: fixed\n  optionz: {size: 100}\n")
    with pytest.raises(ConfigError, match="optionz"):
        load_config(path)


def test_environment_variables_are_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_TEST_DSN", "postgresql://user@host/db")
    path = write(tmp_path, "store:\n  backend: pgvector\n  dsn: ${RAG_TEST_DSN}\n")
    assert load_config(path).store.dsn == "postgresql://user@host/db"


def test_a_default_is_used_when_the_variable_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAG_TEST_URL", raising=False)
    path = write(tmp_path, "ollama:\n  base_url: ${RAG_TEST_URL:http://fallback:11434}\n")
    assert load_config(path).ollama.base_url == "http://fallback:11434"


def test_a_missing_variable_without_a_default_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAG_TEST_MISSING", raising=False)
    path = write(tmp_path, "store:\n  backend: pgvector\n  dsn: ${RAG_TEST_MISSING}\n")
    with pytest.raises(ConfigError, match="RAG_TEST_MISSING"):
        load_config(path)


def test_pgvector_without_a_dsn_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "store:\n  backend: pgvector\n")
    with pytest.raises(ConfigError, match=r"store\.dsn is required"):
        load_config(path)


def test_ef_search_below_dense_k_is_rejected() -> None:
    """pgvector never returns more rows than ef_search, whatever the LIMIT says."""
    with pytest.raises(ConfigError, match="ef_search"):
        build_config(
            {
                "store": {
                    "backend": "pgvector",
                    "dsn": "postgresql://x",
                    "hnsw": {"ef_search": 10},
                },
                "retrieval": {"dense_k": 40},
            }
        )


def test_rerank_window_smaller_than_the_final_cut_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"rerank\.top_n"):
        build_config({"retrieval": {"final_k": 10}, "rerank": {"top_n": 5}})


def test_disabling_both_retrievers_is_rejected() -> None:
    with pytest.raises(ConfigError, match="cannot both be zero"):
        build_config({"retrieval": {"dense_k": 0, "lexical_k": 0}})


def test_an_unknown_chunking_strategy_is_rejected() -> None:
    with pytest.raises(ConfigError, match="chunking"):
        build_config({"chunking": {"strategy": "semantic"}})


def test_missing_file_is_reported_clearly() -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config("config/does-not-exist.yaml")


def test_configuration_is_immutable() -> None:
    config = load_config("config/offline.yaml")
    with pytest.raises(ValueError, match="frozen"):
        config.retrieval.final_k = 99  # type: ignore[misc]


def test_needs_ollama_is_true_when_any_stage_uses_it() -> None:
    assert load_config("config/default.yaml").needs_ollama()
    assert load_config("config/pgvector.yaml").needs_ollama()
