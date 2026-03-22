"""Metadata filtering shared by the vector stores and the lexical index.

Filtering lives in its own module because both retrievers must apply exactly the same
predicate. If the dense side filtered on ``lang == "ru"`` and the lexical side did not,
rank fusion would silently mix two different candidate populations and the result would
look like a relevance bug.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag.types import Metadata, MetadataValue

Filters = Mapping[str, MetadataValue | Sequence[MetadataValue]]


def matches_filters(metadata: Metadata, filters: Filters) -> bool:
    """Equality on scalars, membership on sequences. No ranges, no negation.

    Deliberately small: every operator added here has to be expressible in both the
    in-memory store and in SQL, and the ones above cover document type, language and
    version, which is what the corpus actually needs.
    """
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, list | tuple | set):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
