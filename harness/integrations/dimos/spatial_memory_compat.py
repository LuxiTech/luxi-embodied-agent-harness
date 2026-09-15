"""Local compatibility layer for DimOS named spatial locations.

The pinned DimOS release lets Chroma choose its default text embedding model
for named locations.  That silently introduces a second model download and,
on this workstation, inherits a browser-only SOCKS proxy which Chroma's HTTP
client cannot parse.  Image memory already owns a local CLIP provider, so the
Luxi launcher reuses that provider for location names as well.

Named locations are exact-match by default. DimOS asks named locations before
checking the live camera or semantic image map, so an unconditional nearest
neighbour lookup can turn an unrelated object query such as ``door`` into a
navigation goal for whichever test tag is closest in CLIP space. Semantic
aliases remain available as an explicit opt-in.

This module is installed in the DimOS parent process before workers are
forked.  It deliberately leaves the pinned checkout untouched.
"""

from __future__ import annotations

import os
from typing import Any


_COLLECTION_ATTRIBUTE = "_luxi_clip_location_collection"
_COLLECTION_SUFFIX = "locations_clip"
_SEMANTIC_ALIASES_ENV = "LUXI_SPATIAL_TAG_SEMANTIC_ALIASES"


def _enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _embedding_values(database: Any, text: str) -> list[float]:
    provider = getattr(database, "embedding_provider", None)
    if provider is None:
        from dimos.perception.image_embedding import ImageEmbeddingProvider

        provider = ImageEmbeddingProvider(model_name="clip")
        database.embedding_provider = provider
    embedding = provider.get_text_embedding(text)
    values = embedding.tolist()
    if not isinstance(values, list) or not values:
        raise RuntimeError("CLIP did not return a location-name embedding")
    return [float(value) for value in values]


def _location_collection(database: Any) -> Any:
    collection = getattr(database, _COLLECTION_ATTRIBUTE, None)
    if collection is None:
        collection = database.client.get_or_create_collection(
            name=f"{database.collection_name}_{_COLLECTION_SUFFIX}",
            metadata={"hnsw:space": "cosine"},
        )
        setattr(database, _COLLECTION_ATTRIBUTE, collection)
    return collection


def _latest_exact_location(collection: Any, query: str) -> Any | None:
    """Return the newest exact-name match without running an embedding model."""
    results = collection.get(
        where={"location_name": query},
        include=["metadatas", "documents"],
    )
    metadata = list((results or {}).get("metadatas") or [])
    if not metadata:
        return None
    newest = max(metadata, key=lambda item: float((item or {}).get("timestamp", 0.0)))
    from dimos.types.robot_location import RobotLocation

    return RobotLocation.from_vector_metadata(newest)


def tag_location_with_clip(database: Any, location: Any) -> None:
    """Store a named pose with the already-loaded local CLIP text encoder."""
    collection = _location_collection(database)
    collection.upsert(
        ids=[location.location_id],
        embeddings=[_embedding_values(database, location.name)],
        documents=[location.name],
        metadatas=[location.to_vector_metadata()],
    )


def query_tagged_location_with_clip(database: Any, query: str) -> tuple[Any | None, float]:
    """Resolve exact tags and only use semantic aliases when explicitly enabled."""
    collection = _location_collection(database)
    exact = _latest_exact_location(collection, query)
    if exact is not None:
        return exact, 0.0
    if not _enabled(_SEMANTIC_ALIASES_ENV, default=False):
        # SpatialPerception also applies a distance threshold. A rejected
        # named-location lookup must look maximally distant so the request can
        # continue to live vision and the semantic image map.
        return None, 1.0
    if collection.count() == 0:
        return None, 1.0

    results = collection.query(
        query_embeddings=[_embedding_values(database, query)],
        n_results=1,
        include=["metadatas", "documents", "distances"],
    )
    ids = (results or {}).get("ids") or []
    if not ids or not ids[0]:
        return None, 1.0

    metadata = results["metadatas"][0][0]
    distances = (results or {}).get("distances") or [[0.0]]
    distance = float(distances[0][0])
    from dimos.types.robot_location import RobotLocation

    return RobotLocation.from_vector_metadata(metadata), distance


def install_spatial_memory_location_compat() -> bool:
    """Make DimOS named-location storage fully local and CLIP-backed."""
    if not _enabled("LUXI_SPATIAL_LOCATION_COMPAT"):
        return False

    from dimos.perception.spatial_vector_db import SpatialVectorDB

    if getattr(SpatialVectorDB, "_luxi_clip_location_tags", False):
        return True
    SpatialVectorDB.tag_location = tag_location_with_clip  # type: ignore[method-assign]
    SpatialVectorDB.query_tagged_location = query_tagged_location_with_clip  # type: ignore[method-assign]
    SpatialVectorDB._luxi_clip_location_tags = True
    return True
