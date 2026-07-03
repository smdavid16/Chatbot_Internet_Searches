"""
Search Cache
============
A ChromaDB-backed semantic cache for search results.

When the chatbot searches for something, the query and results are stored
in a persistent vector database.  On subsequent queries, the cache is
checked first — if a semantically similar query was recently searched,
the cached result is returned instantly instead of hitting the internet.

Embeddings are generated locally using ChromaDB's built-in
all-MiniLM-L6-v2 model (no API key required).
"""

import hashlib
import time

import chromadb
from chromadb.config import Settings


class SearchCache:
    """Semantic search-result cache backed by ChromaDB."""

    def __init__(
        self,
        persist_dir: str,
        ttl_seconds: int = 86_400,
        similarity_threshold: float = 0.35,
    ):
        """
        Args:
            persist_dir: Directory for persistent ChromaDB storage.
            ttl_seconds: How long cached results stay valid (seconds).
            similarity_threshold: Max L2 distance for a cache hit.
                Lower = stricter matching, higher = more generous.
        """
        self._ttl = ttl_seconds
        self._threshold = similarity_threshold

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="search_cache",
            metadata={"hnsw:space": "l2"},  # L2 distance
        )

    # ── Public API ──────────────────────────────────────────────────────

    def lookup(self, query: str) -> str | None:
        """Search the cache for a semantically similar past query.

        Returns the cached result string if a close match is found
        and it hasn't expired, otherwise returns None.
        """
        if self._collection.count() == 0:
            return None

        results = self._collection.query(
            query_texts=[query],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return None

        now = time.time()

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Check similarity threshold
            if dist > self._threshold:
                continue

            # Check TTL
            cached_at = meta.get("timestamp", 0)
            age = now - cached_at
            if age > self._ttl:
                # Expired — remove it
                self._remove_by_query(doc)
                continue

            return meta.get("result")

        return None

    def store(self, query: str, result: str) -> None:
        """Store a search result in the cache.

        The query text is used as the document for embedding.
        The full result string is stored in metadata.
        """
        doc_id = self._make_id(query)

        # Upsert so re-searching the same query updates the cache
        self._collection.upsert(
            ids=[doc_id],
            documents=[query],
            metadatas=[{
                "result": result,
                "timestamp": time.time(),
            }],
        )

    def clear(self) -> int:
        """Remove all cached entries.  Returns the number removed."""
        count = self._collection.count()
        if count > 0:
            # Delete the collection and recreate it
            self._client.delete_collection("search_cache")
            self._collection = self._client.get_or_create_collection(
                name="search_cache",
                metadata={"hnsw:space": "l2"},
            )
        return count

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "entries": self._collection.count(),
            "ttl_seconds": self._ttl,
            "similarity_threshold": self._threshold,
        }

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _make_id(query: str) -> str:
        """Deterministic ID from a query string."""
        normalised = query.strip().lower()
        return hashlib.sha256(normalised.encode()).hexdigest()[:16]

    def _remove_by_query(self, query: str) -> None:
        """Remove a cached entry by its original query."""
        doc_id = self._make_id(query)
        try:
            self._collection.delete(ids=[doc_id])
        except Exception:
            pass  # Silently ignore if already gone
