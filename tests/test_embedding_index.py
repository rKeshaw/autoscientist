"""
Tests for EmbeddingIndex — FAISS-backed vector similarity index.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from embedding_index import EmbeddingIndex


def _random_emb(dim=384):
    """Generate a random normalized embedding vector."""
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _similar_emb(base, noise=0.05):
    """Generate an embedding similar to base with small perturbation."""
    v = base + np.random.randn(len(base)).astype(np.float32) * noise
    v /= np.linalg.norm(v)
    return v


class TestEmbeddingIndex:

    def test_add_and_query(self):
        """Add embeddings and verify query returns correct results."""
        idx = EmbeddingIndex(dimension=384)
        embs = {}
        for i in range(50):
            nid = f"node-{i}"
            emb = _random_emb()
            idx.add(nid, emb)
            embs[nid] = emb

        assert idx.size == 50

        # Query with one of the stored embeddings — should return itself
        target = embs["node-0"]
        results = idx.query(target, threshold=0.9, top_k=3)
        assert len(results) >= 1
        assert results[0][0] == "node-0"
        assert results[0][1] > 0.99  # should be ~1.0 (exact match)

    def test_add_replaces_existing(self):
        """Adding a node with same ID replaces the embedding."""
        idx = EmbeddingIndex(dimension=384)
        emb1 = _random_emb()
        emb2 = _random_emb()
        idx.add("node-a", emb1)
        idx.add("node-a", emb2)

        assert idx.size == 1

        # Query with new embedding should match
        results = idx.query(emb2, threshold=0.9, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "node-a"

    def test_remove(self):
        """Remove a node and verify it no longer appears in results."""
        idx = EmbeddingIndex(dimension=384)
        emb = _random_emb()
        idx.add("node-a", emb)
        idx.add("node-b", _random_emb())

        assert idx.size == 2
        idx.remove("node-a")
        assert idx.size == 1

        results = idx.query(emb, threshold=0.1, top_k=5)
        node_ids = [r[0] for r in results]
        assert "node-a" not in node_ids

    def test_remove_nonexistent(self):
        """Removing a non-existent node should be a no-op."""
        idx = EmbeddingIndex(dimension=384)
        idx.add("node-a", _random_emb())
        idx.remove("nonexistent")
        assert idx.size == 1

    def test_empty_index_query(self):
        """Query on empty index returns empty list."""
        idx = EmbeddingIndex(dimension=384)
        results = idx.query(_random_emb(), threshold=0.1, top_k=5)
        assert results == []

    def test_has(self):
        """Check if a node exists in the index."""
        idx = EmbeddingIndex(dimension=384)
        idx.add("node-a", _random_emb())
        assert idx.has("node-a")
        assert not idx.has("node-b")

    def test_get_embedding(self):
        """Retrieve stored embedding for a node."""
        idx = EmbeddingIndex(dimension=384)
        emb = _random_emb()
        idx.add("node-a", emb)

        retrieved = idx.get_embedding("node-a")
        assert retrieved is not None
        assert np.allclose(retrieved, emb, atol=1e-5)

        assert idx.get_embedding("nonexistent") is None

    def test_all_pairwise_above(self):
        """Find all pairs above a similarity threshold."""
        idx = EmbeddingIndex(dimension=384)

        # Add two very similar embeddings
        base = _random_emb()
        similar = _similar_emb(base, noise=0.02)
        distant = _random_emb()

        idx.add("node-a", base)
        idx.add("node-b", similar)
        idx.add("node-c", distant)

        pairs = idx.all_pairwise_above(threshold=0.9)

        # Should find the similar pair
        pair_ids = [(p[0], p[1]) for p in pairs]
        assert ("node-a", "node-b") in pair_ids or \
               ("node-b", "node-a") in pair_ids

        # Distant node should not appear with either
        for p in pairs:
            assert "node-c" not in (p[0], p[1]) or p[2] < 0.9

    def test_all_pairwise_empty(self):
        """Pairwise on empty or single-node index returns empty."""
        idx = EmbeddingIndex(dimension=384)
        assert idx.all_pairwise_above(0.5) == []

        idx.add("node-a", _random_emb())
        assert idx.all_pairwise_above(0.5) == []

    def test_save_and_load(self):
        """Save index to disk and reload — results should be identical."""
        idx = EmbeddingIndex(dimension=384)
        embs = {}
        for i in range(20):
            nid = f"node-{i}"
            emb = _random_emb()
            idx.add(nid, emb)
            embs[nid] = emb

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_index")
            idx.save(path)

            # Load into new instance
            loaded = EmbeddingIndex.load(path)

            assert loaded.size == 20

            # Query should return same results
            target = embs["node-5"]
            results_orig = idx.query(target, threshold=0.5, top_k=5)
            results_loaded = loaded.query(target, threshold=0.5, top_k=5)

            orig_ids = [r[0] for r in results_orig]
            loaded_ids = [r[0] for r in results_loaded]
            assert orig_ids == loaded_ids

    def test_threshold_filtering(self):
        """Verify that results below threshold are excluded."""
        idx = EmbeddingIndex(dimension=384)
        for i in range(100):
            idx.add(f"node-{i}", _random_emb())

        query = _random_emb()
        # With very high threshold, most random vectors shouldn't match
        results = idx.query(query, threshold=0.95, top_k=100)
        for _, score in results:
            assert score >= 0.95

    def test_top_k_limit(self):
        """Verify that top_k limits the number of results."""
        idx = EmbeddingIndex(dimension=384)
        for i in range(50):
            idx.add(f"node-{i}", _random_emb())

        results = idx.query(_random_emb(), threshold=0.0, top_k=3)
        assert len(results) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
