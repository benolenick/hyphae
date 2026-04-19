"""Online growing clusters — the core of Hyphae.

Facts are embedded, routed to the nearest cluster by cosine similarity.
If no cluster is close enough, a new one spins up. Clusters merge when
they overlap. Each cluster lazily computes diffusion map coordinates
for manifold-based gap detection.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from .types import ClusterState, Fact

logger = logging.getLogger("hyphae.cluster")

K_NEIGHBORS = 15   # k-NN for manifold construction
MIN_MANIFOLD = 30  # minimum facts before computing manifold
MANIFOLD_DIM = 12  # diffusion map dimensions
DIRTY_THRESHOLD = 50  # recompute manifold after this many new facts


class ClusterEngine:
    """Online growing cluster manager with lazy manifold computation."""

    def __init__(
        self,
        sim_threshold: float = 0.55,
        merge_threshold: float = 0.85,
    ):
        self.sim_threshold = sim_threshold
        self.merge_threshold = merge_threshold
        self.clusters: dict[int, ClusterState] = {}
        self._next_id = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_from_shard(self, shard) -> None:
        """Load cluster centroids and fact_ids from shard's persistence."""
        saved = shard.load_clusters()
        for cid, (centroid, count) in saved.items():
            self.clusters[cid] = ClusterState(
                id=cid, centroid=centroid, count=count,
                dirty=True,  # manifold needs recompute
            )
            self._next_id = max(self._next_id, cid + 1)

        # Populate fact_ids from the facts table so manifold building works
        if self.clusters:
            fact_cluster_map = shard.get_all_fact_cluster_ids()
            for fact_id, cluster_id in fact_cluster_map:
                cluster = self.clusters.get(cluster_id)
                if cluster:
                    cluster.fact_ids.append(fact_id)

        logger.info(f"Loaded {len(self.clusters)} clusters from shard")

    def save_to_shard(self, shard) -> None:
        """Persist all cluster centroids to shard."""
        for c in self.clusters.values():
            shard.save_cluster(c.id, c.centroid, c.count)

    # ------------------------------------------------------------------
    # Routing & ingestion
    # ------------------------------------------------------------------

    def route(self, embedding: np.ndarray) -> int:
        """Find the best cluster for an embedding, or create a new one."""
        if not self.clusters:
            return self._create_cluster(embedding)

        # Cosine similarity to all centroids
        centroids = np.stack([c.centroid for c in self.clusters.values()])
        sims = centroids @ embedding
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_cluster = list(self.clusters.values())[best_idx]

        if best_sim < self.sim_threshold:
            # Too far from any cluster — spawn new one
            return self._create_cluster(embedding)

        return best_cluster.id

    def ingest(self, fact: Fact, embedding: np.ndarray) -> int:
        """Route fact to a cluster and update cluster state."""
        cluster_id = self.route(embedding)
        cluster = self.clusters[cluster_id]

        # Update running centroid: weighted average
        n = cluster.count
        cluster.centroid = (cluster.centroid * n + embedding) / (n + 1)
        # Re-normalize centroid
        norm = np.linalg.norm(cluster.centroid)
        if norm > 0:
            cluster.centroid = cluster.centroid / norm

        cluster.fact_ids.append(fact.id)
        cluster.count += 1
        cluster.dirty = True

        return cluster_id

    def _create_cluster(self, embedding: np.ndarray) -> int:
        """Spawn a new cluster centered on this embedding."""
        cid = self._next_id
        self._next_id += 1
        self.clusters[cid] = ClusterState(
            id=cid,
            centroid=embedding.copy(),
            count=0,
            dirty=True,
        )
        logger.info(f"New cluster #{cid} spawned")
        return cid

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def maybe_merge(self) -> list[tuple[int, int]]:
        """Check all cluster pairs for merge. Returns list of (absorbed, into)."""
        if len(self.clusters) < 2:
            return []

        ids = list(self.clusters.keys())
        centroids = np.stack([self.clusters[i].centroid for i in ids])
        sim_matrix = centroids @ centroids.T

        merged = []
        absorbed = set()

        for i in range(len(ids)):
            if ids[i] in absorbed:
                continue
            for j in range(i + 1, len(ids)):
                if ids[j] in absorbed:
                    continue
                if sim_matrix[i, j] >= self.merge_threshold:
                    # Absorb smaller into larger
                    ci, cj = self.clusters[ids[i]], self.clusters[ids[j]]
                    if ci.count >= cj.count:
                        large, small = ci, cj
                    else:
                        large, small = cj, ci

                    # Merge centroid (weighted)
                    total = large.count + small.count
                    if total > 0:
                        large.centroid = (large.centroid * large.count +
                                         small.centroid * small.count) / total
                        norm = np.linalg.norm(large.centroid)
                        if norm > 0:
                            large.centroid /= norm

                    large.fact_ids.extend(small.fact_ids)
                    large.count = total
                    large.dirty = True

                    absorbed.add(small.id)
                    merged.append((small.id, large.id))
                    logger.info(f"Merged cluster #{small.id} into #{large.id} "
                                f"(sim={sim_matrix[i,j]:.3f})")

        for cid in absorbed:
            del self.clusters[cid]

        return merged

    # ------------------------------------------------------------------
    # Manifold computation
    # ------------------------------------------------------------------

    def build_manifold(self, cluster_id: int, shard) -> bool:
        """Compute diffusion map coordinates for a cluster.

        Uses the embeddings stored in the shard. Returns True if
        manifold was computed, False if not enough data.
        """
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return False

        fact_ids, embeddings = shard.get_cluster_embeddings(cluster_id)
        n = len(fact_ids)

        if n < MIN_MANIFOLD:
            logger.debug(f"Cluster #{cluster_id}: {n} facts, need {MIN_MANIFOLD} for manifold")
            return False

        k = min(K_NEIGHBORS, n - 1)
        dim = min(MANIFOLD_DIM, n - 2)
        if dim < 2:
            return False

        t0 = time.time()

        # k-NN similarity matrix
        sims = embeddings @ embeddings.T  # cosine sim (already normalized)

        # Build sparse affinity matrix from k-NN
        rows, cols, vals = [], [], []
        for i in range(n):
            # Top-k neighbors (excluding self)
            neighbors = np.argsort(-sims[i])
            count = 0
            for j in neighbors:
                if j == i:
                    continue
                if count >= k:
                    break
                # Gaussian kernel on cosine distance
                d = 1.0 - sims[i, j]
                sigma = 0.5  # bandwidth
                w = float(np.exp(-d * d / (2 * sigma * sigma)))
                rows.append(i)
                cols.append(j)
                vals.append(w)
                # Symmetrize
                rows.append(j)
                cols.append(i)
                vals.append(w)
                count += 1

        W = csr_matrix((vals, (rows, cols)), shape=(n, n))

        # Normalized graph Laplacian: D^{-1/2} W D^{-1/2}
        D = np.array(W.sum(axis=1)).flatten()
        D[D < 1e-10] = 1e-10
        D_inv_sqrt = 1.0 / np.sqrt(D)
        # Apply normalization
        W_norm = W.multiply(D_inv_sqrt[:, None]).multiply(D_inv_sqrt[None, :])
        W_norm = (W_norm + W_norm.T) / 2  # ensure symmetry

        # Top eigenvectors (skip first trivial one)
        try:
            eigenvalues, eigenvectors = eigsh(W_norm, k=dim + 1, which="LM")
        except Exception as e:
            logger.warning(f"Eigsh failed for cluster #{cluster_id}: {e}")
            # Fall back to raw embeddings so recall still works without manifold.
            # Mark dirty=False so we don't retry on every query and peg the CPU.
            cluster.manifold_coords = embeddings[:, :dim]
            cluster.eigenvalues = np.ones(dim)
            cluster.local_embeddings = embeddings
            cluster.fact_ids = fact_ids
            cluster.dirty = False
            return True

        # Sort by eigenvalue descending, skip first
        order = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[order][1:dim + 1]
        eigenvectors = eigenvectors[:, order][:, 1:dim + 1]

        # Scale by eigenvalues for diffusion distance
        coords = eigenvectors * eigenvalues[None, :]

        cluster.manifold_coords = coords
        cluster.eigenvalues = eigenvalues
        cluster.local_embeddings = embeddings
        cluster.fact_ids = fact_ids
        cluster.dirty = False

        elapsed = time.time() - t0
        logger.info(f"Cluster #{cluster_id}: manifold computed ({n} facts, "
                     f"{dim}D, {elapsed:.2f}s)")
        return True

    def ensure_manifold(self, cluster_id: int, shard) -> bool:
        """Ensure manifold is computed for cluster. Returns True if available."""
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return False
        if cluster.manifold_coords is not None and not cluster.dirty:
            return True
        return self.build_manifold(cluster_id, shard)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def nearest_in_cluster(
        self, cluster_id: int, embedding: np.ndarray, k: int = 1,
    ) -> list[tuple[int, float]]:
        """Find nearest facts in a cluster by cosine sim. Returns (local_idx, sim)."""
        cluster = self.clusters.get(cluster_id)
        if cluster is None or cluster.local_embeddings is None:
            return []
        sims = cluster.local_embeddings @ embedding
        top_k = min(k, len(sims))
        indices = np.argpartition(-sims, top_k)[:top_k]
        indices = indices[np.argsort(-sims[indices])]
        return [(int(i), float(sims[i])) for i in indices]

    def knn_at_coords(
        self, cluster_id: int, coords: np.ndarray, k: int = 10, shard=None,
    ) -> list[Fact]:
        """k-NN in manifold coordinate space. Returns facts with distance scores."""
        cluster = self.clusters.get(cluster_id)
        if cluster is None or cluster.manifold_coords is None:
            return []
        dists = np.linalg.norm(cluster.manifold_coords - coords, axis=1)
        nearest = np.argsort(dists)[:k]
        results = []
        for idx in nearest:
            if idx < len(cluster.fact_ids) and shard:
                fact = shard.get(cluster.fact_ids[idx])
                if fact:
                    fact.score = 1.0 / (1.0 + float(dists[idx]))  # distance → score
                    results.append(fact)
        return results

    def top_clusters(self, embedding: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
        """Find the top-k clusters for an embedding. Returns (cluster_id, similarity)."""
        if not self.clusters:
            return []
        ids = list(self.clusters.keys())
        centroids = np.stack([self.clusters[i].centroid for i in ids])
        sims = centroids @ embedding
        top_k = min(k, len(ids))
        top_idx = np.argsort(-sims)[:top_k]
        return [(ids[i], float(sims[i])) for i in top_idx]

    def status(self) -> dict:
        """Return cluster status summary."""
        return {
            "total_clusters": len(self.clusters),
            "clusters": [
                {
                    "id": c.id,
                    "count": c.count,
                    "has_manifold": c.manifold_coords is not None,
                    "dirty": c.dirty,
                    "age_hours": (time.time() - c.created_at) / 3600,
                }
                for c in sorted(self.clusters.values(), key=lambda x: x.count, reverse=True)
            ],
        }
