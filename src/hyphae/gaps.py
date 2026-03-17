"""Gap detection on the cluster manifold.

Given observations and an objective, computes geodesic paths through
cluster manifold space, identifies high-cost edges (gaps in knowledge),
and retrieves facts at gap midpoints to fill the gaps.
"""
from __future__ import annotations

import time
import logging

import numpy as np

from .types import Fact, Gap, GapAnalysis
from .cluster import ClusterEngine, K_NEIGHBORS
from .embed import Embedder
from .shard import Shard, _tags_match

logger = logging.getLogger("hyphae.gaps")


def compute_geodesic(
    manifold_coords: np.ndarray,
    source_indices: list[int],
    target_indices: list[int],
    k: int = K_NEIGHBORS,
) -> tuple[list[int], float]:
    """Greedy geodesic walk through manifold space.

    Walks from the source closest to a target, greedily moving to the
    neighbor closest to the target at each step.

    Returns (path_indices, total_cost).
    """
    if not source_indices or not target_indices:
        return [], float("inf")

    # Find the source-target pair closest in manifold space
    best_source = best_target = None
    best_dist = float("inf")
    for s in source_indices:
        for t in target_indices:
            d = float(np.linalg.norm(manifold_coords[s] - manifold_coords[t]))
            if d < best_dist:
                best_dist = d
                best_source, best_target = s, t

    if best_source is None:
        return [], float("inf")

    path = [best_source]
    visited = {best_source}
    current = best_source
    total_cost = 0.0
    max_steps = min(50, len(manifold_coords))
    target_coords = manifold_coords[best_target]

    for _ in range(max_steps):
        if current == best_target:
            break

        dists = np.linalg.norm(manifold_coords - manifold_coords[current], axis=1)
        sorted_idx = np.argsort(dists)

        moved = False
        for next_idx in sorted_idx[1:k + 1]:
            next_idx = int(next_idx)
            if next_idx in visited:
                continue
            dist_to_target = np.linalg.norm(manifold_coords[next_idx] - target_coords)
            current_dist = np.linalg.norm(manifold_coords[current] - target_coords)
            if dist_to_target < current_dist:
                total_cost += float(dists[next_idx])
                path.append(next_idx)
                visited.add(next_idx)
                current = next_idx
                moved = True
                break

        if not moved:
            # Jump to closest unvisited point to target
            dists_to_target = np.linalg.norm(manifold_coords - target_coords, axis=1)
            for jump_idx in np.argsort(dists_to_target):
                jump_idx = int(jump_idx)
                if jump_idx not in visited:
                    total_cost += float(np.linalg.norm(
                        manifold_coords[current] - manifold_coords[jump_idx]
                    ))
                    path.append(jump_idx)
                    visited.add(jump_idx)
                    current = jump_idx
                    break
            else:
                break

    return path, total_cost


def detect_gaps(
    path: list[int],
    manifold_coords: np.ndarray,
    fact_texts: list[str],
    cluster_id: int = -1,
    top_k: int = 3,
) -> list[Gap]:
    """Find the highest-cost edges on a geodesic path."""
    if len(path) < 2:
        return []

    edge_costs = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        cost = float(np.linalg.norm(manifold_coords[u] - manifold_coords[v]))
        edge_costs.append((i, cost, u, v))

    finite = [c for _, c, _, _ in edge_costs if c < float("inf")]
    avg_cost = float(np.mean(finite)) if finite else 1.0

    edge_costs.sort(key=lambda x: x[1], reverse=True)

    gaps = []
    for pos, cost, u, v in edge_costs[:top_k]:
        midpoint = (manifold_coords[u] + manifold_coords[v]) / 2.0
        u_text = fact_texts[u] if u < len(fact_texts) else f"[{u}]"
        v_text = fact_texts[v] if v < len(fact_texts) else f"[{v}]"
        gaps.append(Gap(
            position=pos,
            edge_cost=cost,
            normalized_cost=cost / avg_cost if avg_cost > 0 else 0,
            from_fact_text=u_text,
            to_fact_text=v_text,
            from_idx=u,
            to_idx=v,
            midpoint_coords=midpoint,
            cluster_id=cluster_id,
        ))

    return gaps


def analyze(
    observations: list[str],
    objective: str,
    cluster_engine: ClusterEngine,
    shard: Shard,
    embedder: Embedder,
    top_k_clusters: int = 2,
    top_k_gaps: int = 3,
    retrieval_k: int = 10,
    remote_shards: list[Shard] | None = None,
    scope: dict[str, str] | None = None,
) -> GapAnalysis:
    """Full gap analysis pipeline.

    1. Embed observations + objective
    2. Route to best cluster(s)
    3. Per cluster: find waypoints, compute geodesic, detect gaps
    4. Retrieve at gap midpoints (local manifold k-NN + all shards)
    5. Return combined analysis
    """
    t0 = time.time()

    all_texts = observations + [objective]
    all_embeddings = embedder.encode(all_texts)
    obs_embeddings = all_embeddings[:-1]
    obj_embedding = all_embeddings[-1]

    # Route to clusters
    cluster_candidates = cluster_engine.top_clusters(obj_embedding, k=top_k_clusters)
    cluster_ids = [cid for cid, sim in cluster_candidates if sim > 0.3]
    if not cluster_ids and cluster_candidates:
        cluster_ids = [cluster_candidates[0][0]]

    all_gaps: list[Gap] = []
    all_retrieved: list[Fact] = []
    seen_texts: set[str] = set()

    for cid in cluster_ids:
        # Ensure manifold is computed
        if not cluster_engine.ensure_manifold(cid, shard):
            # No manifold — fall back to vector search within cluster
            for obs_emb in obs_embeddings:
                results = cluster_engine.knn_at_coords(cid, obs_emb, k=5, shard=shard)
                for f in results:
                    if scope and not _tags_match(f.tags, scope):
                        continue
                    if f.text not in seen_texts:
                        seen_texts.add(f.text)
                        all_retrieved.append(f)
            continue

        cluster = cluster_engine.clusters[cid]
        coords = cluster.manifold_coords

        # Find observation waypoints in this cluster
        source_indices = []
        for obs_emb in obs_embeddings:
            nearest = cluster_engine.nearest_in_cluster(cid, obs_emb, k=1)
            if nearest:
                source_indices.append(nearest[0][0])
        source_indices = list(set(source_indices))

        # Find objective in this cluster
        obj_nearest = cluster_engine.nearest_in_cluster(cid, obj_embedding, k=3)
        target_indices = [idx for idx, _ in obj_nearest]

        if not source_indices or not target_indices:
            continue

        # Geodesic path
        path, cost = compute_geodesic(coords, source_indices, target_indices)
        if not path:
            continue

        # Build fact text lookup
        fact_texts = []
        for fid in cluster.fact_ids:
            fact = shard.get(fid)
            fact_texts.append(fact.text if fact else f"[{fid}]")

        # Detect gaps
        gaps = detect_gaps(path, coords, fact_texts, cluster_id=cid, top_k=top_k_gaps)

        # Retrieve at each gap's midpoint
        for gap in gaps:
            # Manifold k-NN retrieval (over-fetch when scoped)
            fetch_k = retrieval_k * 3 if scope else retrieval_k
            gap_facts = cluster_engine.knn_at_coords(cid, gap.midpoint_coords,
                                                      k=fetch_k, shard=shard)
            count = 0
            for f in gap_facts:
                if scope and not _tags_match(f.tags, scope):
                    continue
                if f.text not in seen_texts:
                    seen_texts.add(f.text)
                    gap.retrieved_facts.append(f)
                    all_retrieved.append(f)
                    count += 1
                    if count >= retrieval_k:
                        break

        all_gaps.extend(gaps)

    # Also query remote shards at gap contexts
    if remote_shards:
        for gap in all_gaps[:3]:
            query = f"{gap.from_fact_text[:100]} {gap.to_fact_text[:100]}"
            for rs in remote_shards:
                try:
                    remote_facts = rs.search_text(query, top_k=5)
                    for f in remote_facts:
                        if f.text not in seen_texts:
                            seen_texts.add(f.text)
                            gap.retrieved_facts.append(f)
                            all_retrieved.append(f)
                except Exception:
                    pass

    # Sort gaps by cost
    all_gaps.sort(key=lambda g: g.edge_cost, reverse=True)

    return GapAnalysis(
        observations=observations,
        objective=objective,
        gaps=all_gaps,
        retrieved_knowledge=all_retrieved,
        clusters_used=cluster_ids,
        elapsed_sec=time.time() - t0,
    )
