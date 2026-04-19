# Hyphae — Known Gaps & Improvement Plan

Status: Identified 2026-03-16. None started yet.

## Context

Hyphae is a self-organizing memory system with online growing clusters, per-cluster diffusion maps, and manifold-aware retrieval. As of today it has ~2800 facts, 272 clusters (22 with manifolds), and a newly added manifold fallback for cross-project discovery.

Reference implementation with more advanced versions of these features: Pathfinder v2 at `C:\Users\om\Desktop\pathfinder\pathfinder_v2_clustered.py` and `C:\Users\om\Desktop\manifold_retrieval\manifold_builder.py`.

## Gap 1: Cluster Fragmentation (HIGH IMPACT)

**Problem:** 109 singleton clusters, 185/272 have ≤5 facts. The routing threshold (0.55 cosine sim) spawns new clusters too aggressively. Most knowledge lives in isolated pockets that never reach the 30-fact manifold threshold.

**Fix:** Either lower `sim_threshold` from 0.55 to ~0.45 (route more facts to existing clusters), or merge more aggressively (lower `merge_threshold` from 0.85 to ~0.75), or both. Could also lower `MIN_MANIFOLD` from 30 to 15.

**Files:** `cluster.py` lines 21-25 (constants), line 85 (routing decision), line 145 (merge decision)

## Gap 2: No Cross-Cluster Traversal (MEDIUM IMPACT)

**Problem:** Each cluster's manifold is an island. If a query spans facts in two clusters, the geometry can't bridge them. Pathfinder has a coarse inter-cluster graph (centroid similarity > 0.3) enabling cross-cluster hops.

**Fix:** Build an inter-cluster adjacency graph from centroid similarities. During gap analysis and manifold fallback, allow traversal across cluster boundaries. Store as a lightweight centroid-to-centroid similarity matrix, rebuilt on merge/create.

**Files:** `cluster.py` (new method), `gaps.py` (use inter-cluster graph in `analyze()`), `__init__.py` (`_manifold_fallback` could hop clusters)

## Gap 3: Fixed Gaussian Bandwidth (LOW-MEDIUM IMPACT)

**Problem:** `cluster.py` line 220 uses hardcoded `sigma = 0.5` for the Gaussian kernel in manifold construction. Pathfinder uses Zelnik-Manor & Perona local scaling where each point gets σ_i = distance to its median neighbor. Fixed σ distorts manifolds in clusters with uneven density.

**Fix:** Replace fixed sigma with local scaling:
```python
# For each point i:
sigma_i = distances_to_kth_neighbor[i]  # k = K_NEIGHBORS // 2
# For each edge (i,j):
w = exp(-d^2 / (2 * sigma_i * sigma_j))
```

**Files:** `cluster.py` `build_manifold()` lines 207-229

**Reference:** `manifold_builder.py` lines 248-332

## Gap 4: No Nyström Extension — Full Recompute (HIGH IMPACT)

**Problem:** When a new fact enters a cluster, the entire diffusion map is recomputed from scratch (next background builder cycle, up to 60s). For a cluster with 700 facts, this is expensive. Pathfinder has Nyström extension that projects new facts onto the existing manifold instantly via weighted k-NN interpolation.

**Fix:** Add a `nystrom_extend()` method to ClusterEngine. When a fact arrives in a clean cluster (manifold already computed), project it onto existing coordinates instead of marking dirty:
```python
def nystrom_extend(self, cluster_id, new_embedding):
    cluster = self.clusters[cluster_id]
    # k-NN in embedding space
    sims = cluster.local_embeddings @ new_embedding
    top_k = argpartition(-sims, 15)[:15]
    distances = 1.0 - sims[top_k]
    sigma = median(distances)
    weights = exp(-distances^2 / (2 * sigma^2))
    weights /= sum(weights)
    new_coords = sum(weights[i] * cluster.manifold_coords[idx] for i, idx in enumerate(top_k))
    # Append to manifold_coords, mark dirty only after DIRTY_THRESHOLD new facts
    return new_coords
```

Still mark dirty after `DIRTY_THRESHOLD` (50) new Nyström extensions for periodic full recompute.

**Files:** `cluster.py` (new method), `__init__.py` `remember()` (call nystrom_extend instead of just marking dirty)

**Reference:** `manifold_builder.py` lines 433-459

## Gap 5: Causal Links Stored but Never Used (MEDIUM IMPACT)

**Problem:** The `causal_links` table has `from_fact_id`, `to_fact_id`, `relation`, `confidence` but nothing in recall, manifold building, or gap detection reads them. Pathfinder boosts explicit relationship edges by 2x in the affinity matrix.

**Fix:** In `build_manifold()`, after constructing the k-NN affinity matrix, query causal_links for any edges within the cluster. Boost those edge weights:
```python
links = shard.get_links_for_cluster(cluster_id)
for link in links:
    i, j = local_index[link.from_fact_id], local_index[link.to_fact_id]
    W[i,j] *= 2.0 * link.confidence
    W[j,i] *= 2.0 * link.confidence
```

Also consider using causal links during recall to expand results (if fact A is recalled and A→B is a causal link, boost B's score).

**Files:** `cluster.py` `build_manifold()`, `shard.py` (new `get_links_for_cluster()`), optionally `__init__.py` recall pipeline

## Gap 6: No Diffusion Time Scaling (MEDIUM IMPACT)

**Problem:** `cluster.py` line 254: `coords = eigenvectors * eigenvalues[None, :]` — this is effectively diffusion time t=1. Pathfinder uses t=3 (`eigenvalues ** 3`), which emphasizes global topology over local noise. With t=1, manifold distances are dominated by fine-grained local structure.

**Fix:** Change line 254 to:
```python
DIFFUSION_TIME = 3
coords = eigenvectors * (eigenvalues[None, :] ** DIFFUSION_TIME)
```

Add `DIFFUSION_TIME = 3` as a module constant next to `MANIFOLD_DIM`.

**Files:** `cluster.py` line 254, add constant at line 24

## Gap 7: Aggressive Decay Kills Niche Facts (LOW-MEDIUM IMPACT)

**Problem:** 30-day half-life + 0.55 minimum score in MCP layer. A fact saved 90 days ago and never recalled is at ~12.5% of its original cosine score. Combined with the MCP threshold, old niche facts are effectively invisible.

**Fix options:**
- Increase `HALF_LIFE_DAYS` from 30 to 90 (slower decay)
- Lower `_MIN_SCORE` in MCP from 0.55 to 0.45 (show weaker matches)
- Add a "pinned" flag that exempts important facts from decay (schema already has this concept in purge)
- Manifold fallback (already implemented today) partially mitigates this since it bypasses the scoped search that applies decay

**Files:** `shard.py` line 21 (`HALF_LIFE_DAYS`), `mcp_server.py` line 93 (`_MIN_SCORE`)

## Priority Order

1. **Gap 4** (Nyström extension) — biggest perf win, instant manifold coords for new facts
2. **Gap 6** (diffusion time t=3) — one-line fix, immediate quality improvement
3. **Gap 1** (cluster fragmentation) — tune thresholds, dramatic manifold coverage increase
4. **Gap 5** (causal links) — free signal already being stored
5. **Gap 2** (cross-cluster traversal) — enables richer geometric reasoning
6. **Gap 3** (local scaling) — manifold quality improvement
7. **Gap 7** (decay tuning) — partially mitigated by manifold fallback
