"""
Hyphae — self-organizing memory that grows.

Add facts. Query gaps. The network grows itself.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

from .types import Fact, Briefing, CausalLink, Gap, GapAnalysis, Stone, River
from .embed import Embedder
from .shard import LocalShard, RemoteShard, Shard
from .cluster import ClusterEngine, MIN_MANIFOLD
from .river import RiverManager
from . import gaps as gap_engine

__version__ = "0.1.1"

logger = logging.getLogger("hyphae")

DEFAULT_DB = str(Path.home() / ".hyphae" / "hyphae.db")


class Hyphae:
    """Main facade — wires together shards, clusters, gaps, and rivers.

    Session scope: call set_session() or pass session_scope at init to
    auto-scope all recalls and auto-tag all remembers. The system manages
    its own decay — callers never need to think about it.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        model: str = "all-MiniLM-L6-v2",
        sim_threshold: float = 0.45,
        merge_threshold: float = 0.75,
        remote_shards: list[dict] | None = None,
        session_scope: dict[str, str] | None = None,
    ):
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.embedder = Embedder(model_name=model)

        # Session scope — auto-applied to recall and remember
        self._session_scope: dict[str, str] | None = session_scope

        # Local shard (primary storage)
        self.local_shard = LocalShard(db_path=db_path, name="local")

        # Remote shards (optional, for pre-built corpora like Memoria)
        self.remote_shards: list[RemoteShard] = []
        for cfg in (remote_shards or []):
            rs = RemoteShard(
                endpoint=cfg.get("endpoint", ""),
                name=cfg.get("name", "remote"),
                timeout=cfg.get("timeout", 15),
            )
            self.remote_shards.append(rs)

        # Cluster engine
        self.cluster_engine = ClusterEngine(
            sim_threshold=sim_threshold,
            merge_threshold=merge_threshold,
        )
        self.cluster_engine.load_from_shard(self.local_shard)

        # River manager (shares the same SQLite connection)
        self.river_manager = RiverManager(self.local_shard.conn)

        # Track last maintenance time
        self._last_maintain = time.time()
        self._maintain_interval = 300  # auto-maintain every 5 min of activity

        # Background manifold builder
        self._bg_stop = threading.Event()
        self._bg_thread = threading.Thread(
            target=self._background_manifold_builder,
            daemon=True,
            name="hyphae-manifold-builder",
        )
        self._bg_thread.start()

        logger.info(
            f"Hyphae initialized: {self.local_shard.count()} facts, "
            f"{len(self.cluster_engine.clusters)} clusters, "
            f"{len(self.remote_shards)} remote shards, "
            f"session_scope={self._session_scope}"
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def set_session(self, scope: dict[str, str]) -> dict:
        """Set the active session scope. All subsequent recalls are auto-scoped
        to this context, and all remembers are auto-tagged with it.

        Automatically warms all facts in this scope — resets their decay
        so they're immediately snappy, even if you haven't touched the
        project in months.

        Example: h.set_session({"project": "htb-autopwn"})
        """
        self._session_scope = scope
        warm_result = self.warm_scope(scope)
        logger.info(f"Session scope set: {scope}")
        return {"session_scope": scope, "status": "active", **warm_result}

    def clear_session(self) -> dict:
        """Clear session scope. Auto-distills before clearing."""
        old = self._session_scope
        briefing_text = None

        # Distill the outgoing session before clearing
        if old and "project" in old:
            briefing = self.distill(old["project"])
            if briefing:
                briefing_text = briefing.text

        self._session_scope = None
        logger.info(f"Session scope cleared (was {old})")
        return {"cleared": old, "briefing": briefing_text}

    def _effective_scope(self, explicit_scope: dict[str, str] | None) -> dict[str, str] | None:
        """Resolve scope: explicit > session > None."""
        if explicit_scope is not None:
            return explicit_scope
        return self._session_scope

    # ------------------------------------------------------------------
    # Remember — store a fact, route to cluster
    # ------------------------------------------------------------------

    def remember(
        self,
        text: str,
        context_id: str = "",
        tags: dict[str, str] | None = None,
        source: str = "",
        cause_of: str = "",
    ) -> tuple[str, int]:
        """Store a fact. Returns (fact_id, cluster_id).

        If a session scope is active, its tags are merged into the fact's
        tags automatically (explicit tags take precedence).
        """
        # Merge session scope into tags
        merged_tags = {}
        if self._session_scope:
            merged_tags.update(self._session_scope)
        if tags:
            merged_tags.update(tags)  # explicit tags win

        embedding = self.embedder.encode_single(text)

        fact = Fact(
            text=text,
            embedding=embedding,
            tags=merged_tags,
            source=source,
            context_id=context_id,
        )

        self._maybe_maintain()

        # Route to cluster
        cluster_id = self.cluster_engine.ingest(fact, embedding)
        fact.cluster_id = cluster_id

        # Persist
        self.local_shard.store(fact)
        self.local_shard.update_cluster_id(fact.id, cluster_id)

        # Save causal link if specified
        if cause_of:
            link = CausalLink(
                from_fact_id=fact.id,
                to_fact_id=cause_of,
                relation="caused_by",
                context_id=context_id,
            )
            self.local_shard.store_link(link)

        # Persist cluster centroid
        cluster = self.cluster_engine.clusters.get(cluster_id)
        if cluster:
            self.local_shard.save_cluster(cluster_id, cluster.centroid, cluster.count)

        # Periodic merge check (every 100 facts)
        if self.local_shard.count() % 100 == 0:
            merged = self.cluster_engine.maybe_merge()
            if merged:
                self.cluster_engine.save_to_shard(self.local_shard)
                for absorbed, into in merged:
                    # Update fact cluster assignments in DB
                    for fid in self.cluster_engine.clusters[into].fact_ids:
                        self.local_shard.update_cluster_id(fid, into)

        return fact.id, cluster_id

    # ------------------------------------------------------------------
    # Recall — search across all shards
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int = 10,
        scope: dict[str, str] | None = None,
    ) -> list[Fact]:
        """Search for facts across all shards.

        Uses session scope automatically unless an explicit scope is passed.
        Pass scope={} to force unscoped search even when a session is active.

        When scoped, also blends in a few unscoped results so that cross-project
        connections can surface (e.g. openkeel facts appearing in HTB context).

        When manifolds are available, re-ranks results using manifold distance
        for sharper relevance separation.

        MANIFOLD FALLBACK: When scoped recall returns weak results (best score
        below threshold), routes the query to the nearest clusters and retrieves
        facts by manifold proximity — ignoring project scope. The geometry
        decides relevance when tags fail.
        """
        effective = self._effective_scope(scope)
        embedding = self.embedder.encode_single(query)

        # Local vector search — over-fetch for re-ranking
        results = self.local_shard.search(embedding, top_k=top_k * 3, scope=effective)

        # Cross-project blending: when scoped, surface a few unscoped facts too
        if effective:
            seen_ids = {f.id for f in results}
            cross_results = self.local_shard.search(embedding, top_k=top_k // 2, scope={})
            for f in cross_results:
                if f.id not in seen_ids:
                    f.score *= 0.7  # discount cross-scope facts
                    f.tags["_cross_scope"] = "true"
                    results.append(f)
                    seen_ids.add(f.id)

        # Manifold re-ranking: three-way blend (cosine + manifold + co-occurrence)
        results = self._manifold_rerank(results, embedding)

        # Manifold fallback: if scoped search is weak, search by geometry instead
        if effective and (not results or results[0].score < 0.55):
            fallback = self._manifold_fallback(embedding, top_k)
            seen_ids = {f.id for f in results}
            for f in fallback:
                if f.id not in seen_ids:
                    results.append(f)

        # Remote shard search (skip when scoped — remotes have no tag filtering)
        if not effective:
            seen = {f.text for f in results}
            for rs in self.remote_shards:
                try:
                    remote_results = rs.search_text(query, top_k=top_k)
                    for f in remote_results:
                        if f.text not in seen:
                            seen.add(f.text)
                            results.append(f)
                except Exception:
                    pass

        # Sort by score, trim
        results.sort(key=lambda f: f.score, reverse=True)
        final = results[:top_k * 2]

        # Record co-occurrences so manifold affinities improve over time
        self.local_shard.record_co_occurrences([f.id for f in final[:top_k]])

        return final

    def _manifold_rerank(
        self,
        facts: list[Fact],
        query_embedding,
    ) -> list[Fact]:
        """Re-rank facts using manifold distance + co-occurrence affinity.

        For facts in clusters with computed manifolds, blend three signals:
            final = alpha * cosine + beta * manifold + gamma * co_occurrence

        Facts in clusters without manifolds get cosine + co-occurrence only.
        Co-occurrence boosts facts that historically appear together in recalls.
        """
        import numpy as np

        alpha, beta, gamma = 0.55, 0.30, 0.15

        if not facts:
            return facts

        # Build co-occurrence lookup: fact_id → mean co-occurrence score
        all_ids = [f.id for f in facts]
        co_scores: dict[str, float] = {fid: 0.0 for fid in all_ids}
        id_set = set(all_ids)
        for i, f in enumerate(facts):
            for j, g in enumerate(facts):
                if i != j and f.id in id_set and g.id in id_set:
                    # Use position proximity as a cheap co-occurrence proxy
                    co_scores[f.id] += 1.0 / (1.0 + abs(i - j))
        # Normalize co-occurrence scores
        max_co = max(co_scores.values()) if co_scores else 1.0
        if max_co > 0:
            co_scores = {k: v / max_co for k, v in co_scores.items()}

        # Group facts by cluster for manifold re-ranking
        by_cluster: dict[int, list[Fact]] = {}
        for f in facts:
            by_cluster.setdefault(f.cluster_id, []).append(f)

        for cid, cluster_facts in by_cluster.items():
            cluster = self.cluster_engine.clusters.get(cid)
            if cluster is None or cluster.manifold_coords is None:
                # No manifold — blend cosine + co-occurrence only
                for f in cluster_facts:
                    co = co_scores.get(f.id, 0.0)
                    f.score = (alpha + beta) * f.score + gamma * co
                continue

            # Project query into manifold space via nearest neighbour
            nearest = self.cluster_engine.nearest_in_cluster(cid, query_embedding, k=1)
            if not nearest:
                continue
            nearest_idx, _ = nearest[0]
            if nearest_idx >= len(cluster.manifold_coords):
                continue
            query_manifold_pos = cluster.manifold_coords[nearest_idx]

            fid_to_idx = {fid: i for i, fid in enumerate(cluster.fact_ids)}

            for f in cluster_facts:
                midx = fid_to_idx.get(f.id)
                if midx is None or midx >= len(cluster.manifold_coords):
                    continue
                manifold_dist = float(np.linalg.norm(
                    cluster.manifold_coords[midx] - query_manifold_pos
                ))
                manifold_score = 1.0 / (1.0 + manifold_dist)
                co = co_scores.get(f.id, 0.0)
                f.score = alpha * f.score + beta * manifold_score + gamma * co

        return facts

    def _manifold_fallback(
        self, query_embedding, top_k: int,
    ) -> list[Fact]:
        """When scoped search fails, use manifold geometry to find relevant facts.

        Routes the query to the top clusters, projects it into manifold space,
        and retrieves the nearest facts by manifold distance — completely
        ignoring project scope. The manifold's structure captures semantic
        relationships that cosine similarity + scope filtering misses.
        """
        fallback_facts: list[Fact] = []
        seen_ids: set[str] = set()

        top_clusters = self.cluster_engine.top_clusters(query_embedding, k=3)
        for cid, sim in top_clusters:
            if sim < 0.3:
                continue
            self.cluster_engine.ensure_manifold(cid, self.local_shard)
            cluster_facts = self.cluster_engine.knn_at_coords(
                cid,
                coords=self._query_manifold_coords(cid, query_embedding),
                k=top_k,
                shard=self.local_shard,
            )
            for f in cluster_facts:
                if f.id not in seen_ids:
                    f.score *= 0.85  # discount vs direct matches
                    f.tags["_manifold_fallback"] = "true"
                    fallback_facts.append(f)
                    seen_ids.add(f.id)

        return fallback_facts

    def _query_manifold_coords(self, cluster_id: int, query_embedding) -> "np.ndarray":
        """Project a query embedding into a cluster's manifold coordinate space."""
        import numpy as np
        cluster = self.cluster_engine.clusters.get(cluster_id)
        if cluster is None or cluster.manifold_coords is None:
            return np.zeros(12)
        nearest = self.cluster_engine.nearest_in_cluster(cluster_id, query_embedding, k=1)
        if not nearest:
            return cluster.manifold_coords[0] if len(cluster.manifold_coords) > 0 else np.zeros(12)
        idx, _ = nearest[0]
        if idx < len(cluster.manifold_coords):
            return cluster.manifold_coords[idx]
        return np.zeros(cluster.manifold_coords.shape[1])

    # ------------------------------------------------------------------
    # Analyze — gap detection
    # ------------------------------------------------------------------

    def analyze(
        self,
        observations: list[str],
        objective: str,
        scope: dict[str, str] | None = None,
    ) -> GapAnalysis:
        """Full gap analysis: embed, route, geodesic, gaps, retrieve."""
        effective = self._effective_scope(scope)
        return gap_engine.analyze(
            observations=observations,
            objective=objective,
            cluster_engine=self.cluster_engine,
            shard=self.local_shard,
            embedder=self.embedder,
            remote_shards=self.remote_shards if not effective else [],
            scope=effective,
        )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def cluster_status(self) -> dict:
        return self.cluster_engine.status()

    def health(self) -> dict:
        shard_health = [self.local_shard.health()]
        for rs in self.remote_shards:
            shard_health.append(rs.health())
        return {
            "facts": self.local_shard.count(),
            "clusters": len(self.cluster_engine.clusters),
            "shards": shard_health,
        }

    # ------------------------------------------------------------------
    # Self-regulation
    # ------------------------------------------------------------------

    def warm_scope(self, scope: dict[str, str]) -> dict:
        """Warm up all facts matching a scope — resets their decay anchor
        to now, as if they were just accessed. Call this when returning to
        a project after time away.

        This is called automatically by set_session().
        """
        now = time.time()
        # Find all facts matching the scope
        rows = self.local_shard.conn.execute(
            "SELECT id, tags_json FROM facts"
        ).fetchall()
        warmed = []
        for row in rows:
            tags = __import__("json").loads(row["tags_json"]) if row["tags_json"] else {}
            match = all(tags.get(k) == v for k, v in scope.items())
            if match:
                warmed.append(row["id"])

        if warmed:
            self.local_shard.conn.executemany(
                "UPDATE facts SET last_accessed_at = ? WHERE id = ?",
                [(now, fid) for fid in warmed],
            )
            self.local_shard.conn.commit()

        logger.info(f"Warmed {len(warmed)} facts for scope {scope}")
        return {"warmed": len(warmed), "scope": scope}

    # ------------------------------------------------------------------
    # Briefing system — short-term memory distillation
    # ------------------------------------------------------------------

    def distill(self, project: str) -> Briefing | None:
        """Distill recent facts for a project into a compressed briefing.

        Gathers facts from the last 24h, groups by type, and templates
        them into a 200-500 token summary. Stores both as a briefing
        record and as a searchable fact.
        """
        import re

        now = time.time()
        since = now - 86400  # last 24 hours
        facts = self.local_shard.get_recent_facts_for_project(project, since=since, limit=30)

        if not facts:
            logger.info(f"Distill {project}: no recent facts, skipping")
            return None

        # Group facts by type tag
        groups: dict[str, list[str]] = {}
        fact_ids = []
        for f in facts:
            fact_type = f.tags.get("type", "general")
            # Strip [project-name] prefix from text
            text = re.sub(r'^\[[\w-]+\]\s*', '', f.text).strip()
            # Strip leading labels like "Decision: ", "Validated: ", "Open: "
            text = re.sub(r'^(Decision|Validated|Open|Actions|Credentials|Attack Chain):\s*', '', text)
            if text:
                groups.setdefault(fact_type, []).append(text)
                fact_ids.append(f.id)

        # Build briefing from template
        sections = []

        # Actions / what happened
        actions = groups.get("action", [])
        if actions:
            sections.append("Did: " + "; ".join(a[:150] for a in actions[:4]))

        # Decisions
        decisions = groups.get("decision", [])
        if decisions:
            sections.append("Decided: " + "; ".join(d[:150] for d in decisions[:4]))

        # Validations
        validations = groups.get("validation", [])
        if validations:
            sections.append("Confirmed: " + "; ".join(v[:120] for v in validations[:3]))

        # Open threads / next steps
        next_steps = groups.get("next_steps", [])
        if next_steps:
            sections.append("Open: " + "; ".join(n[:150] for n in next_steps[:3]))

        # Conversation facts (condensed)
        convos = groups.get("conversation", [])
        if convos:
            sections.append("Discussed: " + "; ".join(c[:120] for c in convos[:3]))

        # Anything else
        for gtype, items in groups.items():
            if gtype in ("action", "decision", "validation", "next_steps",
                         "conversation", "briefing"):
                continue
            if items:
                label = gtype.replace("_", " ").title()
                sections.append(f"{label}: " + "; ".join(i[:120] for i in items[:3]))

        if not sections:
            return None

        briefing_text = " | ".join(sections)
        # Truncate to ~2000 chars if needed
        if len(briefing_text) > 2000:
            briefing_text = briefing_text[:1997] + "..."

        briefing = Briefing(
            project=project,
            text=briefing_text,
            facts_used=fact_ids,
        )

        # Store briefing record
        self.local_shard.store_briefing(briefing)

        # Also store as a fact so it gets clustered and is searchable
        self.remember(
            briefing_text,
            tags={"type": "briefing", "project": project},
            source=f"briefing:{project}",
        )

        logger.info(
            f"Distilled {project}: {len(facts)} facts → {briefing.token_count} tokens"
        )
        return briefing

    def get_briefing(self, project: str) -> Briefing:
        """Get the most recent briefing for a project.

        Falls back to top-5 recent facts if no briefing exists.
        """
        briefing = self.local_shard.get_latest_briefing(project)
        if briefing:
            return briefing

        # Fallback: synthesize from recent facts
        facts = self.local_shard.get_recent_facts_for_project(project, since=0, limit=5)
        if not facts:
            return Briefing(
                id="fallback",
                project=project,
                text="No prior context for this project.",
                is_fallback=True,
            )

        text = " | ".join(f.text[:200] for f in facts)
        return Briefing(
            id="fallback",
            project=project,
            text=text,
            facts_used=[f.id for f in facts],
            is_fallback=True,
        )

    # ------------------------------------------------------------------
    # Self-regulation
    # ------------------------------------------------------------------

    def maintain(self) -> dict:
        """Periodic self-maintenance. Run automatically or call manually.

        - Merges overlapping clusters
        - Builds manifolds for dirty clusters with enough facts
        - Auto-distills active session if new facts exist
        - Reports decay statistics
        """
        t0 = time.time()
        now = time.time()
        report: dict = {"timestamp": now}

        # Merge clusters
        merged = self.cluster_engine.maybe_merge()
        if merged:
            self.cluster_engine.save_to_shard(self.local_shard)
            for absorbed, into in merged:
                cluster = self.cluster_engine.clusters.get(into)
                if cluster:
                    for fid in cluster.fact_ids:
                        self.local_shard.update_cluster_id(fid, into)
        report["clusters_merged"] = len(merged) if merged else 0
        report["total_clusters"] = len(self.cluster_engine.clusters)

        # Build manifolds for dirty clusters
        manifolds_built = 0
        for cid, cluster in self.cluster_engine.clusters.items():
            if cluster.dirty and cluster.count >= MIN_MANIFOLD:
                if self.cluster_engine.build_manifold(cid, self.local_shard):
                    manifolds_built += 1
        report["manifolds_built"] = manifolds_built
        report["manifolds_total"] = sum(
            1 for c in self.cluster_engine.clusters.values()
            if c.manifold_coords is not None
        )

        # Auto-distill active session if there are undistilled facts
        report["auto_distilled"] = None
        if self._session_scope and "project" in self._session_scope:
            proj = self._session_scope["project"]
            last_briefing = self.local_shard.get_last_briefing_time(proj)
            recent = self.local_shard.get_recent_facts_for_project(
                proj, since=max(last_briefing, now - 3600), limit=1,
            )
            if recent:
                briefing = self.distill(proj)
                if briefing:
                    report["auto_distilled"] = proj

        # Decay stats
        rows = self.local_shard.conn.execute(
            "SELECT COUNT(*) as c, "
            "AVG(access_count) as avg_access, "
            "MAX(access_count) as max_access, "
            "SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) as never_accessed "
            "FROM facts"
        ).fetchone()
        report["total_facts"] = rows["c"]
        report["avg_access_count"] = round(rows["avg_access"] or 0, 2)
        report["max_access_count"] = rows["max_access"] or 0
        report["never_accessed"] = rows["never_accessed"] or 0

        report["elapsed_sec"] = round(time.time() - t0, 3)
        self._last_maintain = now
        logger.info(f"Maintenance complete: {report}")
        return report

    def _maybe_maintain(self):
        """Auto-trigger maintenance if enough time has passed."""
        if time.time() - self._last_maintain > self._maintain_interval:
            self.maintain()

    # ------------------------------------------------------------------
    # Background manifold builder (#5 + #6 startup warmup)
    # ------------------------------------------------------------------

    def _background_manifold_builder(self):
        """Background thread: builds manifolds for dirty clusters.

        Runs immediately on startup (warmup), then every 60s thereafter.
        """
        # Brief pause to let the server finish starting
        self._bg_stop.wait(2)

        while not self._bg_stop.is_set():
            try:
                built = 0
                # Sort by count descending — build biggest clusters first
                candidates = sorted(
                    self.cluster_engine.clusters.values(),
                    key=lambda c: c.count,
                    reverse=True,
                )
                for cluster in candidates:
                    if self._bg_stop.is_set():
                        break
                    if cluster.dirty and cluster.count >= MIN_MANIFOLD:
                        if self.cluster_engine.build_manifold(cluster.id, self.local_shard):
                            built += 1
                if built:
                    logger.info(f"Background builder: {built} manifolds computed")
            except Exception as e:
                logger.warning(f"Background manifold builder error: {e}")

            # Wait 60s before next sweep (interruptible)
            self._bg_stop.wait(60)

    def close(self):
        self._bg_stop.set()
        self._bg_thread.join(timeout=5)
        self.cluster_engine.save_to_shard(self.local_shard)
        self.local_shard.close()
