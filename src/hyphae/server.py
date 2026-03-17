"""Hyphae API server — FastAPI on localhost."""
from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger("hyphae.server")

app = FastAPI(title="Hyphae", description="Self-organizing memory with gap-aware retrieval")

# Global state — initialized lazily
_hyphae = None


def get_hyphae():
    global _hyphae
    if _hyphae is None:
        from . import Hyphae
        _hyphae = Hyphae()
    return _hyphae


# --- Request models ---

class RememberRequest(BaseModel):
    text: str
    context_id: str = ""
    tags: dict[str, str] = Field(default_factory=dict)
    source: str = ""
    cause_of: str = ""  # fact ID this was caused by


class RecallRequest(BaseModel):
    query: str
    top_k: int = 10
    scope: Optional[dict[str, str]] = None


class AnalyzeRequest(BaseModel):
    observations: list[str]
    objective: str
    scope: Optional[dict[str, str]] = None


class ConverseRequest(BaseModel):
    role: str  # "user" or "assistant"
    message: str
    source: str = "conversation"


class CreateRiverRequest(BaseModel):
    objective: str
    stones: list[dict] = Field(
        ..., description="List of {goal, success_marker, failure_marker}"
    )


class AttemptRequest(BaseModel):
    output: str
    context: dict = Field(default_factory=dict)
    river_id: str = ""


class SuccessRequest(BaseModel):
    river_id: str = ""


class InsertStoneRequest(BaseModel):
    goal: str
    success_marker: str
    failure_marker: str = ""
    river_id: str = ""


class SessionRequest(BaseModel):
    scope: dict[str, str]


class DistillRequest(BaseModel):
    project: str


# --- Session management ---


# Tower API
try:
    from hyphae.tower_api import tower_router
    app.include_router(tower_router)
except ImportError:
    pass

@app.post("/session/set")
def set_session(req: SessionRequest):
    """Set the active session scope. Auto-warms all matching facts."""
    h = get_hyphae()
    return h.set_session(req.scope)


@app.post("/session/clear")
def clear_session():
    """Clear session scope. Recalls become unscoped with time decay."""
    h = get_hyphae()
    return h.clear_session()


@app.get("/session")
def get_session():
    """Get the current session scope."""
    h = get_hyphae()
    return {"session_scope": h._session_scope}


# --- Briefing endpoints ---

@app.post("/distill")
def distill(req: DistillRequest):
    """Distill recent facts for a project into a compressed briefing."""
    h = get_hyphae()
    briefing = h.distill(req.project)
    if not briefing:
        return {"briefing": None, "facts_used": 0, "token_count": 0}
    return {
        "briefing": briefing.text,
        "briefing_id": briefing.id,
        "facts_used": len(briefing.facts_used),
        "token_count": briefing.token_count,
        "created_at": briefing.created_at,
    }


@app.get("/briefing/{project}")
def get_briefing(project: str):
    """Get the most recent briefing for a project. Falls back to recent facts."""
    h = get_hyphae()
    briefing = h.get_briefing(project)
    return {
        "project": briefing.project,
        "briefing": briefing.text,
        "briefing_id": briefing.id,
        "created_at": briefing.created_at,
        "token_count": briefing.token_count,
        "facts_used": briefing.facts_used,
        "is_fallback": briefing.is_fallback,
    }


# --- Memory endpoints ---

@app.post("/remember")
def remember(req: RememberRequest):
    h = get_hyphae()
    fact_id, cluster_id = h.remember(
        req.text,
        context_id=req.context_id,
        tags=req.tags,
        source=req.source,
        cause_of=req.cause_of,
    )
    # Tower organic growth — check if this fact belongs near a tower
    try:
        from hyphae.tower_hook import check_tower_proximity
        check_tower_proximity(req.text)
    except Exception:
        pass


    return {"fact_id": fact_id, "cluster_id": cluster_id}


@app.post("/recall")
def recall(req: RecallRequest):
    h = get_hyphae()
    facts = h.recall(req.query, top_k=req.top_k, scope=req.scope)
    return {
        "results": [
            {"text": f.text, "score": f.score, "cluster_id": f.cluster_id,
             "source": f.source, "tags": f.tags}
            for f in facts
        ]
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    h = get_hyphae()
    analysis = h.analyze(req.observations, req.objective, scope=req.scope)
    return {
        "clusters_used": analysis.clusters_used,
        "gaps": [
            {
                "position": g.position,
                "edge_cost": g.edge_cost,
                "normalized_cost": g.normalized_cost,
                "from_text": g.from_fact_text[:200],
                "to_text": g.to_fact_text[:200],
                "cluster_id": g.cluster_id,
                "retrieved_count": len(g.retrieved_facts),
                "retrieved": [
                    {"text": f.text[:300], "score": f.score}
                    for f in g.retrieved_facts[:5]
                ],
            }
            for g in analysis.gaps
        ],
        "retrieved_knowledge": [
            {"text": f.text[:300], "score": f.score, "source": f.source}
            for f in analysis.retrieved_knowledge[:20]
        ],
        "total_knowledge": len(analysis.retrieved_knowledge),
        "elapsed_sec": analysis.elapsed_sec,
    }


# --- Conversation processing ---

@app.post("/converse")
def converse(req: ConverseRequest):
    """Process a conversation turn. Extracts facts, gates by novelty, stores.

    Call this with each user/assistant message. The system decides what's
    worth remembering — callers don't need to filter.
    """
    from . import converse as conv
    h = get_hyphae()
    stored = conv.process_turn(req.role, req.message, h, source=req.source)
    return {"stored": stored, "count": len(stored)}


# --- Cluster endpoints ---

@app.get("/clusters")
def clusters():
    h = get_hyphae()
    return h.cluster_status()


@app.post("/clusters/merge")
def merge_clusters():
    h = get_hyphae()
    merged = h.cluster_engine.maybe_merge()
    if merged:
        h.cluster_engine.save_to_shard(h.local_shard)
        # Update fact cluster_ids for merged clusters
        for absorbed, into in merged:
            for fid in h.cluster_engine.clusters.get(into, h.cluster_engine.clusters.get(absorbed, None)).fact_ids:
                h.local_shard.update_cluster_id(fid, into)
    return {"merged": merged, "total_clusters": len(h.cluster_engine.clusters)}


# --- River endpoints ---

@app.post("/river/create")
def create_river(req: CreateRiverRequest):
    h = get_hyphae()
    return h.river_manager.create(req.objective, req.stones)


@app.get("/river/status")
def river_status(river_id: str = ""):
    h = get_hyphae()
    return h.river_manager.status(river_id)


@app.post("/river/attempt")
def river_attempt(req: AttemptRequest):
    h = get_hyphae()
    return h.river_manager.attempt(req.output, req.context, req.river_id)


@app.post("/river/success")
def river_success(req: SuccessRequest):
    h = get_hyphae()
    return h.river_manager.success(req.river_id)


@app.post("/river/insert")
def river_insert(req: InsertStoneRequest):
    h = get_hyphae()
    return h.river_manager.insert(
        req.goal, req.success_marker, req.failure_marker, req.river_id
    )


@app.get("/river/history")
def river_history(stone_id: str = "", river_id: str = ""):
    h = get_hyphae()
    return h.river_manager.history(stone_id, river_id)


# --- Maintenance ---

@app.post("/maintain")
def maintain():
    """Trigger self-maintenance: merge clusters, report decay stats."""
    h = get_hyphae()
    return h.maintain()


# --- Scopes ---

@app.get("/scopes")
def scopes():
    """List available scope values (distinct tag key/value pairs)."""
    h = get_hyphae()
    rows = h.local_shard.conn.execute(
        "SELECT tags_json FROM facts WHERE tags_json != '{}'"
    ).fetchall()
    scope_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        tags = json.loads(row["tags_json"]) if row["tags_json"] else {}
        for k, v in tags.items():
            if k not in scope_counts:
                scope_counts[k] = {}
            scope_counts[k][v] = scope_counts[k].get(v, 0) + 1
    return scope_counts


# --- Health ---

@app.get("/health")
def health():
    h = get_hyphae()
    shard_health = [s.health() for s in [h.local_shard] + h.remote_shards]
    return {
        "status": "ok",
        "service": "hyphae",
        "facts": h.local_shard.count(),
        "clusters": len(h.cluster_engine.clusters),
        "shards": shard_health,
    }


@app.get("/stats")
def stats():
    """Comprehensive stats: manifold coverage, cluster distribution, decay, FAISS status."""
    h = get_hyphae()
    clusters = h.cluster_engine.clusters

    # Cluster size distribution
    counts = sorted([c.count for c in clusters.values()], reverse=True)
    manifold_count = sum(1 for c in clusters.values() if c.manifold_coords is not None)
    dirty_count = sum(1 for c in clusters.values() if c.dirty)
    eligible = sum(1 for c in clusters.values() if c.count >= 30)

    # Decay stats
    row = h.local_shard.conn.execute(
        "SELECT COUNT(*) as total, "
        "AVG(access_count) as avg_access, "
        "MAX(access_count) as max_access, "
        "SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) as never_accessed, "
        "MIN(created_at) as oldest, "
        "MAX(created_at) as newest "
        "FROM facts"
    ).fetchone()

    # FAISS status
    faiss_status = "not_built"
    faiss_vectors = 0
    if h.local_shard._faiss_index is not None:
        faiss_status = "active"
        faiss_vectors = h.local_shard._faiss_index.ntotal
    elif h.local_shard._faiss_dirty:
        faiss_status = "dirty"

    return {
        "facts": {
            "total": row["total"],
            "avg_access_count": round(row["avg_access"] or 0, 2),
            "max_access_count": row["max_access"] or 0,
            "never_accessed": row["never_accessed"] or 0,
            "oldest_timestamp": row["oldest"],
            "newest_timestamp": row["newest"],
        },
        "clusters": {
            "total": len(clusters),
            "largest": counts[0] if counts else 0,
            "smallest": counts[-1] if counts else 0,
            "median": counts[len(counts) // 2] if counts else 0,
            "singleton_count": sum(1 for c in counts if c <= 1),
            "size_distribution": {
                "1": sum(1 for c in counts if c == 1),
                "2-5": sum(1 for c in counts if 2 <= c <= 5),
                "6-29": sum(1 for c in counts if 6 <= c <= 29),
                "30-99": sum(1 for c in counts if 30 <= c <= 99),
                "100+": sum(1 for c in counts if c >= 100),
            },
        },
        "manifolds": {
            "built": manifold_count,
            "eligible": eligible,
            "dirty": dirty_count,
            "coverage_pct": round(manifold_count / eligible * 100, 1) if eligible else 0,
        },
        "faiss": {
            "status": faiss_status,
            "vectors": faiss_vectors,
        },
        "background_builder": {
            "alive": h._bg_thread.is_alive(),
        },
    }


def run(host: str = "127.0.0.1", port: int = 8100):
    """Start the Hyphae server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
