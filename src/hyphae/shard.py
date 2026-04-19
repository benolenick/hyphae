"""Pluggable storage backends (shards).

Each shard stores facts + embeddings and provides vector search.
Multiple shards can be active — the cluster engine spans all of them.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .types import Fact, CausalLink, Briefing

# --- Time decay constants ---
HALF_LIFE_DAYS = 30.0       # unaccessed fact loses half its recency weight in 30 days
ACCESS_BOOST_ALPHA = 0.15   # how much each log(access_count) boosts the score

logger = logging.getLogger("hyphae.shard")


def _tags_match(fact_tags: dict[str, str], scope: dict[str, str]) -> bool:
    """Return True if fact_tags contain all key/value pairs in scope."""
    for k, v in scope.items():
        if fact_tags.get(k) != v:
            return False
    return True


def _decay_score(
    vector_sim: float,
    created_at: float,
    last_accessed_at: float,
    access_count: int,
    now: float,
    scoped: bool,
) -> float:
    """Blend vector similarity with time decay and access boost.

    When scoped (caller specified a project/profile filter), time decay
    is skipped — the scope itself is the relevance signal.

    Unscoped queries apply exponential decay based on the more recent of
    created_at and last_accessed_at, so recalled facts stay warm.

    Access boost: 1 + alpha * log(1 + access_count)
    — diminishing returns, but frequently recalled facts gain prominence.
    """
    access_boost = 1.0 + ACCESS_BOOST_ALPHA * math.log1p(access_count)

    if scoped:
        return vector_sim * access_boost

    # Decay from whichever is more recent: creation or last access
    # last_accessed_at == 0 means never accessed — use created_at only
    if last_accessed_at > 0:
        anchor = max(created_at, last_accessed_at)
    else:
        anchor = created_at if created_at > 0 else now
    age_days = max(0.0, (now - anchor) / 86400.0)
    recency = math.exp(-math.log(2) * age_days / HALF_LIFE_DAYS)

    return vector_sim * recency * access_boost


class Shard(ABC):
    """Abstract storage backend."""

    name: str

    @abstractmethod
    def store(self, fact: Fact) -> str:
        """Persist a fact with its embedding. Returns fact ID."""

    @abstractmethod
    def search(self, embedding: np.ndarray, top_k: int = 10) -> list[Fact]:
        """Vector nearest-neighbor search. Returns facts with scores."""

    @abstractmethod
    def get(self, fact_id: str) -> Fact | None:
        """Retrieve a fact by ID."""

    @abstractmethod
    def count(self) -> int:
        """Total facts stored."""

    @abstractmethod
    def all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Return (fact_ids, embeddings_matrix) for all facts."""

    def health(self) -> dict:
        return {"name": self.name, "count": self.count(), "status": "ok"}


class LocalShard(Shard):
    """SQLite + FAISS local storage."""

    def __init__(self, db_path: str = "hyphae.db", name: str = "local"):
        self.name = name
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._faiss_index = None
        self._faiss_ids: list[str] = []
        self._faiss_dirty = True

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                tags_json TEXT DEFAULT '{}',
                source TEXT DEFAULT '',
                context_id TEXT DEFAULT '',
                cluster_id INTEGER DEFAULT -1,
                created_at REAL,
                last_accessed_at REAL DEFAULT 0,
                access_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                fact_id TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                dim INTEGER NOT NULL,
                FOREIGN KEY (fact_id) REFERENCES facts(id)
            );

            CREATE TABLE IF NOT EXISTS causal_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_fact_id TEXT NOT NULL,
                to_fact_id TEXT NOT NULL,
                relation TEXT DEFAULT 'caused_by',
                context_id TEXT DEFAULT '',
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (from_fact_id) REFERENCES facts(id),
                FOREIGN KEY (to_fact_id) REFERENCES facts(id)
            );

            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY,
                centroid BLOB NOT NULL,
                count INTEGER DEFAULT 0,
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS briefings (
                id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                text TEXT NOT NULL,
                facts_used TEXT DEFAULT '[]',
                created_at REAL,
                token_count INTEGER DEFAULT 0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
                USING fts5(text, content=facts, content_rowid=rowid);

            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, text)
                    VALUES('delete', old.rowid, old.text);
            END;
        """)
        # Migrate existing DBs: add columns if missing
        self._migrate_add_column("facts", "last_accessed_at", "REAL DEFAULT 0")
        self._migrate_add_column("facts", "access_count", "INTEGER DEFAULT 0")
        self.conn.commit()

    def _migrate_add_column(self, table: str, column: str, col_type: str):
        """Add a column if it doesn't exist (idempotent migration)."""
        try:
            self.conn.execute(f"SELECT {column} FROM {table} LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            logger.info(f"Migrated: added {table}.{column}")

    def _build_faiss_index(self):
        """Build FAISS index from all stored embeddings."""
        try:
            import faiss
        except ImportError:
            logger.warning("faiss-cpu not installed, vector search will use numpy fallback")
            self._faiss_index = None
            return

        rows = self.conn.execute(
            "SELECT fact_id, data, dim FROM embeddings"
        ).fetchall()

        if not rows:
            dim = 384  # default
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_ids = []
            self._faiss_dirty = False
            return

        dim = rows[0]["dim"]
        self._faiss_ids = []
        vectors = []
        for row in rows:
            self._faiss_ids.append(row["fact_id"])
            vec = np.frombuffer(row["data"], dtype=np.float32).copy()
            vectors.append(vec)

        matrix = np.stack(vectors)
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(matrix)
        self._faiss_dirty = False
        logger.info(f"FAISS index built: {len(rows)} vectors, {dim}D")

    def _ensure_faiss(self):
        if self._faiss_dirty or self._faiss_index is None:
            self._build_faiss_index()

    def store(self, fact: Fact) -> str:
        """Store fact + embedding. Deduplicates by content hash ID."""
        # Check for duplicate
        existing = self.conn.execute(
            "SELECT id FROM facts WHERE id = ?", (fact.id,)
        ).fetchone()
        if existing:
            return fact.id

        self.conn.execute(
            "INSERT INTO facts (id, text, tags_json, source, context_id, cluster_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fact.id, fact.text, json.dumps(fact.tags), fact.source,
             fact.context_id, fact.cluster_id, fact.created_at),
        )

        if fact.embedding is not None:
            self.conn.execute(
                "INSERT INTO embeddings (fact_id, data, dim) VALUES (?, ?, ?)",
                (fact.id, fact.embedding.tobytes(), len(fact.embedding)),
            )
            # Incremental FAISS update
            if self._faiss_index is not None and not self._faiss_dirty:
                self._faiss_index.add(fact.embedding.reshape(1, -1))
                self._faiss_ids.append(fact.id)
            else:
                self._faiss_dirty = True

        self.conn.commit()
        return fact.id

    def store_link(self, link: CausalLink) -> None:
        """Store a causal link between facts."""
        self.conn.execute(
            "INSERT INTO causal_links (from_fact_id, to_fact_id, relation, context_id, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (link.from_fact_id, link.to_fact_id, link.relation,
             link.context_id, link.confidence),
        )
        self.conn.commit()

    def get_links(self, fact_id: str) -> list[CausalLink]:
        """Get all causal links involving a fact."""
        rows = self.conn.execute(
            "SELECT * FROM causal_links WHERE from_fact_id = ? OR to_fact_id = ?",
            (fact_id, fact_id),
        ).fetchall()
        return [
            CausalLink(
                from_fact_id=r["from_fact_id"],
                to_fact_id=r["to_fact_id"],
                relation=r["relation"],
                context_id=r["context_id"],
                confidence=r["confidence"],
            )
            for r in rows
        ]

    def search(
        self, embedding: np.ndarray, top_k: int = 10,
        scope: dict[str, str] | None = None,
    ) -> list[Fact]:
        """Vector search via FAISS (or numpy fallback).

        If scope is provided (e.g. {"project": "htb-autopwn"}), results are
        filtered to facts whose tags contain all the given key/value pairs.
        Over-fetches to have enough candidates for post-filtering and re-ranking.

        Time decay is applied to unscoped queries only. Scoped queries skip
        decay (the scope itself signals active interest in that topic).

        All returned facts get their access bumped.
        """
        self._ensure_faiss()

        # Over-fetch: need candidates for scope filtering + decay re-ranking
        fetch_k = max(top_k * 8, 50)
        now = time.time()
        scoped = scope is not None

        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            candidates = []
            scores, indices = self._faiss_index.search(
                embedding.reshape(1, -1), min(fetch_k, self._faiss_index.ntotal)
            )
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._faiss_ids):
                    continue
                fact = self.get(self._faiss_ids[idx])
                if fact:
                    if scope and not _tags_match(fact.tags, scope):
                        continue
                    fact.score = _decay_score(
                        float(score), fact.created_at,
                        fact.last_accessed_at, fact.access_count,
                        now, scoped,
                    )
                    candidates.append(fact)

            # Re-sort by decayed score and take top_k
            candidates.sort(key=lambda f: f.score, reverse=True)
            results = candidates[:top_k]
            self.bump_access([f.id for f in results])
            return results

        # Numpy fallback
        return self._numpy_search(embedding, top_k, scope=scope)

    def _numpy_search(
        self, embedding: np.ndarray, top_k: int,
        scope: dict[str, str] | None = None,
    ) -> list[Fact]:
        """Fallback search without FAISS."""
        ids, matrix = self.all_embeddings()
        if len(ids) == 0:
            return []
        sims = matrix @ embedding
        now = time.time()
        scoped = scope is not None

        # Gather candidates (over-fetch for re-ranking)
        fetch_k = max(top_k * 8, 50)
        top_idx = np.argsort(-sims)[:fetch_k]
        candidates = []
        for idx in top_idx:
            fact = self.get(ids[idx])
            if fact:
                if scope and not _tags_match(fact.tags, scope):
                    continue
                fact.score = _decay_score(
                    float(sims[idx]), fact.created_at,
                    fact.last_accessed_at, fact.access_count,
                    now, scoped,
                )
                candidates.append(fact)

        candidates.sort(key=lambda f: f.score, reverse=True)
        results = candidates[:top_k]
        self.bump_access([f.id for f in results])
        return results

    def text_search(self, query: str, top_k: int = 10) -> list[Fact]:
        """FTS5 keyword search fallback."""
        rows = self.conn.execute(
            "SELECT f.* FROM facts f JOIN facts_fts fts ON f.rowid = fts.rowid "
            "WHERE facts_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, top_k),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def get(self, fact_id: str) -> Fact | None:
        row = self.conn.execute(
            "SELECT * FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_fact(row)

    def _row_to_fact(self, row) -> Fact:
        return Fact(
            id=row["id"],
            text=row["text"],
            tags=json.loads(row["tags_json"]) if row["tags_json"] else {},
            source=row["source"],
            context_id=row["context_id"],
            cluster_id=row["cluster_id"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"] or 0.0,
            access_count=row["access_count"] or 0,
        )

    def bump_access(self, fact_ids: list[str]) -> None:
        """Record that these facts were accessed (recalled). Batched."""
        if not fact_ids:
            return
        now = time.time()
        self.conn.executemany(
            "UPDATE facts SET last_accessed_at = ?, access_count = access_count + 1 "
            "WHERE id = ?",
            [(now, fid) for fid in fact_ids],
        )
        self.conn.commit()

    def update_cluster_id(self, fact_id: str, cluster_id: int) -> None:
        self.conn.execute(
            "UPDATE facts SET cluster_id = ? WHERE id = ?", (cluster_id, fact_id)
        )
        self.conn.commit()

    def get_cluster_facts(self, cluster_id: int) -> list[Fact]:
        """Get all facts in a cluster."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE cluster_id = ?", (cluster_id,)
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def get_cluster_embeddings(self, cluster_id: int) -> tuple[list[str], np.ndarray]:
        """Get fact IDs and embeddings for a cluster."""
        rows = self.conn.execute(
            "SELECT f.id, e.data, e.dim FROM facts f "
            "JOIN embeddings e ON f.id = e.fact_id "
            "WHERE f.cluster_id = ?", (cluster_id,)
        ).fetchall()
        if not rows:
            return [], np.array([])
        ids = [r["id"] for r in rows]
        vecs = [np.frombuffer(r["data"], dtype=np.float32).copy() for r in rows]
        return ids, np.stack(vecs)

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as c FROM facts").fetchone()
        return row["c"]

    def all_embeddings(self) -> tuple[list[str], np.ndarray]:
        rows = self.conn.execute(
            "SELECT fact_id, data FROM embeddings"
        ).fetchall()
        if not rows:
            return [], np.array([])
        ids = [r["fact_id"] for r in rows]
        vecs = [np.frombuffer(r["data"], dtype=np.float32).copy() for r in rows]
        return ids, np.stack(vecs)

    def save_cluster(self, cluster_id: int, centroid: np.ndarray, count: int) -> None:
        """Persist cluster centroid."""
        self.conn.execute(
            "INSERT OR REPLACE INTO clusters (id, centroid, count, created_at) "
            "VALUES (?, ?, ?, ?)",
            (cluster_id, centroid.tobytes(), count, time.time()),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Briefing helpers
    # ------------------------------------------------------------------

    def store_briefing(self, briefing: Briefing) -> str:
        """Store a briefing. Returns briefing ID."""
        self.conn.execute(
            "INSERT OR REPLACE INTO briefings (id, project, text, facts_used, created_at, token_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (briefing.id, briefing.project, briefing.text,
             json.dumps(briefing.facts_used), briefing.created_at, briefing.token_count),
        )
        self.conn.commit()
        return briefing.id

    def get_latest_briefing(self, project: str) -> Briefing | None:
        """Get the most recent briefing for a project."""
        row = self.conn.execute(
            "SELECT * FROM briefings WHERE project = ? ORDER BY created_at DESC LIMIT 1",
            (project,),
        ).fetchone()
        if not row:
            return None
        return Briefing(
            id=row["id"],
            project=row["project"],
            text=row["text"],
            facts_used=json.loads(row["facts_used"]) if row["facts_used"] else [],
            created_at=row["created_at"],
            token_count=row["token_count"],
        )

    _has_json_extract: bool | None = None

    def get_recent_facts_for_project(
        self, project: str, since: float = 0, limit: int = 20,
    ) -> list[Fact]:
        """Get recent facts for a project, ordered by created_at desc."""
        # Check json_extract support once
        if LocalShard._has_json_extract is None:
            try:
                self.conn.execute("SELECT json_extract('{}', '$')")
                LocalShard._has_json_extract = True
            except sqlite3.OperationalError:
                LocalShard._has_json_extract = False

        if LocalShard._has_json_extract:
            query = (
                "SELECT * FROM facts WHERE json_extract(tags_json, '$.project') = ? "
                "AND created_at > ? ORDER BY created_at DESC LIMIT ?"
            )
            param = project
        else:
            query = (
                "SELECT * FROM facts WHERE tags_json LIKE ? "
                "AND created_at > ? ORDER BY created_at DESC LIMIT ?"
            )
            # Match both "project": "x" and "project":"x" (with/without space)
            param = f'%"project":%"{project}"%'

        rows = self.conn.execute(query, (param, since, limit)).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def get_last_briefing_time(self, project: str) -> float:
        """Get created_at of the most recent briefing for this project, or 0."""
        row = self.conn.execute(
            "SELECT created_at FROM briefings WHERE project = ? ORDER BY created_at DESC LIMIT 1",
            (project,),
        ).fetchone()
        return row["created_at"] if row else 0.0

    def record_co_occurrences(self, fact_ids: list[str]) -> None:
        """Increment co-occurrence count for every pair of facts in this recall."""
        if len(fact_ids) < 2:
            return
        now = time.time()
        pairs = [(fact_ids[i], fact_ids[j])
                 for i in range(len(fact_ids))
                 for j in range(i + 1, len(fact_ids))]
        try:
            self.conn.executemany(
                "INSERT INTO co_occurrences (fact_id_a, fact_id_b, count, last_seen) "
                "VALUES (?, ?, 1, ?) "
                "ON CONFLICT(fact_id_a, fact_id_b) DO UPDATE SET "
                "count = count + 1, last_seen = excluded.last_seen",
                [(a, b, now) for a, b in pairs],
            )
            self.conn.commit()
        except Exception:
            pass  # table may not exist in older DBs

    def get_co_occurrences_for_cluster(
        self, cluster_id: int, limit: int = 200,
    ) -> list[tuple[str, str, int]]:
        """Return (fact_id_a, fact_id_b, count) pairs for facts in this cluster."""
        try:
            rows = self.conn.execute(
                "SELECT co.fact_id_a, co.fact_id_b, co.count "
                "FROM co_occurrences co "
                "JOIN facts fa ON co.fact_id_a = fa.id "
                "JOIN facts fb ON co.fact_id_b = fb.id "
                "WHERE fa.cluster_id = ? AND fb.cluster_id = ? "
                "ORDER BY co.count DESC LIMIT ?",
                (cluster_id, cluster_id, limit),
            ).fetchall()
            return [(r["fact_id_a"], r["fact_id_b"], r["count"]) for r in rows]
        except Exception:
            return []

    def get_causal_links_for_cluster(
        self, cluster_id: int,
    ) -> list[tuple[str, str, float]]:
        """Return (from_fact_id, to_fact_id, confidence) for facts in this cluster."""
        try:
            rows = self.conn.execute(
                "SELECT cl.from_fact_id, cl.to_fact_id, cl.confidence "
                "FROM causal_links cl "
                "JOIN facts fa ON cl.from_fact_id = fa.id "
                "JOIN facts fb ON cl.to_fact_id = fb.id "
                "WHERE fa.cluster_id = ? AND fb.cluster_id = ?",
                (cluster_id, cluster_id),
            ).fetchall()
            return [(r["from_fact_id"], r["to_fact_id"], r["confidence"]) for r in rows]
        except Exception:
            return []

    def get_all_fact_cluster_ids(self) -> list[tuple[str, int]]:
        """Return (fact_id, cluster_id) for all facts. Used to populate cluster fact_ids on load."""
        rows = self.conn.execute(
            "SELECT id, cluster_id FROM facts WHERE cluster_id >= 0"
        ).fetchall()
        return [(r["id"], r["cluster_id"]) for r in rows]

    def load_clusters(self) -> dict[int, tuple[np.ndarray, int]]:
        """Load all cluster centroids. Returns {id: (centroid, count)}."""
        rows = self.conn.execute("SELECT * FROM clusters").fetchall()
        result = {}
        for r in rows:
            centroid = np.frombuffer(r["centroid"], dtype=np.float32).copy()
            result[r["id"]] = (centroid, r["count"])
        return result

    def close(self) -> None:
        self.conn.close()


class RemoteShard(Shard):
    """HTTP client to a remote Memoria-compatible server."""

    def __init__(self, endpoint: str = "http://127.0.0.1:8000",
                 name: str = "remote", timeout: int = 15):
        self.name = name
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict | None:
        import urllib.request
        import urllib.error
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.endpoint}{path}",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"RemoteShard({self.name}) {path} failed: {e}")
            return None

    def _get(self, path: str) -> dict | None:
        import urllib.request
        try:
            req = urllib.request.Request(f"{self.endpoint}{path}", method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"RemoteShard({self.name}) GET {path} failed: {e}")
            return None

    def store(self, fact: Fact) -> str:
        result = self._post("/memorize", {
            "text": fact.text,
            "metadata": {"tags": fact.tags, "source": fact.source, "context_id": fact.context_id},
        })
        return fact.id

    def search(self, embedding: np.ndarray, top_k: int = 10) -> list[Fact]:
        # Remote shards search by text query, not raw embedding
        # This gets called with an embedding; we need to convert back
        # For now, this shard is query-based via search_text
        return []

    def search_text(self, query: str, top_k: int = 10) -> list[Fact]:
        """Text-based search on remote shard."""
        result = self._post("/search", {"query": query, "top_k": top_k})
        if not result:
            return []
        facts = []
        for item in result.get("results", result.get("facts", [])):
            if isinstance(item, dict):
                text = item.get("text", item.get("fact", ""))
                score = item.get("relevance", item.get("score", item.get("similarity", 0.0)))
                if text:
                    f = Fact(text=text, source=f"shard:{self.name}", score=float(score))
                    facts.append(f)
        return facts

    def get(self, fact_id: str) -> Fact | None:
        return None  # Remote shards don't support ID lookup

    def count(self) -> int:
        info = self._get("/health")
        if info:
            return info.get("memory_facts", info.get("fact_count", info.get("facts", 0)))
        return 0

    def all_embeddings(self) -> tuple[list[str], np.ndarray]:
        raise NotImplementedError("Remote shards don't expose raw embeddings")

    def health(self) -> dict:
        info = self._get("/health")
        if info:
            return {"name": self.name, "endpoint": self.endpoint,
                    "count": self.count(), "status": "ok"}
        return {"name": self.name, "endpoint": self.endpoint, "status": "unreachable"}


# Need json import for RemoteShard
import json
