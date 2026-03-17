"""River — stepping stone lifecycle with persistence.

Tracks structured goals from objective to completion.
Each stone has success/failure markers and a diagnostic cycle on failure.
State persists to SQLite so sessions can be resumed.
"""
from __future__ import annotations

import json
import sqlite3
import time
import logging
from typing import Optional

from .types import Stone, River

logger = logging.getLogger("hyphae.river")


class RiverDB:
    """SQLite persistence for rivers, stones, and attempt logs."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS rivers (
                id TEXT PRIMARY KEY,
                objective TEXT NOT NULL,
                current_stone_idx INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                created_at REAL,
                updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS stones (
                id TEXT PRIMARY KEY,
                river_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                goal TEXT NOT NULL,
                success_marker TEXT DEFAULT '',
                failure_marker TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                attempts INTEGER DEFAULT 0,
                max_attempts INTEGER DEFAULT 3,
                created_at REAL,
                updated_at REAL,
                FOREIGN KEY (river_id) REFERENCES rivers(id)
            );

            CREATE TABLE IF NOT EXISTS attempt_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stone_id TEXT NOT NULL,
                attempt_num INTEGER,
                context_json TEXT,
                output TEXT,
                result TEXT,
                timestamp REAL,
                FOREIGN KEY (stone_id) REFERENCES stones(id)
            );
        """)
        self.conn.commit()

    def create_river(self, river: River) -> str:
        now = time.time()
        self.conn.execute(
            "INSERT INTO rivers (id, objective, current_stone_idx, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (river.id, river.objective, 0, "active", now, now),
        )
        for i, stone in enumerate(river.stones):
            self.conn.execute(
                "INSERT INTO stones (id, river_id, idx, goal, success_marker, failure_marker, "
                "status, attempts, max_attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (stone.id, river.id, i, stone.goal, stone.success_marker,
                 stone.failure_marker, stone.status, 0, stone.max_attempts, now, now),
            )
        self.conn.commit()
        return river.id

    def get_river(self, river_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM rivers WHERE id = ?", (river_id,)).fetchone()
        if not row:
            return None
        stones = self.conn.execute(
            "SELECT * FROM stones WHERE river_id = ? ORDER BY idx", (river_id,)
        ).fetchall()
        return {
            "id": row["id"],
            "objective": row["objective"],
            "current_stone_idx": row["current_stone_idx"],
            "status": row["status"],
            "stones": [dict(s) for s in stones],
        }

    def get_active_river(self) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT id FROM rivers WHERE status = 'active' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row:
            return self.get_river(row["id"])
        return None

    def update_stone_status(self, stone_id: str, status: str):
        self.conn.execute(
            "UPDATE stones SET status = ?, updated_at = ? WHERE id = ?",
            (status, time.time(), stone_id),
        )
        self.conn.commit()

    def advance_river(self, river_id: str):
        river = self.get_river(river_id)
        if not river:
            return
        new_idx = river["current_stone_idx"] + 1
        if new_idx >= len(river["stones"]):
            self.conn.execute(
                "UPDATE rivers SET status = 'completed', current_stone_idx = ?, updated_at = ? WHERE id = ?",
                (new_idx, time.time(), river_id),
            )
        else:
            self.conn.execute(
                "UPDATE rivers SET current_stone_idx = ?, updated_at = ? WHERE id = ?",
                (new_idx, time.time(), river_id),
            )
        self.conn.commit()

    def log_attempt(self, stone_id: str, attempt_num: int, context: dict,
                    output: str, result: str):
        self.conn.execute(
            "INSERT INTO attempt_log (stone_id, attempt_num, context_json, output, result, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (stone_id, attempt_num, json.dumps(context), output[:5000], result, time.time()),
        )
        self.conn.execute(
            "UPDATE stones SET attempts = ?, updated_at = ? WHERE id = ?",
            (attempt_num, time.time(), stone_id),
        )
        self.conn.commit()

    def get_stone_history(self, stone_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM attempt_log WHERE stone_id = ? ORDER BY attempt_num",
            (stone_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def insert_stone(self, river_id: str, stone: Stone, after_idx: int):
        """Insert a new stone after the given index."""
        now = time.time()
        # Shift subsequent stones
        self.conn.execute(
            "UPDATE stones SET idx = idx + 1 WHERE river_id = ? AND idx > ?",
            (river_id, after_idx),
        )
        self.conn.execute(
            "INSERT INTO stones (id, river_id, idx, goal, success_marker, failure_marker, "
            "status, attempts, max_attempts, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (stone.id, river_id, after_idx + 1, stone.goal, stone.success_marker,
             stone.failure_marker, "pending", 0, stone.max_attempts, now, now),
        )
        self.conn.commit()


class RiverManager:
    """Orchestrates stepping stone lifecycle."""

    def __init__(self, conn: sqlite3.Connection):
        self.db = RiverDB(conn)

    def create(self, objective: str, stones: list[dict]) -> dict:
        """Create a river. stones: list of {goal, success_marker, failure_marker}."""
        river = River(objective=objective)
        for s in stones:
            river.stones.append(Stone(
                goal=s.get("goal", ""),
                success_marker=s.get("success_marker", ""),
                failure_marker=s.get("failure_marker", ""),
                max_attempts=s.get("max_attempts", 3),
            ))
        if river.stones:
            river.stones[0].status = "attempting"
        self.db.create_river(river)
        return self.status(river.id)

    def status(self, river_id: str = "") -> dict:
        if river_id:
            river = self.db.get_river(river_id)
        else:
            river = self.db.get_active_river()
        if not river:
            return {"error": "No active river"}

        idx = river["current_stone_idx"]
        stones_display = []
        for s in river["stones"]:
            icon = {"pending": "○", "attempting": "◐", "succeeded": "●",
                    "failed": "✗", "blocked": "⊘"}.get(s["status"], "?")
            marker = "→" if s["idx"] == idx else " "
            stones_display.append(f"{marker} {icon} [{s['status']}] {s['goal']}")

        current = river["stones"][idx]["goal"] if idx < len(river["stones"]) else "COMPLETE"
        return {
            "river_id": river["id"],
            "objective": river["objective"],
            "status": river["status"],
            "current_stone": current,
            "progress": f"{idx}/{len(river['stones'])}",
            "stones": stones_display,
        }

    def attempt(self, output: str, context: dict = None, river_id: str = "") -> dict:
        """Record a failed attempt. Returns diagnostic info."""
        context = context or {}
        if not river_id:
            river = self.db.get_active_river()
            river_id = river["id"] if river else ""
        if not river_id:
            return {"error": "No active river"}

        river = self.db.get_river(river_id)
        idx = river["current_stone_idx"]
        if idx >= len(river["stones"]):
            return {"error": "River completed"}

        stone = river["stones"][idx]
        stone_id = stone["id"]
        attempt_num = stone["attempts"] + 1

        self.db.update_stone_status(stone_id, "attempting")
        self.db.log_attempt(stone_id, attempt_num, context, output, "failure")

        result = {
            "stone": stone["goal"],
            "attempt": attempt_num,
            "max_attempts": stone["max_attempts"],
        }

        if attempt_num >= stone["max_attempts"]:
            self.db.update_stone_status(stone_id, "blocked")
            result["blocked"] = True
            result["message"] = (
                f"Stone blocked after {attempt_num} attempts. "
                f"Consider rethinking the approach."
            )
        else:
            result["remaining"] = stone["max_attempts"] - attempt_num

        return result

    def success(self, river_id: str = "") -> dict:
        """Mark current stone as succeeded and advance."""
        if not river_id:
            river = self.db.get_active_river()
            river_id = river["id"] if river else ""
        if not river_id:
            return {"error": "No active river"}

        river = self.db.get_river(river_id)
        idx = river["current_stone_idx"]
        if idx < len(river["stones"]):
            self.db.update_stone_status(river["stones"][idx]["id"], "succeeded")
            self.db.advance_river(river_id)
            # Mark next stone as attempting
            if idx + 1 < len(river["stones"]):
                self.db.update_stone_status(river["stones"][idx + 1]["id"], "attempting")

        return self.status(river_id)

    def insert(self, goal: str, success_marker: str, failure_marker: str = "",
               river_id: str = "") -> dict:
        """Insert a new stone after the current one."""
        if not river_id:
            river = self.db.get_active_river()
            river_id = river["id"] if river else ""
        if not river_id:
            return {"error": "No active river"}

        river = self.db.get_river(river_id)
        stone = Stone(goal=goal, success_marker=success_marker,
                      failure_marker=failure_marker)
        self.db.insert_stone(river_id, stone, river["current_stone_idx"])
        return {"inserted": stone.id, "goal": goal}

    def history(self, stone_id: str = "", river_id: str = "") -> list[dict]:
        """Get attempt history for a stone (defaults to current)."""
        if not stone_id:
            if not river_id:
                river = self.db.get_active_river()
                river_id = river["id"] if river else ""
            if river_id:
                river = self.db.get_river(river_id)
                idx = river["current_stone_idx"]
                if idx < len(river["stones"]):
                    stone_id = river["stones"][idx]["id"]
        if not stone_id:
            return []
        return self.db.get_stone_history(stone_id)
