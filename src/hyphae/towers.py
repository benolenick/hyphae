#!/usr/bin/env python3
"""Tower organic growth module for Hyphae."""
import sqlite3
import json
import time
import hashlib
import os
import logging

logger = logging.getLogger("hyphae.towers")

DB_PATH = "/home/om/.hyphae/hyphae.db"
SEED_THRESHOLD = 5          # facts in a cluster before auto-seeding a tower
SYNTHESIS_THRESHOLD = 10    # raw materials before triggering floor synthesis
TOWER_PROXIMITY = 0.75      # cosine similarity to consider a fact "near" a tower

def init_tower_tables():
    """Create tower tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS towers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            purpose TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            foundation_clusters TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS tower_floors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tower_id TEXT NOT NULL,
            floor_num INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY (tower_id) REFERENCES towers(id),
            UNIQUE(tower_id, floor_num)
        );
        CREATE TABLE IF NOT EXISTS tower_connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tower_id TEXT NOT NULL,
            connection_type TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            description TEXT,
            strength REAL DEFAULT 1.0,
            FOREIGN KEY (tower_id) REFERENCES towers(id)
        );
        CREATE TABLE IF NOT EXISTS tower_views (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tower_id TEXT NOT NULL,
            direction TEXT NOT NULL,
            description TEXT NOT NULL,
            related_clusters TEXT DEFAULT '[]',
            FOREIGN KEY (tower_id) REFERENCES towers(id)
        );
        CREATE TABLE IF NOT EXISTS tower_materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tower_id TEXT NOT NULL,
            fact_id TEXT NOT NULL,
            fact_text TEXT NOT NULL,
            added_at REAL NOT NULL,
            synthesized INTEGER DEFAULT 0,
            FOREIGN KEY (tower_id) REFERENCES towers(id),
            UNIQUE(tower_id, fact_id)
        );
    """)
    conn.commit()
    conn.close()


def get_tower_by_name(name):
    """Find a tower by name (case insensitive)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM towers WHERE LOWER(name) = LOWER(?)", (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_towers():
    """List all active towers."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM towers WHERE status = 'active'").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_tower_floors(tower_id):
    """Get all floors for a tower, ordered."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM tower_floors WHERE tower_id = ? ORDER BY floor_num", (tower_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_tower_materials(tower_id, unsynthesized_only=True):
    """Get raw materials for a tower."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if unsynthesized_only:
        rows = conn.execute(
            "SELECT * FROM tower_materials WHERE tower_id = ? AND synthesized = 0 ORDER BY added_at",
            (tower_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM tower_materials WHERE tower_id = ? ORDER BY added_at", (tower_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def seed_tower(name, purpose, source_facts=None):
    """Plant a tower seed. Auto-generates Floor 0 from the purpose."""
    tower_id = hashlib.sha256(name.lower().encode()).hexdigest()[:16]
    now = time.time()
    conn = sqlite3.connect(DB_PATH)

    # Check if already exists
    existing = conn.execute("SELECT id FROM towers WHERE id = ?", (tower_id,)).fetchone()
    if existing:
        conn.close()
        return tower_id

    conn.execute(
        "INSERT INTO towers (id, name, purpose, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (tower_id, name, purpose, "seed", now, now)
    )

    # Auto Floor 0 from purpose
    conn.execute(
        "INSERT INTO tower_floors (tower_id, floor_num, title, content, updated_at) VALUES (?,?,?,?,?)",
        (tower_id, 0, f"What is {name}?", purpose, now)
    )

    # Add source facts as raw materials
    if source_facts:
        for fact in source_facts:
            fact_id = hashlib.sha256(fact.encode()).hexdigest()[:16]
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO tower_materials (tower_id, fact_id, fact_text, added_at) VALUES (?,?,?,?)",
                    (tower_id, fact_id, fact[:500], now)
                )
            except:
                pass

    conn.commit()
    conn.close()
    logger.info(f"Seeded tower '{name}' ({tower_id})")
    return tower_id


def add_material(tower_id, fact_text):
    """Add a raw material fact to a tower's pile."""
    fact_id = hashlib.sha256(fact_text.encode()).hexdigest()[:16]
    now = time.time()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO tower_materials (tower_id, fact_id, fact_text, added_at) VALUES (?,?,?,?)",
            (tower_id, fact_id, fact_text[:500], now)
        )
        conn.commit()
    except:
        pass
    conn.close()


def check_synthesis_needed(tower_id):
    """Check if a tower has enough raw materials to synthesize new floors."""
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute(
        "SELECT COUNT(*) FROM tower_materials WHERE tower_id = ? AND synthesized = 0",
        (tower_id,)
    ).fetchone()[0]
    conn.close()
    return count >= SYNTHESIS_THRESHOLD


def synthesize_floors(tower_id):
    """Take raw materials and build/update floors using LLM.

    Groups materials by theme, generates floor content.
    This is called when enough materials accumulate.
    """
    materials = get_tower_materials(tower_id, unsynthesized_only=True)
    if not materials:
        return

    tower = None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    tower_row = conn.execute("SELECT * FROM towers WHERE id = ?", (tower_id,)).fetchone()
    if tower_row:
        tower = dict(tower_row)
    conn.close()

    if not tower:
        return

    # Get existing floors
    existing_floors = get_tower_floors(tower_id)
    max_floor = max([f["floor_num"] for f in existing_floors]) if existing_floors else 0

    # Group materials into themes using simple keyword clustering
    themes = {
        "problems": [],
        "successes": [],
        "architecture": [],
        "connections": [],
        "general": [],
    }

    for mat in materials:
        text = mat["fact_text"].lower()
        if any(w in text for w in ["error", "fail", "broken", "bug", "issue", "problem", "crash", "dead"]):
            themes["problems"].append(mat["fact_text"])
        elif any(w in text for w in ["success", "pwned", "rooted", "worked", "flag", "owned"]):
            themes["successes"].append(mat["fact_text"])
        elif any(w in text for w in ["architecture", "component", "endpoint", "api", "server", "client", "hook"]):
            themes["architecture"].append(mat["fact_text"])
        elif any(w in text for w in ["connect", "depend", "feed", "query", "upstream", "downstream"]):
            themes["connections"].append(mat["fact_text"])
        else:
            themes["general"].append(mat["fact_text"])

    # Build new floors from themes that have content
    now = time.time()
    conn = sqlite3.connect(DB_PATH)
    new_floors = 0

    floor_templates = {
        "problems": ("Known Issues", "Problems and issues discovered: "),
        "successes": ("Results & Wins", "Concrete results and successes: "),
        "architecture": ("Architecture Notes", "Technical architecture details: "),
        "connections": ("Connections & Dependencies", "Relationships with other systems: "),
        "general": ("Additional Knowledge", "Other relevant information: "),
    }

    for theme, facts in themes.items():
        if not facts:
            continue
        title, prefix = floor_templates[theme]

        # Check if a floor with this title already exists
        existing = conn.execute(
            "SELECT id, content FROM tower_floors WHERE tower_id = ? AND title = ?",
            (tower_id, title)
        ).fetchone()

        content = prefix + " | ".join(f[:200] for f in facts[:10])
        content = content[:2000]

        if existing:
            # Append to existing floor
            old_content = existing[1]
            new_content = old_content + " | " + " | ".join(f[:200] for f in facts[:10])
            conn.execute(
                "UPDATE tower_floors SET content = ?, updated_at = ? WHERE id = ?",
                (new_content[:2000], now, existing[0])
            )
        else:
            max_floor += 1
            conn.execute(
                "INSERT INTO tower_floors (tower_id, floor_num, title, content, updated_at) VALUES (?,?,?,?,?)",
                (tower_id, max_floor, title, content, now)
            )
            new_floors += 1

    # Mark materials as synthesized
    for mat in materials:
        conn.execute(
            "UPDATE tower_materials SET synthesized = 1 WHERE id = ?", (mat["id"],)
        )

    # Update tower status from seed to active if we have 3+ floors
    total_floors = conn.execute(
        "SELECT COUNT(*) FROM tower_floors WHERE tower_id = ?", (tower_id,)
    ).fetchone()[0]
    if total_floors >= 3:
        conn.execute("UPDATE towers SET status = 'active', updated_at = ? WHERE id = ?", (now, tower_id))

    conn.commit()
    conn.close()
    logger.info(f"Synthesized {new_floors} new floors for tower {tower['name']} from {len(materials)} materials")


def auto_seed_from_box_completion(box_name, attack_chain, flags, credentials=None):
    """Auto-build a Target Tower when a box is completed."""
    purpose = f"HTB box {box_name}. Attack chain: {attack_chain[:200]}"
    tower_id = seed_tower(
        name=f"HTB-{box_name}",
        purpose=purpose,
    )

    now = time.time()
    conn = sqlite3.connect(DB_PATH)

    # Floor 1: Attack chain
    conn.execute(
        "INSERT OR REPLACE INTO tower_floors (tower_id, floor_num, title, content, updated_at) VALUES (?,?,?,?,?)",
        (tower_id, 1, "Attack Chain", attack_chain[:2000], now)
    )

    # Floor 2: Flags
    conn.execute(
        "INSERT OR REPLACE INTO tower_floors (tower_id, floor_num, title, content, updated_at) VALUES (?,?,?,?,?)",
        (tower_id, 2, "Flags", flags, now)
    )

    # Floor 3: Credentials if any
    if credentials:
        conn.execute(
            "INSERT OR REPLACE INTO tower_floors (tower_id, floor_num, title, content, updated_at) VALUES (?,?,?,?,?)",
            (tower_id, 3, "Credentials Found", credentials[:2000], now)
        )

    conn.execute("UPDATE towers SET status = 'active', updated_at = ? WHERE id = ?", (now, tower_id))
    conn.commit()
    conn.close()
    return tower_id


def auto_seed_failure_tower(approach, failures, context=""):
    """Auto-build a Failure Tower when repeated failures detected."""
    name = f"FAIL-{approach[:30]}"
    purpose = f"Failed approach: {approach}. {len(failures)} attempts failed."
    tower_id = seed_tower(name=name, purpose=purpose, source_facts=failures)
    return tower_id


def walk_tower(tower_name):
    """Walk through a tower — returns all floors, connections, views."""
    tower = get_tower_by_name(tower_name)
    if not tower:
        return None

    floors = get_tower_floors(tower["id"])

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    connections = [dict(r) for r in conn.execute(
        "SELECT * FROM tower_connections WHERE tower_id = ?", (tower["id"],)
    ).fetchall()]
    views = [dict(r) for r in conn.execute(
        "SELECT * FROM tower_views WHERE tower_id = ?", (tower["id"],)
    ).fetchall()]
    materials_count = conn.execute(
        "SELECT COUNT(*) FROM tower_materials WHERE tower_id = ? AND synthesized = 0",
        (tower["id"],)
    ).fetchone()[0]
    conn.close()

    return {
        "tower": tower,
        "floors": floors,
        "connections": connections,
        "views": views,
        "pending_materials": materials_count,
    }


# Initialize on import
init_tower_tables()

if __name__ == "__main__":
    # Test
    init_tower_tables()
    print("Tables initialized")

    # Seed a test tower
    tid = seed_tower("Test Tower", "A test tower for development")
    print(f"Seeded: {tid}")

    # Add materials
    for i in range(12):
        add_material(tid, f"Test fact number {i} about error handling and bug fixes")

    # Check synthesis
    print(f"Needs synthesis: {check_synthesis_needed(tid)}")

    # Synthesize
    synthesize_floors(tid)

    # Walk
    result = walk_tower("Test Tower")
    print(f"Walk: {result['tower']['name']}, {len(result['floors'])} floors, {result['pending_materials']} pending materials")
    for f in result["floors"]:
        print(f"  Floor {f['floor_num']}: {f['title']}")

    # Clean up test
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM towers WHERE name = 'Test Tower'")
    conn.execute("DELETE FROM tower_floors WHERE tower_id = ?", (tid,))
    conn.execute("DELETE FROM tower_materials WHERE tower_id = ?", (tid,))
    conn.commit()
    conn.close()
    print("Test cleaned up")
