"""Hyphae memory curator — Haiku-powered curation of project memories.

Two things it maintains:
  1. Project thesis: a living 2-3 sentence description of what the project is
     trying to accomplish. Survives context compression. Grows with progress.

  2. Methods: reusable HOW-TO procedures (e.g. "how to navigate Chrome with
     Playwright"). Detects conflicts between old and new methods and flags the
     old one for pruning so Claude doesn't re-learn stale approaches.

Triggered automatically at session end (Stop hook), with a 6h cooldown per
project to avoid hammering Haiku on short sessions.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error

logger = logging.getLogger("hyphae.curate")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
COOLDOWN_HOURS = 6
MAX_FACTS_PER_TYPE = 20   # cap per type to keep prompt lean
MAX_PROMPT_CHARS = 12000  # stay well under Haiku's context


# ---------------------------------------------------------------------------
# Anthropic API call
# ---------------------------------------------------------------------------

def _call_haiku(prompt: str, api_key: str) -> str | None:
    """Call Haiku via raw HTTP. Returns assistant text or None on failure."""
    payload = json.dumps({
        "model": HAIKU_MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.warning(f"Haiku API error {e.code}: {body[:300]}")
        return None
    except Exception as e:
        logger.warning(f"Haiku call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Fetch project facts from shard
# ---------------------------------------------------------------------------

def _fetch_project_facts(shard, project: str) -> dict[str, list[str]]:
    """Fetch all facts for a project, grouped by type tag."""
    # Fetch without since= filter so we get the full picture
    all_facts = shard.get_recent_facts_for_project(project, since=0, limit=500)

    groups: dict[str, list[str]] = {}
    for f in all_facts:
        fact_type = f.tags.get("type", "general")
        text = f.text.strip()
        if len(text) > 30:
            groups.setdefault(fact_type, []).append(text)

    # Trim each group
    for k in groups:
        groups[k] = groups[k][:MAX_FACTS_PER_TYPE]

    return groups


# ---------------------------------------------------------------------------
# Thesis storage helpers
# ---------------------------------------------------------------------------

def _get_existing_thesis(shard, project: str) -> str | None:
    """Return the most recent thesis fact text for this project, if any."""
    try:
        rows = shard.conn.execute(
            "SELECT text, created_at FROM facts "
            "WHERE tags_json LIKE ? AND tags_json LIKE ? "
            "ORDER BY created_at DESC LIMIT 1",
            (f'%"project":"{project}"%', '%"type":"thesis"%'),
        ).fetchall()
        if rows:
            return rows[0]["text"]
    except Exception:
        pass
    return None


def _get_last_curate_time(shard, project: str) -> float:
    """Return unix timestamp of last thesis fact, or 0."""
    try:
        rows = shard.conn.execute(
            "SELECT created_at FROM facts "
            "WHERE tags_json LIKE ? AND tags_json LIKE ? "
            "ORDER BY created_at DESC LIMIT 1",
            (f'%"project":"{project}"%', '%"type":"thesis"%'),
        ).fetchall()
        if rows:
            return rows[0]["created_at"]
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Build Haiku prompt
# ---------------------------------------------------------------------------

def _build_prompt(project: str, groups: dict[str, list[str]], existing_thesis: str | None) -> str:
    sections = []

    if existing_thesis:
        sections.append(f"CURRENT THESIS (from last session):\n{existing_thesis}")

    # Priority types to surface
    type_labels = {
        "thesis": "PRIOR THESIS / DIRECTION",
        "method": "METHODS / HOW-TO PROCEDURES",
        "workflow": "WORKFLOWS",
        "decision": "DECISIONS",
        "action": "RECENT ACTIONS",
        "validation": "VALIDATIONS",
        "conversation": "CONVERSATION EXTRACTS",
        "general": "OTHER FACTS",
    }

    chars_used = sum(len(s) for s in sections)
    for type_key in ["thesis", "method", "workflow", "decision", "action", "validation",
                     "conversation", "general"]:
        facts = groups.get(type_key, [])
        if not facts:
            continue
        label = type_labels.get(type_key, type_key.upper())
        block = f"\n{label}:\n" + "\n".join(f"- {f[:200]}" for f in facts)
        if chars_used + len(block) > MAX_PROMPT_CHARS:
            break
        sections.append(block)
        chars_used += len(block)

    facts_dump = "\n".join(sections)

    prompt = f"""You are curating long-term memory for a software project called "{project}".

Below are all the stored memories for this project, grouped by type.

{facts_dump}

---

Your job: return a JSON object with exactly these three keys:

1. "thesis" — A 2-4 sentence summary of what this project IS and what it is TRYING TO DO.
   Write it as if briefing a new engineer who just joined. Include the core goal, the
   current approach/stack, and any important constraints or recent direction changes.
   This will survive context compression and be injected at every session start.

2. "method_conflicts" — A list of objects identifying method conflicts. A conflict is when
   two stored memories describe HOW to do the same thing but in incompatible ways
   (old approach vs new approach). For each conflict:
   {{
     "topic": "brief label",
     "keep": "the text of the NEWER/CORRECT method",
     "prune": "the text of the OLDER/STALE method",
     "reason": "one-sentence explanation of why the old one is wrong"
   }}
   If there are no conflicts, return an empty list.

3. "stale_facts" — A list of fact texts that are clearly outdated, superseded, or no
   longer relevant. Be conservative — only flag facts that are definitively wrong or
   replaced by something in the memory. If unsure, don't flag it.

Respond with ONLY the JSON object, no preamble or markdown fencing.
"""
    return prompt


# ---------------------------------------------------------------------------
# Parse Haiku response
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> dict | None:
    """Parse Haiku's JSON response. Strips markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON substring
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    logger.warning(f"Could not parse Haiku response: {text[:200]}")
    return None


# ---------------------------------------------------------------------------
# Main curate function
# ---------------------------------------------------------------------------

def curate(
    project: str,
    shard,
    hyphae_instance,
    api_key: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Run curation for a project. Returns a report dict.

    Args:
        project: Project name to curate
        shard: LocalShard instance
        hyphae_instance: Hyphae instance (for remember/pin calls)
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
        dry_run: If True, return recommendations without storing anything
        force: If True, skip cooldown check
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set", "project": project}

    # Cooldown check
    if not force:
        last_curate = _get_last_curate_time(shard, project)
        age_hours = (time.time() - last_curate) / 3600
        if last_curate > 0 and age_hours < COOLDOWN_HOURS:
            return {
                "project": project,
                "skipped": True,
                "reason": f"Curated {age_hours:.1f}h ago (cooldown: {COOLDOWN_HOURS}h)",
            }

    # Fetch project facts
    groups = _fetch_project_facts(shard, project)
    if not any(groups.values()):
        return {"project": project, "skipped": True, "reason": "No facts found for project"}

    existing_thesis = _get_existing_thesis(shard, project)

    # Build and send prompt
    prompt = _build_prompt(project, groups, existing_thesis)
    logger.info(f"Curating project '{project}' ({sum(len(v) for v in groups.values())} facts)")

    response_text = _call_haiku(prompt, api_key)
    if not response_text:
        return {"project": project, "error": "Haiku call failed"}

    parsed = _parse_response(response_text)
    if not parsed:
        return {"project": project, "error": "Could not parse Haiku response", "raw": response_text[:500]}

    thesis = parsed.get("thesis", "").strip()
    conflicts = parsed.get("method_conflicts", [])
    stale = parsed.get("stale_facts", [])

    report = {
        "project": project,
        "dry_run": dry_run,
        "thesis": thesis,
        "thesis_changed": thesis != (existing_thesis or ""),
        "method_conflicts": conflicts,
        "stale_facts": stale,
        "facts_scanned": sum(len(v) for v in groups.values()),
    }

    if dry_run:
        return report

    # Store updated thesis as a pinned fact
    if thesis and thesis != existing_thesis:
        thesis_text = f"[PROJECT THESIS — {project}] {thesis}"
        fact_id, cluster_id = hyphae_instance.remember(
            thesis_text,
            tags={"type": "thesis", "project": project},
            source="curate",
        )
        # Pin it so it doesn't decay
        try:
            shard.conn.execute(
                "INSERT OR IGNORE INTO pinned_facts (fact_id) VALUES (?)",
                (fact_id,),
            )
            shard.conn.commit()
        except Exception:
            pass  # pinned_facts table may not exist yet
        report["thesis_fact_id"] = fact_id
        logger.info(f"Stored updated thesis for '{project}' (id: {fact_id})")

    # Store method conflicts as facts (so they show up in future recalls)
    conflict_facts = []
    for conflict in conflicts:
        keep = conflict.get("keep", "")
        prune = conflict.get("prune", "")
        reason = conflict.get("reason", "")
        topic = conflict.get("topic", "method")
        if keep and prune:
            note_text = (
                f"[METHOD UPDATE — {project}] {topic}: prefer '{keep[:150]}' "
                f"over '{prune[:150]}'. Reason: {reason}"
            )
            fact_id, _ = hyphae_instance.remember(
                note_text,
                tags={"type": "method_update", "project": project},
                source="curate",
            )
            conflict_facts.append(fact_id)
    report["conflict_facts_stored"] = len(conflict_facts)

    logger.info(
        f"Curation complete for '{project}': thesis={'updated' if thesis else 'none'}, "
        f"conflicts={len(conflicts)}, stale_suggestions={len(stale)}"
    )
    return report
