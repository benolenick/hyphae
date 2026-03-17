"""Hyphae MCP server — exposes memory as native Claude Code tools."""
from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("hyphae.mcp")

HYPHAE_ENDPOINT = "http://127.0.0.1:8100"

mcp = FastMCP(
    "Hyphae Memory",
    instructions=(
        "Hyphae is your long-term memory. Use recall_memory when past context would "
        "help — errors you've seen before, projects you've worked on, decisions made, "
        "techniques tried. Use remember_fact to save important discoveries, decisions, "
        "or lessons learned. Memory is scoped to the current project by default."
    ),
)


def _post(path: str, payload: dict) -> dict:
    """POST to Hyphae HTTP API."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{HYPHAE_ENDPOINT}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(path: str) -> dict:
    """GET from Hyphae HTTP API."""
    req = urllib.request.Request(f"{HYPHAE_ENDPOINT}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


@mcp.tool()
def recall_memory(query: str, top_k: int = 8) -> str:
    """Search long-term memory for relevant past knowledge.

    Use this when:
    - You encounter an error or problem that might have been solved before
    - You're starting work on a project and want to know what's been done
    - The user references past work, decisions, or conversations
    - You need context about infrastructure, credentials, or architecture
    - You're stuck and want to check if there's relevant history

    Results are scoped to the current project by default.

    Args:
        query: What to search for — be specific (e.g. "SSH connection timeout to jagg" not "SSH")
        top_k: Number of results to return (default 8)
    """
    try:
        result = _post("/recall", {"query": query, "top_k": top_k})
        facts = result.get("results", [])
        if not facts:
            return "No relevant memories found."
        lines = []
        for f in facts:
            score = f.get("score", 0)
            text = f.get("text", "")
            tags = f.get("tags", {})
            project = tags.get("project", "")
            source = tags.get("source", "") or f.get("source", "")
            prefix = f"[{score:.2f}]"
            if project:
                prefix += f" ({project})"
            lines.append(f"{prefix} {text}")
        return "\n".join(lines)
    except Exception as e:
        return f"Memory recall failed: {e}"


@mcp.tool()
def recall_all_projects(query: str, top_k: int = 10) -> str:
    """Search memory across ALL projects (unscoped).

    Use this when:
    - The user asks about work across multiple projects
    - You need to find something but don't know which project it's in
    - The user asks "what projects have I worked on" or similar cross-project questions

    Args:
        query: What to search for
        top_k: Number of results to return
    """
    try:
        result = _post("/recall", {"query": query, "top_k": top_k, "scope": {}})
        facts = result.get("results", [])
        if not facts:
            return "No relevant memories found across any project."
        lines = []
        for f in facts:
            score = f.get("score", 0)
            text = f.get("text", "")
            tags = f.get("tags", {})
            project = tags.get("project", "unknown")
            lines.append(f"[{score:.2f}] ({project}) {text}")
        return "\n".join(lines)
    except Exception as e:
        return f"Memory recall failed: {e}"


@mcp.tool()
def remember_fact(text: str, source: str = "agent") -> str:
    """Save an important fact, decision, or lesson to long-term memory.

    Use this to remember:
    - Solutions to problems (so you don't re-solve them)
    - Architecture decisions and why they were made
    - Credentials, endpoints, or infrastructure details
    - Lessons learned from failures
    - User preferences or corrections

    Facts are auto-tagged with the current project.

    Args:
        text: The fact to remember — be specific and self-contained
        source: Who's saving this (default "agent")
    """
    try:
        result = _post("/remember", {"text": text, "source": source})
        fact_id = result.get("id", "unknown")
        cluster = result.get("cluster_id", "?")
        return f"Remembered (id: {fact_id}, cluster: {cluster})"
    except Exception as e:
        return f"Failed to save memory: {e}"


@mcp.tool()
def memory_status() -> str:
    """Check Hyphae memory health and stats.

    Use this to verify memory is working or to see how many facts exist.
    """
    try:
        health = _get("/health")
        facts = health.get("facts", "?")
        clusters = health.get("clusters", "?")
        status = health.get("status", "unknown")
        parts = [f"Status: {status}", f"Facts: {facts}", f"Clusters: {clusters}"]
        try:
            stats = _get("/stats")
            manifolds = stats.get("manifold_coverage", {})
            built = manifolds.get("clusters_with_manifold", 0)
            total = manifolds.get("total_clusters", 0)
            parts.append(f"Manifolds: {built}/{total}")
        except Exception:
            pass
        return " | ".join(parts)
    except Exception as e:
        return f"Memory offline: {e}"


if __name__ == "__main__":
    mcp.run()
