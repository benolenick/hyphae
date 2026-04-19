#!/usr/bin/env python3
"""Ingest OpenKeel profiles and agent history into Hyphae.

Parses:
  1. Profile YAMLs → domain knowledge about tools, safety, workflows
  2. Agent history story.txt files → project experiences, decisions, outcomes
"""
import os
import re
import sys
import logging
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ingest")

PROFILES_DIR = r"C:\Users\om\openkeel\profiles"
HISTORY_DIR = r"C:\Users\om\Documents\agent history"
DB_PATH = os.path.join(os.path.expanduser("~"), ".hyphae", "hyphae.db")


# ---------------------------------------------------------------------------
# Profile ingestion
# ---------------------------------------------------------------------------

def parse_yaml_simple(path: str) -> dict:
    """Load a YAML file."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        log.warning("PyYAML not installed, skipping YAML parsing")
        return {}


def extract_profile_facts(profile_path: str) -> list[tuple[str, dict]]:
    """Extract knowledge facts from a profile YAML.

    Returns list of (fact_text, tags_dict).
    """
    data = parse_yaml_simple(profile_path)
    if not data:
        return []

    name = data.get("name", os.path.basename(profile_path).replace(".yaml", ""))
    desc = data.get("description", "")
    facts = []
    tags = {"source": "profile", "profile": name}

    # Profile description
    if desc:
        facts.append((f"Profile '{name}': {desc}", {**tags, "type": "description"}))

    # Blocked patterns — encode what's dangerous
    blocked = data.get("blocked", {})
    if isinstance(blocked, dict):
        for pattern_list in blocked.values():
            if isinstance(pattern_list, list):
                for p in pattern_list:
                    if isinstance(p, str) and len(p) > 10:
                        # Extract readable description from regex
                        readable = _regex_to_description(p)
                        if readable:
                            facts.append((
                                f"In {name} context: '{readable}' is a blocked/dangerous operation",
                                {**tags, "type": "safety_rule", "tier": "blocked"},
                            ))

    # Safe patterns — encode what tools are available
    safe = data.get("safe", {})
    if isinstance(safe, dict):
        for category, pattern_list in safe.items():
            if isinstance(pattern_list, list):
                tools = _extract_tool_names(pattern_list)
                if tools:
                    facts.append((
                        f"In {name} context: safe tools include {', '.join(tools)}",
                        {**tags, "type": "tool_knowledge", "tier": "safe"},
                    ))

    # Scope — encode network/path boundaries
    scope = data.get("scope", {})
    if isinstance(scope, dict):
        allowed_ips = scope.get("allowed_ips", [])
        allowed_hosts = scope.get("allowed_hostnames", [])
        if allowed_ips:
            facts.append((
                f"In {name} context: allowed IP ranges are {', '.join(str(ip) for ip in allowed_ips)}",
                {**tags, "type": "scope"},
            ))
        if allowed_hosts:
            facts.append((
                f"In {name} context: allowed hostnames are {', '.join(str(h) for h in allowed_hosts)}",
                {**tags, "type": "scope"},
            ))

    # Activities — encode workflow knowledge
    activities = data.get("activities", [])
    if isinstance(activities, list):
        for act in activities:
            if isinstance(act, dict):
                act_name = act.get("name", "")
                act_desc = act.get("description", "")
                timebox = act.get("timebox_minutes", "")
                if act_name:
                    text = f"In {name}: activity '{act_name}'"
                    if act_desc:
                        text += f" — {act_desc}"
                    if timebox:
                        text += f" (timeboxed to {timebox} minutes)"
                    facts.append((text, {**tags, "type": "workflow"}))

    # Phases — encode workflow progression
    phases = data.get("phases", [])
    if isinstance(phases, list):
        phase_names = []
        for phase in phases:
            if isinstance(phase, dict):
                pname = phase.get("name", "")
                pdesc = phase.get("description", "")
                if pname:
                    phase_names.append(pname)
                    if pdesc:
                        facts.append((
                            f"In {name}: phase '{pname}' — {pdesc}",
                            {**tags, "type": "workflow_phase"},
                        ))
        if phase_names:
            facts.append((
                f"In {name}: workflow follows phases: {' → '.join(phase_names)}",
                {**tags, "type": "workflow"},
            ))

    # Memoria hooks — tool query mappings
    memoria_hooks = data.get("memoria_hooks", {})
    if isinstance(memoria_hooks, dict):
        tool_queries = memoria_hooks.get("tool_queries", {})
        if isinstance(tool_queries, dict) and tool_queries:
            for tool, query_template in tool_queries.items():
                facts.append((
                    f"When using {tool}: query memory for '{query_template}'",
                    {**tags, "type": "tool_knowledge"},
                ))

    return facts


def _regex_to_description(pattern: str) -> str:
    """Try to make a regex pattern human-readable."""
    # Extract literal words from regex
    words = re.findall(r'[a-zA-Z_-]{3,}', pattern)
    if not words:
        return ""
    # Filter out common regex artifacts
    words = [w for w in words if w not in ("tcp", "dev", "bin", "usr", "etc", "tmp")]
    if len(words) > 6:
        words = words[:6]
    return " ".join(words)


def _extract_tool_names(patterns: list) -> list[str]:
    """Extract tool/command names from safe patterns."""
    tools = set()
    for p in patterns:
        if not isinstance(p, str):
            continue
        # Look for tool names at start of patterns
        m = re.match(r'^(?:\(\?i\))?\^?\(?([a-zA-Z][a-zA-Z0-9_-]+)', p)
        if m:
            tool = m.group(1)
            if len(tool) >= 2 and tool not in ("http", "https", "localhost"):
                tools.add(tool)
        # Also extract from alternation groups
        for m in re.finditer(r'\b([a-z][a-z0-9_-]{2,})\b', p):
            word = m.group(1)
            if word in ("nmap", "sqlmap", "gobuster", "ffuf", "nikto", "hydra",
                        "john", "hashcat", "crackmapexec", "bloodhound", "impacket",
                        "msfconsole", "linpeas", "winpeas", "chisel", "ligolo",
                        "pytest", "docker", "ansible", "terraform", "kubectl",
                        "curl", "wget", "git", "pip", "npm", "python", "node",
                        "suricata", "wazuh", "crowdsec", "grafana", "prometheus",
                        "evil-winrm", "kerbrute", "sshuttle", "secretsdump",
                        "paramiko", "fastapi", "uvicorn", "nginx", "caddy"):
                tools.add(word)
    return sorted(tools)


# ---------------------------------------------------------------------------
# History ingestion
# ---------------------------------------------------------------------------

def extract_history_facts(story_path: str, project: str) -> list[tuple[str, dict]]:
    """Extract facts from a story.txt file.

    Parses entries separated by --- and extracts:
    - Actions taken
    - Decisions made (with reasoning)
    - Validation results
    - Next steps / open issues
    - Technical details (IPs, services, files)
    """
    try:
        with open(story_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return []

    if not content.strip():
        return []

    facts = []
    tags_base = {"source": "history", "project": project}

    # Split into entries by --- separator
    entries = re.split(r'\n-{3,}\n', content)

    for entry in entries:
        entry = entry.strip()
        if not entry or len(entry) < 30:
            continue

        # Extract timestamp
        ts_match = re.search(r'\[?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\]?', entry)
        timestamp = ts_match.group(1) if ts_match else ""

        # Extract context line (Profile, Target, etc.)
        context_match = re.search(r'(?:Profile|Target|Mission|Mode):\s*(.+?)$', entry, re.MULTILINE)
        context = context_match.group(1).strip() if context_match else ""

        # Extract request
        req_match = re.search(r'Request:\s*(.+?)$', entry, re.MULTILINE)
        request = req_match.group(1).strip() if req_match else ""

        tags = {**tags_base}
        if timestamp:
            tags["timestamp"] = timestamp

        # Extract sections
        sections = _extract_sections(entry)

        # Actions as facts
        actions = sections.get("actions", [])
        if actions:
            # Summarize as one fact per entry
            action_summary = "; ".join(a.strip("- ").strip() for a in actions[:5])
            if request:
                fact_text = f"[{project}] {request}: {action_summary}"
            else:
                fact_text = f"[{project}] Actions: {action_summary}"
            if len(fact_text) > 50:
                facts.append((fact_text[:500], {**tags, "type": "action"}))

        # Decisions as individual facts (these are high-value)
        decisions = sections.get("decisions", [])
        for d in decisions:
            d = d.strip("- ").strip()
            if len(d) > 20:
                facts.append((
                    f"[{project}] Decision: {d[:400]}",
                    {**tags, "type": "decision"},
                ))

        # Validation results
        validations = sections.get("validation", [])
        for v in validations:
            v = v.strip("- ").strip()
            if len(v) > 15:
                facts.append((
                    f"[{project}] Validated: {v[:300]}",
                    {**tags, "type": "validation"},
                ))

        # Next steps (useful for context)
        next_steps = sections.get("next steps", sections.get("next_steps", []))
        if next_steps:
            steps_text = "; ".join(s.strip("- ").strip() for s in next_steps[:3])
            if len(steps_text) > 20:
                facts.append((
                    f"[{project}] Open: {steps_text[:300]}",
                    {**tags, "type": "next_steps"},
                ))

        # Special sections (== HEADER == style)
        for header in ["ATTACK CHAIN", "PROGRESS", "CREDENTIALS", "INFRASTRUCTURE",
                        "ARCHITECTURE", "DEPLOYMENT", "KNOWN ISSUES"]:
            section = sections.get(header.lower(), [])
            if section:
                text = "; ".join(s.strip("- ").strip() for s in section[:5])
                if len(text) > 20:
                    facts.append((
                        f"[{project}] {header.title()}: {text[:400]}",
                        {**tags, "type": header.lower().replace(" ", "_")},
                    ))

    return facts


def _extract_sections(entry: str) -> dict[str, list[str]]:
    """Extract named sections from an entry."""
    sections: dict[str, list[str]] = {}
    current_section = ""
    current_lines: list[str] = []

    for line in entry.split("\n"):
        stripped = line.strip()

        # Check for == HEADER == style
        header_match = re.match(r'^==\s*(.+?)\s*==$', stripped)
        if header_match:
            if current_section and current_lines:
                sections[current_section] = current_lines
            current_section = header_match.group(1).lower()
            current_lines = []
            continue

        # Check for "Section:" style
        section_match = re.match(r'^(Actions|Decisions|Changed Files|Validation|Next Steps|Request|Open Issues):\s*$', stripped, re.IGNORECASE)
        if section_match:
            if current_section and current_lines:
                sections[current_section] = current_lines
            current_section = section_match.group(1).lower()
            current_lines = []
            continue

        # Accumulate lines
        if current_section and stripped.startswith("-"):
            current_lines.append(stripped)
        elif current_section and stripped:
            current_lines.append(stripped)

    if current_section and current_lines:
        sections[current_section] = current_lines

    return sections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from hyphae import Hyphae

    log.info(f"Initializing Hyphae at {DB_PATH}")
    h = Hyphae(db_path=DB_PATH, sim_threshold=0.45)

    initial_count = h.local_shard.count()
    log.info(f"Starting with {initial_count} existing facts")

    total_ingested = 0
    t0 = time.time()

    # --- Ingest profiles ---
    log.info("=== Ingesting profiles ===")
    profile_files = [f for f in os.listdir(PROFILES_DIR) if f.endswith(".yaml")]
    for pf in sorted(profile_files):
        path = os.path.join(PROFILES_DIR, pf)
        facts = extract_profile_facts(path)
        for text, tags in facts:
            fid, cid = h.remember(text, tags=tags, source=f"profile:{pf}")
        log.info(f"  {pf}: {len(facts)} facts")
        total_ingested += len(facts)

    # --- Ingest agent histories ---
    log.info("=== Ingesting agent histories ===")
    if os.path.isdir(HISTORY_DIR):
        projects = sorted(os.listdir(HISTORY_DIR))
        for project in projects:
            project_dir = os.path.join(HISTORY_DIR, project)
            if not os.path.isdir(project_dir):
                continue

            # Look for story.txt and any other .txt files
            for fname in os.listdir(project_dir):
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(project_dir, fname)
                facts = extract_history_facts(fpath, project)
                if facts:
                    for text, tags in facts:
                        fid, cid = h.remember(text, tags=tags, source=f"history:{project}/{fname}")
                    log.info(f"  {project}/{fname}: {len(facts)} facts")
                    total_ingested += len(facts)

    elapsed = time.time() - t0

    # --- Summary ---
    log.info("=== Ingestion complete ===")
    log.info(f"Total facts ingested: {total_ingested}")
    log.info(f"Total facts in DB: {h.local_shard.count()}")
    log.info(f"Elapsed: {elapsed:.1f}s")

    # Run merge check
    merged = h.cluster_engine.maybe_merge()
    if merged:
        log.info(f"Merged {len(merged)} cluster pairs")
        h.cluster_engine.save_to_shard(h.local_shard)

    # Show cluster status
    status = h.cluster_status()
    log.info(f"Clusters: {status['total_clusters']}")
    for c in status["clusters"][:15]:
        log.info(f"  #{c['id']}: {c['count']} facts")

    h.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
