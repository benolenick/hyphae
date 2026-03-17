# Hyphae

Self-organizing memory for AI agents. Add facts, query gaps, the network grows itself.

Hyphae is a long-term memory server that stores knowledge as facts, clusters them semantically, builds manifold geometry over the clusters, and retrieves along geodesic paths rather than flat cosine similarity. It's designed to be the memory layer for AI agents that need to remember decisions, discoveries, and context across sessions.

## Features

- **Remember/Recall API** - Store facts with tags and source attribution, recall by semantic similarity
- **Auto-clustering** - Facts are automatically routed to semantic clusters with adaptive merge/split
- **Manifold re-ranking** - Clusters with 30+ facts get diffusion map manifolds built in the background; recall results are re-ranked by manifold distance for sharper relevance
- **Session scoping** - Scope recalls to a project so cross-project facts don't bleed
- **Gap detection** - Analyze observations against an objective to find knowledge gaps and retrieve facts to fill them
- **Briefing system** - Auto-distills recent facts into compressed session summaries
- **Ivory Towers** - Structured knowledge landmarks that organize facts into navigable multi-floor buildings
- **MCP server** - Claude Code integration via Model Context Protocol
- **Rivers & Stones** - Goal tracking with stepping-stone progression

## Quick Start

```bash
pip install hyphae-memory[all]
hyphae
```

This starts the server on `http://127.0.0.1:8100`. Facts are stored in `~/.hyphae/hyphae.db`.

```bash
# Custom port
hyphae --port 9000

# Bind to all interfaces (for remote access)
hyphae --host 0.0.0.0 --port 8100
```

## CLI Agent Bootstrap

If you're an AI agent and someone just pointed you at a Hyphae server, here's how to use it.

### 1. Check if Hyphae is running

```bash
curl -s http://127.0.0.1:8100/health
```

If you get a JSON response with `"status": "ok"`, you're connected.

### 2. Set your session scope

Scope your session so your facts don't mix with other projects:

```bash
curl -s -X POST http://127.0.0.1:8100/session/set \
  -H "Content-Type: application/json" \
  -d '{"scope": {"project": "my-project-name"}}'
```

### 3. Check what's already known

Before starting work, recall relevant context:

```bash
curl -s -X POST http://127.0.0.1:8100/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "what has been done on this project", "top_k": 10}'
```

### 4. Save important things as you work

When you make a decision, discover something, or solve a problem, save it:

```bash
curl -s -X POST http://127.0.0.1:8100/remember \
  -H "Content-Type: application/json" \
  -d '{"text": "The database migration requires running migrate.py before starting the app", "source": "agent"}'
```

### 5. Before you end your session

Save a summary of what you did:

```bash
curl -s -X POST http://127.0.0.1:8100/remember \
  -H "Content-Type: application/json" \
  -d '{"text": "Session summary: fixed the login bug by updating the JWT expiry check, deployed to staging, tests passing", "source": "agent"}'
```

### Agent integration pattern

For any agent framework (LangChain, CrewAI, AutoGen, custom), the pattern is:

```python
import requests

HYPHAE = "http://127.0.0.1:8100"

def recall(query, top_k=5):
    """Search memory before acting."""
    r = requests.post(f"{HYPHAE}/recall", json={"query": query, "top_k": top_k})
    return r.json().get("results", [])

def remember(text, source="agent"):
    """Save important facts after acting."""
    requests.post(f"{HYPHAE}/remember", json={"text": text, "source": source})

# At session start
requests.post(f"{HYPHAE}/session/set", json={"scope": {"project": "my-project"}})
context = recall("what was done last session")

# During work
remember("discovered that the API rate limits at 100 req/min")

# When stuck
from_memory = recall("how to handle rate limiting")
```

## HTTP API Reference

### POST /remember
Store a fact in memory.

```json
{
  "text": "The fix for the 502 was adding proxy_pass to nginx",
  "source": "agent",
  "tags": {"project": "myapp", "type": "fix"},
  "context_id": "session-123",
  "cause_of": "fact-id-of-the-problem"
}
```

Returns: `{"id": "fact-uuid", "cluster_id": 42}`

### POST /recall
Search for facts by semantic similarity.

```json
{
  "query": "nginx 502 error fix",
  "top_k": 10,
  "scope": {"project": "myapp"}
}
```

- Omit `scope` to use the session scope (set via `/session/set`)
- Pass `"scope": {}` to search across ALL projects

Returns: `{"results": [{"text": "...", "score": 0.87, "tags": {...}, ...}]}`

### POST /session/set
Set the active session scope. Auto-warms all facts in this scope.

```json
{"scope": {"project": "myapp"}}
```

### POST /session/clear
Clear the session scope. Auto-distills a briefing before clearing.

### POST /analyze
Gap detection - find what's missing between observations and an objective.

```json
{
  "observations": ["found open port 6379", "Redis no auth"],
  "objective": "get shell access"
}
```

Returns gaps with suggested facts to fill them.

### GET /health
Returns `{"status": "ok", "facts": 3042, "clusters": 296}`

### GET /stats
Detailed statistics: fact counts, cluster sizes, manifold coverage, FAISS status, background builder health.

### POST /maintain
Trigger manual maintenance: merge clusters, build manifolds, auto-distill, report decay stats.

### GET /tower/list
List all Ivory Towers.

### GET /tower/walk/{name}
Walk through all floors of a tower.

### POST /tower/seed
Create a new tower with initial floors.

## MCP Server (Claude Code)

Hyphae integrates with Claude Code via the Model Context Protocol. Start the HTTP server first, then add the MCP config.

### Setup

1. Start the Hyphae server:
```bash
hyphae
```

2. Add to your Claude Code MCP config (`~/.claude/settings.json` or project-level):
```json
{
  "mcpServers": {
    "hyphae": {
      "command": "python",
      "args": ["-m", "hyphae.mcp_server"],
      "env": {}
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `recall_memory(query, top_k)` | Search memory for relevant facts. Scoped to current project by default. Use when past context would help - errors seen before, decisions made, architecture details. |
| `remember_fact(text, source)` | Save a fact to memory. Auto-tagged with current project. Use for decisions, discoveries, solutions, preferences. |
| `recall_all_projects(query, top_k)` | Search across ALL projects (unscoped). Use when you don't know which project something is in, or for cross-project questions. |
| `memory_status()` | Check if Hyphae is running, how many facts/clusters exist, manifold coverage. |

### How the agent should use it

- **On session start:** `recall_memory("what was done last session")` to get context
- **When encountering a problem:** `recall_memory("specific error message or topic")` to check if it's been solved before
- **After making decisions:** `remember_fact("decided to use X because Y")`
- **After solving problems:** `remember_fact("the fix for X was Y")`
- **When user corrects you:** `remember_fact("user prefers X over Y because Z")`

## Architecture

```
Facts (text + embedding + tags)
  -> Clusters (semantic grouping, adaptive merge/split)
    -> Manifolds (diffusion maps per cluster, built in background)
      -> Geodesic re-ranking (manifold distance blended with cosine)

Towers (structured knowledge buildings)
  -> Floors (themed content synthesized from nearby facts)
    -> Connections (typed links between towers)
```

### How manifold re-ranking works

When a cluster accumulates 30+ facts, Hyphae builds a diffusion map manifold over it:

1. k-NN graph with locally-scaled Gaussian kernels
2. Normalized graph Laplacian eigendecomposition
3. 6-dimensional diffusion map embedding

At recall time, results from clusters with manifolds get re-ranked by blending cosine similarity (60%) with manifold proximity (40%). This finds facts that are *connected through chains* rather than just textually similar.

See the paper: [Geodesic Retrieval over Learned Manifolds](https://zenodo.org/records/18971939) (DOI: 10.5281/zenodo.18971939)

## Configuration

Hyphae stores its database at `~/.hyphae/hyphae.db` by default. Override with:

```python
from hyphae import Hyphae
h = Hyphae(db_path="/path/to/my.db")
```

### Embedding model

Default: `all-MiniLM-L6-v2` (384 dimensions, runs locally via sentence-transformers).

```python
h = Hyphae(model="all-MiniLM-L6-v2")
```

### Remote shards

Connect to external knowledge stores:

```python
h = Hyphae(remote_shards=[
    {"endpoint": "http://192.168.0.224:8000", "name": "memoria", "timeout": 15}
])
```

## Python API

```python
from hyphae import Hyphae

h = Hyphae()

# Remember
fact_id, cluster_id = h.remember("important discovery", tags={"project": "myapp"})

# Recall
facts = h.recall("what did I discover?", top_k=5)
for f in facts:
    print(f"[{f.score:.2f}] {f.text}")

# Session scoping
h.set_session({"project": "myapp"})  # auto-scopes all recalls
facts = h.recall("deployment")        # only returns myapp facts

# Gap analysis
analysis = h.analyze(
    observations=["server returned 502", "nginx running"],
    objective="fix the 502 error"
)

# Maintenance (runs automatically, or call manually)
report = h.maintain()

# Briefing
briefing = h.get_briefing("myapp")
print(briefing.text)
```

## Requirements

- Python 3.10+
- numpy, scipy, scikit-learn, sentence-transformers
- Optional: faiss-cpu (faster vector search), fastapi + uvicorn (HTTP server)

## License

MIT
