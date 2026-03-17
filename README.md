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
pip install hyphae[all]
hyphae
```

This starts the server on `http://127.0.0.1:8100`. Facts are stored in `~/.hyphae/hyphae.db`.

## API

### Remember a fact
```bash
curl -X POST http://127.0.0.1:8100/remember \
  -H "Content-Type: application/json" \
  -d '{"text": "Redis CONFIG SET can write to authorized_keys for SSH access", "source": "agent"}'
```

### Recall facts
```bash
curl -X POST http://127.0.0.1:8100/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "SSH key injection", "top_k": 5}'
```

### Scoped recall (project-specific)
```bash
curl -X POST http://127.0.0.1:8100/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "deployment config", "top_k": 5, "scope": {"project": "myapp"}}'
```

### Set session scope
```bash
curl -X POST http://127.0.0.1:8100/session/set \
  -H "Content-Type: application/json" \
  -d '{"scope": {"project": "myapp"}}'
```

### Gap analysis
```bash
curl -X POST http://127.0.0.1:8100/analyze \
  -H "Content-Type: application/json" \
  -d '{"observations": ["found open port 6379", "Redis no auth"], "objective": "get shell access"}'
```

### Health check
```bash
curl http://127.0.0.1:8100/health
```

### Stats
```bash
curl http://127.0.0.1:8100/stats
```

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

## MCP Server (Claude Code integration)

Add to your Claude Code MCP config:

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

This gives Claude Code tools: `recall_memory`, `remember_fact`, `recall_all_projects`, `memory_status`, `learn_topic`.

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

Connect to external knowledge stores (e.g., Memoria):

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
