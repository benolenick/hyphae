"""Shared data structures for Hyphae."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Fact:
    text: str
    id: str = ""
    embedding: Optional[np.ndarray] = None
    tags: dict[str, str] = field(default_factory=dict)
    source: str = ""
    context_id: str = ""  # groups facts from same work session
    cluster_id: int = -1
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    access_count: int = 0
    score: float = 0.0  # relevance score from search

    def __post_init__(self):
        if not self.id:
            self.id = _content_hash(self.text)
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_accessed_at:
            self.last_accessed_at = self.created_at


@dataclass
class CausalLink:
    from_fact_id: str
    to_fact_id: str
    relation: str = "caused_by"  # caused_by, led_to, required_for, etc.
    context_id: str = ""
    confidence: float = 1.0


@dataclass
class ClusterState:
    id: int
    centroid: np.ndarray
    fact_ids: list[str] = field(default_factory=list)
    count: int = 0
    created_at: float = 0.0
    dirty: bool = True  # needs manifold recompute
    manifold_coords: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None
    local_embeddings: Optional[np.ndarray] = None  # stacked embeddings for this cluster
    nystrom_extensions: int = 0  # count of Nystrom extensions since last full recompute

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()


@dataclass
class Gap:
    position: int
    edge_cost: float
    normalized_cost: float
    from_fact_text: str
    to_fact_text: str
    from_idx: int
    to_idx: int
    midpoint_coords: np.ndarray
    cluster_id: int = -1
    retrieved_facts: list[Fact] = field(default_factory=list)


@dataclass
class GapAnalysis:
    observations: list[str]
    objective: str
    gaps: list[Gap]
    retrieved_knowledge: list[Fact]
    clusters_used: list[int]
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== Gap Analysis ===",
            f"Objective: {self.objective}",
            f"Clusters: {self.clusters_used}",
            f"Gaps: {len(self.gaps)}",
            f"Retrieved: {len(self.retrieved_knowledge)} facts",
            f"Elapsed: {self.elapsed_sec:.2f}s",
        ]
        for g in self.gaps[:3]:
            lines.append(f"\nGAP #{g.position} cluster={g.cluster_id} (cost={g.edge_cost:.4f}, {g.normalized_cost:.1f}x avg)")
            lines.append(f"  From: {g.from_fact_text[:100]}")
            lines.append(f"  To:   {g.to_fact_text[:100]}")
            lines.append(f"  Fill: {len(g.retrieved_facts)} facts")
            for rf in g.retrieved_facts[:3]:
                lines.append(f"    - {rf.text[:120]}")
        return "\n".join(lines)


@dataclass
class Stone:
    id: str = ""
    goal: str = ""
    success_marker: str = ""
    failure_marker: str = ""
    status: str = "pending"  # pending, attempting, succeeded, failed, blocked
    attempts: int = 0
    max_attempts: int = 3
    created_at: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = _content_hash(f"{self.goal}{time.time()}")
        if not self.created_at:
            self.created_at = time.time()


@dataclass
class River:
    id: str = ""
    objective: str = ""
    stones: list[Stone] = field(default_factory=list)
    current_stone_idx: int = 0
    status: str = "active"  # active, completed, abandoned
    created_at: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = _content_hash(f"{self.objective}{time.time()}")
        if not self.created_at:
            self.created_at = time.time()

    @property
    def current_stone(self) -> Optional[Stone]:
        if 0 <= self.current_stone_idx < len(self.stones):
            return self.stones[self.current_stone_idx]
        return None


@dataclass
class Briefing:
    project: str
    text: str
    facts_used: list[str] = field(default_factory=list)
    id: str = ""
    created_at: float = 0.0
    token_count: int = 0
    is_fallback: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = _content_hash(f"{self.project}:{self.text}")
        if not self.created_at:
            self.created_at = time.time()
        if not self.token_count:
            self.token_count = len(self.text.split())
