"""Conversation processing — extract facts and gate by novelty.

Takes raw conversation turns (user/assistant messages) and decides what's
worth remembering. Not every message is a fact — we want decisions,
discoveries, errors, solutions, and technical details.
"""
from __future__ import annotations

import logging
import re
import time

from .types import Fact

logger = logging.getLogger("hyphae.converse")

# Patterns that signal a message contains something worth remembering
SIGNAL_PATTERNS = [
    # Decisions and conclusions
    r"\b(?:decided|decision|chose|choosing|going with|opted for|settled on)\b",
    r"\b(?:because|reason is|the issue was|root cause|turns out)\b",
    r"\b(?:solution|fix|workaround|resolved|fixed)\b",
    # Discoveries
    r"\b(?:found|discovered|noticed|realized|learned|TIL)\b",
    r"\b(?:CVE-\d{4}-\d+)\b",
    r"\b(?:the problem is|the issue is|it fails because|broke because)\b",
    # Technical facts
    r"\b(?:runs on|hosted at|deployed to|lives at|stored in)\b",
    r"\b(?:port \d+|version \d+)\b",
    r"\b(?:password|credential|token|key|secret)\b",
    r"\b(?:IP|subnet|CIDR|VLAN)\b",
    # Architecture and design
    r"\b(?:architecture|design|pattern|approach|strategy|workflow)\b",
    r"\b(?:we should|we need to|next step|TODO|action item)\b",
    # Errors and failures
    r"\b(?:error|exception|traceback|failed|crash|timeout|refused)\b",
    r"\b(?:doesn't work|broken|bug|regression)\b",
]

# Patterns that signal a message is NOT worth remembering (noise)
NOISE_PATTERNS = [
    r"^(?:ok|okay|yes|no|sure|thanks|thank you|got it|sounds good|lgtm)\s*[.!]?\s*$",
    r"^(?:hi|hello|hey|yo)\b",
    r"^(?:can you|could you|please|would you)\b",  # questions/requests
    r"^(?:what|how|why|where|when|who|which)\b.*\?$",  # pure questions
    r"^\.?/",  # slash commands
]

# Minimum message length to consider (skip tiny messages)
MIN_LENGTH = 40

# Maximum fact length to store
MAX_FACT_LENGTH = 500

# Novelty threshold — if a fact is this similar to an existing one, skip it
NOVELTY_THRESHOLD = 0.85


def has_signal(text: str) -> bool:
    """Check if text contains patterns worth remembering."""
    for pattern in SIGNAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def is_noise(text: str) -> bool:
    """Check if text is conversational noise."""
    stripped = text.strip()
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True
    return False


def extract_facts(text: str) -> list[str]:
    """Extract fact-worthy sentences from a message.

    Splits on sentence boundaries and keeps sentences that contain
    signal patterns. Returns deduplicated list of fact strings.
    """
    if len(text) < MIN_LENGTH:
        return []

    if is_noise(text):
        return []

    # Split into sentences (rough but good enough)
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+|\n(?=-\s|\d+\.)', text)

    facts = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 25:
            continue
        # Skip code blocks and command output
        if sent.startswith("```") or sent.startswith("$") or sent.startswith(">>>"):
            continue
        if has_signal(sent):
            # Truncate to max length
            fact = sent[:MAX_FACT_LENGTH]
            facts.append(fact)

    # If no individual sentences have signals, but the whole message does,
    # store a condensed version of the whole message
    if not facts and has_signal(text):
        # Take first meaningful chunk
        condensed = text[:MAX_FACT_LENGTH].strip()
        if len(condensed) >= MIN_LENGTH:
            facts.append(condensed)

    return facts


def check_novelty(fact_text: str, shard, embedder, threshold: float = NOVELTY_THRESHOLD) -> bool:
    """Return True if this fact is novel (not too similar to existing facts).

    Embeds the candidate fact and checks against existing facts.
    Returns True if the fact should be stored (it's novel enough).
    """
    embedding = embedder.encode_single(fact_text)
    existing = shard.search(embedding, top_k=1)
    if existing and existing[0].score >= threshold:
        logger.debug(
            f"Novelty gate: rejected (sim={existing[0].score:.3f}): {fact_text[:60]}..."
        )
        return False
    return True


def process_turn(
    role: str,
    message: str,
    hyphae,
    source: str = "conversation",
) -> list[dict]:
    """Process a conversation turn — extract, gate, and remember facts.

    Args:
        role: "user" or "assistant"
        message: The message text
        hyphae: Hyphae instance (for remember + novelty checking)
        source: Source tag for stored facts

    Returns:
        List of {"fact_id", "cluster_id", "text"} for each stored fact.
    """
    facts = extract_facts(message)
    if not facts:
        return []

    stored = []
    for fact_text in facts:
        # Prefix with role for context
        tagged = f"[{role}] {fact_text}" if role == "user" else fact_text

        # Novelty gate
        if not check_novelty(tagged, hyphae.local_shard, hyphae.embedder):
            continue

        # Store with session scope auto-applied by hyphae.remember()
        fact_id, cluster_id = hyphae.remember(
            tagged,
            tags={"type": "conversation", "role": role},
            source=source,
        )
        stored.append({
            "fact_id": fact_id,
            "cluster_id": cluster_id,
            "text": tagged[:200],
        })

    if stored:
        logger.info(f"Conversation: stored {len(stored)}/{len(facts)} facts from {role}")

    return stored
