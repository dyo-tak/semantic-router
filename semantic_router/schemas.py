"""Typed models for the triage domain (pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high", "critical"]
Action = Literal["auto_resolve", "route", "escalate"]


class Ticket(BaseModel):
    id: str | None = None
    subject: str
    body: str
    customer_tier: str = "standard"  # free | standard | premium | enterprise


class QueueSpec(BaseModel):
    """A routing destination."""

    name: str
    description: str


class RoutingPolicy(BaseModel):
    """Thresholds for the triage decision tree. Tune here, not in code."""

    urgency_score_type: str = "urgency"

    # noul gate: below this P(true) the ticket is not urgent -> no LLM spend
    urgent_noul_threshold: float = 0.55
    # score rubric level (0..3) at/above which the ticket is escalated to an LLM review
    escalate_score_threshold: float = 2.0
    # minimum choice confidence to trust the queue routing without LLM review
    min_route_confidence: float = 0.60
    # premium/enterprise customers below the confidence bar get LLM review instead of fallback queue
    llm_review_tiers: list[str] = Field(default_factory=lambda: ["premium", "enterprise"])
    # fallback queue when no confident routing decision exists
    fallback_queue: str = "general"


DEFAULT_QUEUES = [
    QueueSpec(name="billing", description="payments, refunds, invoices, subscription charges"),
    QueueSpec(name="support", description="product help, bugs, how-to questions"),
    QueueSpec(name="sales", description="new purchases, upgrades, pricing questions"),
    QueueSpec(name="security", description="account compromise, phishing, data privacy, breaches"),
]


class TriageResult(BaseModel):
    ticket_id: str | None = None
    urgent: bool
    urgency_noul: float
    urgency_score: float | None = None
    queue: str | None = None
    queue_confidence: float | None = None
    queue_probabilities: dict[str, float] | None = None
    action: Action
    latency_ms: float
    llm_calls: int = 0
    reason: str
