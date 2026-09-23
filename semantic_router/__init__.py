"""Semantic router / triage engine for support tickets, powered by Laya (System-1 decision model)."""

from .schemas import DEFAULT_QUEUES, QueueSpec, RoutingPolicy, Ticket, TriageResult
from .triage import TriageEngine

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_QUEUES",
    "QueueSpec",
    "RoutingPolicy",
    "Ticket",
    "TriageResult",
    "TriageEngine",
    "__version__",
]
