"""Triage policy tests with a stubbed Laya engine — no weights needed."""

import pytest

from semantic_router.schemas import RoutingPolicy, Ticket, TriageResult
from semantic_router.triage import TriageEngine


class StubLaya:
    """Returns canned answers without touching the model."""

    def __init__(self, noul=0.1, score=0.5, choice="billing", conf=0.9):
        self.noul = noul
        self.score = score
        self.choice = choice
        self.conf = conf

    def system_one(self, state, questions):
        return {
            "model": "laya",
            "answers": {
                "is_urgent": {"type": "noul", "noul": self.noul, "rl_agent": {"act_probability": 0.5}},
                "urgency": {
                    "type": "score",
                    "score": self.score,
                    "legend": {},
                    "probabilities": {},
                    "confidence": 0.5,
                    "rl_agent": {"act_probability": 0.5},
                },
                "queue": {
                    "type": "choice",
                    "choice": self.choice,
                    "probabilities": {self.choice: self.conf},
                    "confidence": self.conf,
                    "rl_agent": {"act_probability": 0.5},
                },
            },
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "latency_ms": 1.0,
        }


TICKET = Ticket(id="x", subject="s", body="b", customer_tier="standard")


def _engine(noul=0.1, score=0.5, choice="billing", conf=0.9, reviewer=None, policy=None):
    return TriageEngine(StubLaya(noul, score, choice, conf), policy=policy, llm_reviewer=reviewer)


def test_not_urgent_routes_on_device_no_llm():
    r = _engine(noul=0.1, conf=0.9).triage(TICKET)
    assert r.action == "route"
    assert r.queue == "billing"
    assert r.llm_calls == 0
    assert not r.urgent


def test_urgent_high_score_escalates():
    r = _engine(noul=0.9, score=2.5).triage(TICKET)
    assert r.action == "escalate"
    assert r.urgent


def test_urgent_but_moderate_score_routes():
    r = _engine(noul=0.7, score=1.2, conf=0.9).triage(TICKET)
    assert r.action == "route"
    assert r.queue == "billing"


def test_low_confidence_standard_falls_back():
    pol = RoutingPolicy(fallback_queue="general")
    r = _engine(noul=0.1, conf=0.3, policy=pol).triage(TICKET)
    assert r.action == "route"
    assert r.queue == "general"


def test_low_confidence_premium_escalates_to_llm():
    called = []

    def reviewer(t, hint):
        called.append(hint)
        return {"queue": "security"}

    r = _engine(noul=0.1, conf=0.3, reviewer=reviewer).triage(
        Ticket(id="x", subject="s", body="b", customer_tier="premium")
    )
    assert r.action == "escalate"
    assert r.llm_calls == 1
    assert r.queue == "security"
    assert called


def test_escalate_without_reviewer_keeps_queue_none():
    r = _engine(noul=0.9, score=2.8).triage(TICKET)
    assert r.action == "escalate"
    assert r.queue is None


def test_policy_thresholds_respected():
    pol = RoutingPolicy(urgent_noul_threshold=0.95, escalate_score_threshold=3.5)
    r = _engine(noul=0.9, score=2.5, policy=pol).triage(TICKET)
    # with a 0.95 gate, noul 0.9 counts as not urgent -> plain route
    assert r.action == "route"


def test_result_is_pydantic():
    r = _engine().triage(TICKET)
    assert isinstance(r, TriageResult)
    assert r.latency_ms >= 0
