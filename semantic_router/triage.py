"""TriageEngine — the decision tree around Laya.

Policy: Laya answers (1) noul "is this urgent?" and (2) choice "which queue?" and
(3) score "how urgent 0..3?" in ONE forward pass (one ~50ms inference). The router then:

  1. noul < threshold            -> not urgent: route on Laya's choice alone, zero LLM spend
  2. score >= escalate_threshold -> hand to LLM for nuanced handling (escalate)
  3. choice confidence < bar     -> low-confidence routing: LLM review for premium/enterprise, else fallback queue
  4. otherwise                   -> Laya routes it fully on-device

So the LLM is only consulted for genuinely ambiguous or critical tickets (~10% target).
"""

from __future__ import annotations

import time
from typing import Any

from .schemas import (
    DEFAULT_QUEUES,
    RoutingPolicy,
    Ticket,
    TriageResult,
)
from .laya_engine.laya import LayaEngine


class TriageEngine:
    def __init__(
        self,
        laya: LayaEngine,
        policy: RoutingPolicy | None = None,
        queues: list[dict[str, str]] | None = None,
        llm_reviewer: Any | None = None,
    ):
        self.laya = laya
        self.policy = policy or RoutingPolicy()
        qs = queues or [{"name": q.name, "description": q.description} for q in DEFAULT_QUEUES]
        self.queues = {q["name"]: q for q in qs}
        # llm_reviewer: optional callable(ticket, triage_hint) -> {"queue": str, "reply": str}|None
        self.llm_reviewer = llm_reviewer

    def _questions(self) -> dict[str, dict]:
        queue_criteria = {name: q["description"] for name, q in self.queues.items()}
        return {
            "is_urgent": {
                "type": "noul",
                "instructions": "This ticket needs attention within the hour (outage, security incident, angry customer, money lost).",
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"],
            },
            "queue": {
                "type": "choice",
                "instructions": "Which team should handle this ticket?",
                "criteria": queue_criteria,
            },
        }

    def triage(self, ticket: Ticket) -> TriageResult:
        t0 = time.perf_counter()
        state = {
            "subject": ticket.subject,
            "body": ticket.body,
            "customer_tier": ticket.customer_tier,
        }
        res = self.laya.system_one(state, self._questions())
        a = res["answers"]
        noul = float(a["is_urgent"]["noul"])
        score = float(a["urgency"]["score"])
        pol = self.policy

        queue = None
        conf = None
        probs = a["queue"].get("probabilities")
        # routing confidence = probability mass on the picked queue (interpretable),
        # not Jev entropy-confidence which reads low even for peaked 4-way distributions
        conf = max(probs.values()) if probs else float(a["queue"].get("confidence", 0.0))
        action: str = "route"
        reason = ""

        if noul < pol.urgent_noul_threshold:
            # not urgent — trust Laya's routing, never call the LLM
            queue = str(a["queue"]["choice"])
            if conf < pol.min_route_confidence:
                action, reason, queue = self._low_confidence(ticket, a, queue)
                if action == "escalate":
                    pass
                else:
                    reason = f"not urgent (noul={noul:.2f}); low routing confidence ({conf:.2f}) -> {queue}"
            else:
                action, reason = "route", f"not urgent (noul={noul:.2f}); routed on-device"
        elif score >= pol.escalate_score_threshold:
            action, reason = "escalate", f"urgent (noul={noul:.2f}) and severity score {score:.2f} >= {pol.escalate_score_threshold}"
            queue = None
        else:
            queue = str(a["queue"]["choice"])
            if conf < pol.min_route_confidence:
                action, reason, queue = self._low_confidence(ticket, a, queue)
                if action != "escalate":
                    reason = f"urgent (noul={noul:.2f}), but low routing confidence ({conf:.2f}) -> {queue}"
            else:
                action, reason = "route", f"urgent (noul={noul:.2f}), routed on-device with confidence {conf:.2f}"

        llm_calls = 0
        if action == "escalate" and self.llm_reviewer is not None:
            llm_calls = 1
            review = self.llm_reviewer(ticket, {"urgency_noul": noul, "urgency_score": score, "laya_queue": a["queue"]})
            if isinstance(review, dict) and review.get("queue"):
                queue = review["queue"]

        return TriageResult(
            ticket_id=ticket.id,
            urgent=noul >= pol.urgent_noul_threshold,
            urgency_noul=noul,
            urgency_score=score,
            queue=queue,
            queue_confidence=conf,
            queue_probabilities=probs,
            action=action,  # type: ignore[arg-type]
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            llm_calls=llm_calls,
            reason=reason,
        )

    def _low_confidence(self, ticket: Ticket, answers: dict, laya_queue: str) -> tuple[str, str, str | None]:
        pol = self.policy
        if ticket.customer_tier in pol.llm_review_tiers and self.llm_reviewer is not None:
            return ("escalate", f"low routing confidence for {ticket.customer_tier} customer -> LLM review", None)
        return ("route", f"low routing confidence -> fallback queue {pol.fallback_queue}", pol.fallback_queue)
