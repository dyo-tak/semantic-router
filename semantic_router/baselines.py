"""OpenRouter baseline — an LLM does the whole triage decision per ticket (the 'before' world).

Uses structured output via OpenRouter's chat-completions API with a free-tier model.
"""

from __future__ import annotations

import json
import os
import time

import httpx

PROMPT = """You are a support-triage system. Analyze the ticket and answer in JSON:
{"is_urgent": bool, "urgency": 0-3 integer (0 not urgent, 3 critical), "queue": one of %(queues)s}

Ticket:
Subject: %(subject)s
Body: %(body)s
Customer tier: %(tier)s

Respond with ONLY the JSON object."""


class OpenRouterTriage:
    name = "openrouter"

    def __init__(self, model: str, api_key: str | None = None, queues: list[str] | None = None):
        self.model = model
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.queues = queues or ["billing", "support", "sales", "security"]
        self.infer_ms: float | None = None  # network latency incl. model time, set per call

    def triage(self, ticket) -> dict:
        prompt = PROMPT % {
            "queues": ", ".join(self.queues),
            "subject": ticket.subject,
            "body": ticket.body,
            "tier": ticket.customer_tier,
        }
        t0 = time.perf_counter()
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 100,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        latency = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        d = json.loads(content)
        urgency = max(0, min(3, int(d.get("urgency", 0))))
        return {
            "is_urgent": bool(d.get("is_urgent", urgency >= 2)),
            "urgency_score": float(urgency),
            "queue": str(d.get("queue", "general")) if str(d.get("queue", "")) in self.queues else "general",
            "latency_ms": latency,
        }
