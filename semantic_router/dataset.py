"""Labeled support-ticket dataset for the triage benchmark (48 tickets).

Labels: queue (billing/support/sales/security), urgency (0-3), escalate (should go to a human/LLM).
Synthetic but realistic; each row: {id, subject, body, customer_tier, queue, urgency, escalate}.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import Ticket

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "tickets_labeled.jsonl"


class LabeledTicket(Ticket):
    queue: str
    urgency: int
    escalate: bool = False


def load_labeled(path: Path | str = DATA) -> list[LabeledTicket]:
    items: list[LabeledTicket] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(LabeledTicket(**json.loads(line)))
    return items


def rows() -> list[dict]:
    out = []
    with open(DATA) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
