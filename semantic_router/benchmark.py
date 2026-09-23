"""Benchmark harness: Laya triage engine vs OpenRouter LLM baseline.

Measures, per system: queue accuracy, urgency MAE, LLM-call/escalation rate, latency
(median / p95). Laya runs fully on-device; the baseline pays a network round-trip per ticket.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

from .dataset import LabeledTicket, load_labeled
from .schemas import Ticket
from .triage import TriageEngine


def _pack(lat: list[float], calls: int, n: int, q_ok: int, urg_err: list[float]) -> dict:
    lat_sorted = sorted(lat)
    return {
        "n": n,
        "queue_accuracy": round(q_ok / n, 3),
        "urgency_mae": round(statistics.mean(urg_err), 3),
        "llm_calls": calls,
        "llm_call_rate": round(calls / n, 3),
        "latency_ms": {"median": round(statistics.median(lat_sorted), 1), "p95": round(lat_sorted[int(0.95 * (len(lat_sorted) - 1))], 1)},
    }


def laya_metrics(engine: TriageEngine, tickets: list[LabeledTicket]) -> dict:
    lat, calls, q_ok, urg_err = [], 0, 0, []
    for t in tickets:
        r = engine.triage(Ticket(id=t.id, subject=t.subject, body=t.body, customer_tier=t.customer_tier))
        lat.append(r.latency_ms)
        calls += r.llm_calls
        pred_q = r.queue or "escalated"
        # correct if the queue matches, or the label says escalate and we escalated
        handled = pred_q == t.queue or (r.action == "escalate" and t.escalate)
        q_ok += int(handled)
        urg_err.append(abs(r.urgency_score - t.urgency))
    return _pack(lat, calls, len(tickets), q_ok, urg_err)


def openrouter_metrics(baseline, tickets: list[LabeledTicket], pace_s: float = 3.5) -> dict:
    lat, q_ok, urg_err = [], 0, []
    for i, t in enumerate(tickets):
        if i:
            time.sleep(pace_s)  # stay under :free per-minute rate limits
        d = baseline.triage(t)
        lat.append(d["latency_ms"])
        q_ok += int(d["queue"] == t.queue)
        urg_err.append(abs(d["urgency_score"] - t.urgency))
    # the LLM baseline makes exactly one call per ticket by construction
    return _pack(lat, len(tickets), len(tickets), q_ok, urg_err)


def run_benchmark(
    dataset: list[LabeledTicket] | None = None,
    model_dir: str | Path | None = None,
    openrouter_model: str | None = None,
    limit: int | None = None,
    warmup: int = 3,
    intra_op_threads: int = 2,
    out: Path | None = None,
) -> list[dict]:
    from .baselines import OpenRouterTriage
    from .laya_engine.laya import LayaEngine

    if dataset is None:
        dataset = load_labeled()
    if limit:
        dataset = dataset[:limit]

    engine = TriageEngine(LayaEngine(model_dir, intra_op_threads=intra_op_threads))
    for t in dataset[:warmup]:
        engine.triage(t)
    laya_m = laya_metrics(engine, dataset)
    rows: list[dict] = [{"system": "laya (on-device)", **laya_m}]

    if openrouter_model and os.environ.get("OPENROUTER_API_KEY"):
        baseline = OpenRouterTriage(openrouter_model)
        for t in dataset[:warmup]:
            baseline.triage(t)
        rows.append({"system": f"openrouter ({openrouter_model})", **openrouter_metrics(baseline, dataset)})
    else:
        rows.append({"system": "openrouter baseline", "skipped": "set OPENROUTER_API_KEY and --openrouter-model to include"})

    text = json.dumps(rows, indent=2)
    print(text)
    if out:
        out.write_text(text)
    return rows
