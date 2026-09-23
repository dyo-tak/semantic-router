"""CLI: smoke-triage a ticket, or run the benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .schemas import Ticket
from .triage import TriageEngine

console = Console()

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "laya-onnx"


def _load_engine(model_dir, threads):
    from .laya_engine.laya import LayaEngine

    return LayaEngine(model_dir or DEFAULT_MODEL_DIR, intra_op_threads=threads)


def cmd_triage(args):
    engine = TriageEngine(_load_engine(args.model_dir, args.threads))
    t = Ticket(
        subject=args.subject,
        body=args.body,
        customer_tier=args.tier,
    )
    r = engine.triage(t)
    table = Table(title="Triage result", show_header=False)
    table.add_row("action", r.action)
    table.add_row("queue", str(r.queue))
    table.add_row("urgent", f"{r.urgent} (noul={r.urgency_noul})")
    table.add_row("urgency score", str(r.urgency_score))
    table.add_row("confidence", str(r.queue_confidence))
    table.add_row("llm calls", str(r.llm_calls))
    table.add_row("latency", f"{r.latency_ms} ms")
    table.add_row("reason", r.reason)
    console.print(table)
    print(r.model_dump_json(indent=2))


def cmd_bench(args):
    from .benchmark import run_benchmark
    from .dataset import load_labeled

    run_benchmark(
        dataset=load_labeled(args.dataset),
        model_dir=args.model_dir,
        openrouter_model=args.openrouter_model,
        limit=args.limit,
        warmup=args.warmup,
        intra_op_threads=args.threads,
        out=Path(args.out) if args.out else None,
    )


def main():
    ap = argparse.ArgumentParser(prog="semantic-router")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triage", help="triage a single ticket with Laya")
    t.add_argument("--subject", required=True)
    t.add_argument("--body", required=True)
    t.add_argument("--tier", default="standard")
    t.add_argument("--model-dir", default=None)
    t.add_argument("--threads", type=int, default=4)
    t.set_defaults(func=cmd_triage)

    b = sub.add_parser("bench", help="benchmark Laya vs OpenRouter baseline")
    b.add_argument("--dataset", default="data/tickets_labeled.jsonl")
    b.add_argument("--openrouter-model", default=None, help="e.g. meta-llama/llama-3.3-70b-instruct:free")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--warmup", type=int, default=3)
    b.add_argument("--model-dir", default=None)
    b.add_argument("--threads", type=int, default=2, help="ONNX intra-op threads (match vCPU count)")
    b.add_argument("--out", default=None, help="write JSON results here")
    b.set_defaults(func=cmd_bench)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
