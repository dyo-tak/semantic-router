# semantic-router

**Semantic triage engine for support tickets — System-1 routing with Laya, LLM only for escalation.**

Every ticket is classified, scored and routed by [Laya](https://huggingface.co/convaiinnovations/laya)
(a ModernBERT-large decision model, ONNX, single forward pass) running **fully on-device**.
The LLM is consulted only for genuinely ambiguous or critical tickets — target: **~10% escalation rate**.

## Why

A classic LLM pipeline pays a full generation round-trip for *every* ticket: slow (seconds),
costly, and your ticket text leaves the box. Here, one ~30–200 ms on-device forward pass answers
three typed questions at once:

| primitive | question | use |
|---|---|---|
| `noul` | "needs attention within the hour?" — calibrated P(true) | urgency gate |
| `score` | "how urgent?" 0–3 on an ordered rubric | severity, escalation |
| `choice` | "which team?" billing / support / sales / security | queue routing |

## Decision policy

```
ticket ──> Laya (one forward pass: noul + score + choice)
   │
   ├─ noul < 0.55 ................. not urgent → route on-device, 0 LLM calls
   ├─ score ≥ 2.0 (urgent)  ....... escalate to LLM/human review
   ├─ queue prob < 0.60 ........... low confidence → premium/enterprise get LLM review, rest fall back to `general`
   └─ otherwise ................... routed on-device with calibrated confidence
```

All thresholds live in `RoutingPolicy` — tune there, not in code.

## Quickstart

```sh
uv sync
uv run python -m semantic_router.download          # ~1.7 GB ONNX bundle from HF
uv run python -m semantic_router.cli triage \
  --subject "Suspicious login alerts" \
  --body "6 login alerts from Tokyo, I'm in Chicago" \
  --tier standard
```

## Benchmark

```sh
export OPENROUTER_API_KEY=...
uv run python -m semantic_router.cli bench \
  --openrouter-model <model:free> --threads 2
```

On the 36-ticket labeled set (`data/tickets_labeled.jsonl`) it reports queue accuracy,
urgency MAE, LLM-call rate and median/p95 latency for **Laya on-device vs the OpenRouter
LLM baseline**, which pays one generation round-trip per ticket.

## Repo layout

```
semantic_router/
  laya_engine/sequence.py   # pure port of Laya's request rendering (unit-tested, no weights)
  laya_engine/laya.py       # ONNX engine: system_one() in a single forward pass
  triage.py                 # TriageEngine: noul gate + score escalation + choice routing
  baselines.py              # OpenRouter LLM baseline ("before" world: LLM per ticket)
  benchmark.py              # accuracy / MAE / latency / escalation-rate harness
  dataset.py, schemas.py, cli.py, download.py
data/tickets_labeled.jsonl  # 36 labeled tickets (queue, urgency 0-3, escalate)
models/laya-onnx/           # gitignored ONNX bundle
```

## Tests

`uv run pytest` — 24 model-free tests (sequence layout matches the TS reference bit-for-bit
given the same tokenizer, policy branches, dataset integrity). Weighted smoke test:
`uv run pytest -m model`.

## License

MIT
