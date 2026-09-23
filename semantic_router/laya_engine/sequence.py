"""Pure (model-free) port of Laya's request rendering — checkpoint `rl_common.py`.

Sequence layout, option rendering, temperature buckets and confidence, exactly as
github.com/receptron/laya src/sequence.ts. Unit-testable without weights.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, TypedDict

QTYPES: dict[str, int] = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = ["choice", "score", "noul"]


class Question(TypedDict, total=False):
    type: str  # choice | score | noul
    instructions: str
    # choice: option -> description (or plain list); score: ordered levels; noul: {true, false}
    criteria: Any


class InternalQ(TypedDict):
    t: str
    ins: str
    crit: Any


def to_internal(q: Question) -> InternalQ:
    crit = q.get("criteria")
    if q["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = q["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)
    return {"t": q["type"], "ins": ins, "crit": crit}


def render_options(q: InternalQ) -> list[str]:
    """Option texts in label-index order. Noul is always [false, true] so p[1] == noul."""
    if q["t"] == "choice":
        return [f"{k}: {v}" if v else k for k, v in q["crit"].items()]
    if q["t"] == "score":
        return [f"level {i}: {c}" for i, c in enumerate(q["crit"])]
    c = q.get("crit") or {}
    return [
        "false: " + (c.get("false") or "no, the statement does not hold"),
        "true: " + (c.get("true") or "yes, the statement holds"),
    ]


def serialize_state(state: Any) -> str:
    """Python json.dumps(ensure_ascii=False) — insertion key order, ', '/': ' separators."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def size_bucket(k: int) -> str:
    if k <= 2:
        return "2"
    if k <= 5:
        return "3-5"
    if k <= 10:
        return "6-10"
    return "11+"


def temp_bucket(qtype: int, k: int) -> str:
    """Key for the per-cardinality temperature."""
    return f"{QTYPE_NAMES[qtype]}:{size_bucket(k)}"


def confidence_from_probs(p: list[float]) -> float:
    """Jev-style confidence: 1 - normalized entropy of the answer distribution."""
    k = len(p)
    if k < 2:
        return 1.0
    ent = 0.0
    for x in p:
        ent -= x * math.log(max(x, 1e-12))
    return 1.0 - ent / math.log(k)


def softmax(z: list[float]) -> list[float]:
    zmax = max(z)
    e = [math.exp(v - zmax) for v in z]
    s = sum(e)
    return [v / s for v in e]


class SpecialIds(TypedDict):
    cls: int
    sep: int
    mask: int
    pad: int
    mask_tok: str  # literal mask token text, scrubbed from user text so it cannot inject a marker


Encode = Callable[[str], list[int]]


def build_sequence(
    encode: Encode,
    ids: SpecialIds,
    state: Any,
    q: InternalQ,
    max_len: int,
    head_max_len: int,
) -> tuple[list[int], list[int]]:
    """rl_common.build_sequence:

        [CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]

    Returns (ids, marker_positions).
    """
    def scrub(s: str) -> str:
        return s.replace(ids["mask_tok"], " ")

    opts = render_options(q)
    head_ids = encode(f"{q['t']} question: {scrub(q['ins'])}")
    opt_ids = [[ids["mask"], *encode(" " + scrub(o))[:48]] for o in opts]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        # too many / too long options: shrink every option text evenly
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]

    seq = [ids["cls"], *head_ids, ids["sep"]]
    markers: list[int] = []
    for o in opt_ids:
        markers.append(len(seq))
        seq.extend(o)
    seq.append(ids["sep"])
    room = max(0, max_len - len(seq) - 1)
    seq.extend(encode(scrub(serialize_state(state)))[:room])
    seq.append(ids["sep"])
    return seq[:max_len], [m for m in markers if m < max_len]
