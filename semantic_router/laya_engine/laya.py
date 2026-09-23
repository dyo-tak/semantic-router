"""Laya ONNX engine — Python port of github.com/receptron/laya src/laya.ts on top of onnxruntime.

`LayaEngine.system_one()` answers any number of typed questions about one state in a
single forward pass: same sequence layout, same per-cardinality temperature, same rounding
as the TypeScript / Python reference implementations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from .sequence import (
    QTYPES,
    SpecialIds,
    build_sequence,
    confidence_from_probs,
    render_options,
    softmax,
    temp_bucket,
    to_internal,
)

ROUND4 = lambda x: round(x, 4)  # noqa: E731


class LayaEngine:
    def __init__(self, model_dir: str | Path, intra_op_threads: int = 4):
        self.model_dir = Path(model_dir)
        cfg_path = self.model_dir / "laya_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"no laya_config.json under {self.model_dir} — download the ONNX bundle first "
                f"(python -m semantic_router.download)"
            )
        self.config: dict[str, Any] = json.loads(cfg_path.read_text())
        tok = Tokenizer.from_file(str(self.model_dir / "tokenizer" / "tokenizer.json"))
        tok.no_padding()
        self._tok = tok
        tok_cfg = json.loads((self.model_dir / "tokenizer" / "tokenizer_config.json").read_text())

        def token_id(t: str) -> int:
            v = tok.token_to_id(t)
            if v is None:
                raise ValueError(f"special token {t} missing from tokenizer")
            return v

        self.ids: SpecialIds = {
            "cls": token_id("[CLS]"),
            "sep": token_id("[SEP]"),
            "mask": token_id("[MASK]"),
            "pad": token_id(tok_cfg.get("pad_token", "[PAD]")),
            "mask_tok": "[MASK]",
        }

        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_dir / "laya.onnx"), so, providers=["CPUExecutionProvider"])
        self._encode = lambda text: tok.encode(text, add_special_tokens=False).ids

    # ------------------------------------------------------------------ #
    def system_one(self, state: Any, questions: dict[str, dict]) -> dict[str, Any]:
        """Answer every question about `state` in one forward pass (Jev system_one shape)."""
        if not questions:
            raise ValueError("system_one: at least one question is required")
        qids = list(questions.keys())
        items = []
        for qid in qids:
            q = to_internal(questions[qid])
            ids, markers = build_sequence(
                self._encode, self.ids, state, q, self.config["max_len"], self.config["head_max_len"]
            )
            if len(markers) != len(render_options(q)):
                raise ValueError(f"question {qid!r}: options do not fit in head_max_len={self.config['head_max_len']} tokens")
            items.append({"q": q, "ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})

        # rl_common.collate_items: right-pad to longest sequence / widest option set in batch
        n = len(items)
        L = max(len(it["ids"]) for it in items)
        K = max(len(it["markers"]) for it in items)
        input_ids = np.full((n, L), self.ids["pad"], dtype=np.int64)
        attention = np.zeros((n, L), dtype=np.int64)
        marker_pos = np.zeros((n, K), dtype=np.int64)
        marker_mask = np.zeros((n, K), dtype=bool)
        qtype = np.zeros((n,), dtype=np.int64)
        n_tokens = 0
        for i, it in enumerate(items):
            seq = it["ids"]
            input_ids[i, : len(seq)] = seq
            attention[i, : len(seq)] = 1
            n_tokens += len(seq)
            for j, m in enumerate(it["markers"]):
                marker_pos[i, j] = m
                marker_mask[i, j] = True
            qtype[i] = it["qtype"]

        t0 = time.perf_counter()
        out = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention,
                "marker_pos": marker_pos,
                "marker_mask": marker_mask,
                "qtype": qtype,
            },
        )
        infer_ms = (time.perf_counter() - t0) * 1000

        logits = np.asarray(out[0], dtype=np.float32)  # [n, K], masked slots = -1e4
        act_probs = np.asarray(out[1], dtype=np.float32)  # [n, 2]

        answers: dict[str, dict] = {}
        for r, it in enumerate(items):
            qid = qids[r]
            q = it["q"]
            k = len(it["markers"])
            temp = self.config.get("temperature_by_options", {}).get(
                temp_bucket(it["qtype"], k), self.config["temperature"][it["qtype"]] if it["qtype"] < len(self.config["temperature"]) else 1.0
            )
            p = softmax([float(v) / temp for v in logits[r, :k]])
            ext = {"act_probability": round(float(act_probs[r, 0]), 4)}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                best = max(range(k), key=lambda i: p[i])
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[best],
                    "probabilities": {kk: ROUND4(p[i]) for i, kk in enumerate(keys)},
                    "confidence": ROUND4(confidence_from_probs(p)),
                    "rl_agent": ext,
                }
            elif q["t"] == "score":
                crit = q["crit"]
                answers[qid] = {
                    "type": "score",
                    "score": ROUND4(sum(i * v for i, v in enumerate(p))),
                    "legend": {str(i): c for i, c in enumerate(crit)},
                    "probabilities": {str(i): ROUND4(v) for i, v in enumerate(p)},
                    "confidence": ROUND4(confidence_from_probs(p)),
                    "rl_agent": ext,
                }
            else:
                answers[qid] = {"type": "noul", "noul": ROUND4(p[1]), "rl_agent": ext}

        return {"model": "laya", "answers": answers, "usage": {"input_tokens": n_tokens, "output_tokens": 0}, "latency_ms": round(infer_ms, 1)}
