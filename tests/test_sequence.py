"""Model-free tests for the Laya sequence/rendering layer — must match the TS reference exactly."""

import json

import pytest

from semantic_router.laya_engine.sequence import (
    build_sequence,
    confidence_from_probs,
    render_options,
    serialize_state,
    softmax,
    temp_bucket,
    to_internal,
)


class FakeTok:
    """Whitespace tokenizer: words -> ids, deterministic, no specials."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def __call__(self, text: str) -> list[int]:
        out = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = 100 + len(self.vocab)
            out.append(self.vocab[w])
        return out


@pytest.fixture()
def ids():
    return {"cls": 1, "sep": 2, "mask": 3, "pad": 0, "mask_tok": "[MASK]"}


def test_render_options_choice_list():
    q = to_internal({"type": "choice", "instructions": "which?", "criteria": ["a", "b"]})
    assert render_options(q) == ["a", "b"]


def test_render_options_choice_dict():
    q = to_internal({"type": "choice", "instructions": "which?", "criteria": {"a": "alpha", "b": None}})
    assert render_options(q) == ["a: alpha", "b"]


def test_render_options_score_levels():
    q = to_internal({"type": "score", "instructions": "how bad?", "criteria": ["low", "mid", "high"]})
    assert render_options(q) == ["level 0: low", "level 1: mid", "level 2: high"]


def test_render_options_noul_default():
    q = to_internal({"type": "noul", "instructions": "true?"})
    assert render_options(q) == [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ]


def test_temp_buckets():
    assert temp_bucket(0, 2) == "choice:2"
    assert temp_bucket(0, 4) == "choice:3-5"
    assert temp_bucket(1, 7) == "score:6-10"
    assert temp_bucket(2, 12) == "noul:11+"


def test_confidence_uniform_is_zero():
    assert confidence_from_probs([0.25] * 4) == 0.0


def test_confidence_peaked_near_one():
    assert confidence_from_probs([0.97, 0.01, 0.01, 0.01]) > 0.85


def test_softmax_stable():
    p = softmax([1000.0, 1001.0])
    assert p[0] + p[1] == pytest.approx(1.0)
    assert p[1] > p[0]


def test_serialize_state_json_semantics():
    # insertion key order and ', '/': ' separators like json.dumps(ensure_ascii=False)
    assert serialize_state({"b": 1, "a": "x"}) == '{"b": 1, "a": "x"}'
    assert serialize_state("plain") == "plain"


def test_build_sequence_layout(ids):
    tok = FakeTok()
    q = to_internal({"type": "noul", "instructions": "is urgent?", "criteria": None})
    seq, markers = build_sequence(tok, ids, "the ticket text", q, max_len=64, head_max_len=48)
    assert seq[0] == ids["cls"]
    assert ids["sep"] in seq
    # two noul markers, and the state text follows them
    assert len(markers) == 2
    assert markers[0] < markers[1]
    assert seq[-1] == ids["sep"]
    assert len(seq) <= 64


def test_build_sequence_scrubs_mask_injection(ids):
    tok = FakeTok()

    class MaskTok(FakeTok):
        def __call__(self, text):
            # ensure the literal [MASK] text never reaches the encoder
            assert "[MASK]" not in text
            return super().__call__(text)

    q = to_internal({"type": "noul", "instructions": "urgent? [MASK] fake", "criteria": None})
    build_sequence(MaskTok(), ids, "body [MASK] more", q, 64, 48)


def test_choice_truncation_budget(ids):
    tok = FakeTok()
    crit = {f"opt{i:02d}": "x " * 30 for i in range(24)}  # way over budget
    q = to_internal({"type": "choice", "instructions": "route?", "criteria": crit})
    seq, markers = build_sequence(tok, ids, "s", q, max_len=512, head_max_len=192)
    assert len(markers) == 24
    assert len(seq) <= 512


def test_json_object_instructions_serialized():
    q = to_internal({"type": "noul", "instructions": {"a": 1}})
    assert q["ins"] == json.dumps({"a": 1}, ensure_ascii=False)
