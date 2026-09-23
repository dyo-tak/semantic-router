"""Dataset integrity tests."""

from semantic_router.dataset import load_labeled
from semantic_router.schemas import DEFAULT_QUEUES


def test_dataset_loads():
    items = load_labeled()
    assert len(items) >= 36
    ids = [t.id for t in items]
    assert len(ids) == len(set(ids)), "duplicate ticket ids"


def test_labels_valid():
    valid_queues = {q.name for q in DEFAULT_QUEUES}
    for t in load_labeled():
        assert t.queue in valid_queues, f"{t.id}: bad queue {t.queue}"
        assert 0 <= t.urgency <= 3, f"{t.id}: bad urgency"
        assert isinstance(t.escalate, bool)


def test_dataset_balanced_enough():
    items = load_labeled()
    queues = {t.queue for t in items}
    assert queues == {q.name for q in DEFAULT_QUEUES}
    urgencies = {t.urgency for t in items}
    assert urgencies == {0, 1, 2, 3}
