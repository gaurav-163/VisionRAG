"""
Object normalization helpers for detector and VLM outputs.
"""

from __future__ import annotations

from collections import Counter

HUMAN_LABELS: set[str] = {
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
    "human",
    "worker",
    "employee",
    "staff",
    "customer",
    "pedestrian",
    "rider",
    "driver",
}


def normalize_object_counts(counts: dict) -> dict:
    """Normalize object counts and collapse human labels into `person`."""
    normalized = Counter()

    for label, value in (counts or {}).items():
        if label is None:
            continue

        label_text = str(label).strip().lower()
        if not label_text:
            continue

        normalized_label = "person" if label_text in HUMAN_LABELS else label_text

        try:
            normalized[normalized_label] += int(value)
        except (TypeError, ValueError):
            continue

    return dict(normalized)
