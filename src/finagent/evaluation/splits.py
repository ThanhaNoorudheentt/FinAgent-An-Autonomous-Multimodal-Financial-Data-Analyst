"""Deterministic dev/test split, so design decisions use dev and final numbers use test."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import Literal, TypeVar

E = TypeVar("E")
Split = Literal["all", "dev", "test"]
DEV_FRACTION_TENTHS = 3  # 30% dev, 70% test


def split_examples(examples: Iterable[E], split: Split, *, key: Callable[[E], str]) -> list[E]:
    """Hash each example's id into dev (30%) or test (70%); stable across runs and machines."""
    items = list(examples)
    if split == "all":
        return items

    def is_dev(example: E) -> bool:
        return int(hashlib.sha256(key(example).encode()).hexdigest(), 16) % 10 < DEV_FRACTION_TENTHS

    return [ex for ex in items if is_dev(ex) == (split == "dev")]
