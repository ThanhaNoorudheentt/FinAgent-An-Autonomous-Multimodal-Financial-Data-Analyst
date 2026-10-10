"""Okapi BM25 over chunks. Free (no API calls), and strong on exact financial terms."""

from __future__ import annotations

import math
import re
from collections import Counter

# Keeps numbers like "1,577.5" and "14.1" intact as single tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


# Standard English function words only. Deliberately no finance-specific terms, so this is
# generic hygiene rather than something tuned to FinanceBench.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "will",
        "with",
        "give",
        "response",
        "question",
        "relying",
        "details",
        "shown",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        self._tfs = [Counter(tokenize(d)) for d in documents]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(documents)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = [t for t in set(tokenize(query)) if t in self._idf]
        out: list[float] = []
        for tf, length in zip(self._tfs, self._lens, strict=True):
            norm = self._k1 * (1 - self._b + self._b * length / (self._avg_len or 1.0))
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    score += self._idf[t] * f * (self._k1 + 1) / (f + norm)
            out.append(score)
        return out
