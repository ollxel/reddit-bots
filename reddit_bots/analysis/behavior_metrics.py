"""Behavior and stylometry helpers for account-level feature engineering."""

from __future__ import annotations

import math
import re
import string
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
PUNCTUATION_SET = set(string.punctuation)


def safe_mean(values: Iterable[float]) -> float:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.mean())


def safe_std(values: Iterable[float]) -> float:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=0))


def compute_burstiness(timestamps: pd.Series) -> float:
    """Burstiness = std(delta_t) / mean(delta_t), using seconds deltas."""
    values = pd.to_numeric(timestamps, errors="coerce").dropna().sort_values()
    if len(values) <= 2:
        return 0.0

    deltas = values.diff().dropna()
    mean_delta = float(deltas.mean())
    if mean_delta <= 0:
        return 0.0
    return float(deltas.std(ddof=0) / mean_delta)


def activity_span_days(timestamps: pd.Series) -> float:
    values = pd.to_numeric(timestamps, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    span_seconds = float(values.max() - values.min())
    return max(0.0, span_seconds / 86400.0)


def shannon_entropy(values: pd.Series) -> float:
    non_null = values.dropna().astype(str)
    if non_null.empty:
        return 0.0

    probs = non_null.value_counts(normalize=True)
    return float(-(probs * np.log2(probs)).sum())


def text_metrics(text: str) -> Tuple[float, float, float, float, int]:
    """Returns avg_word_len, avg_sentence_len, punctuation_ratio, uppercase_ratio, char_len."""
    if not isinstance(text, str) or not text:
        return 0.0, 0.0, 0.0, 0.0, 0

    words = WORD_RE.findall(text)
    word_lengths = [len(word) for word in words]
    avg_word_len = float(np.mean(word_lengths)) if word_lengths else 0.0

    sentence_chunks = [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(text) if chunk.strip()]
    if sentence_chunks:
        sentence_word_counts = [len(WORD_RE.findall(chunk)) for chunk in sentence_chunks]
        avg_sentence_len = float(np.mean(sentence_word_counts)) if sentence_word_counts else 0.0
    else:
        avg_sentence_len = float(len(words)) if words else 0.0

    char_len = len(text)
    punctuation_count = sum(1 for ch in text if ch in PUNCTUATION_SET)
    punctuation_ratio = float(punctuation_count / char_len) if char_len else 0.0

    alpha_chars = [ch for ch in text if ch.isalpha()]
    uppercase_ratio = float(sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)) if alpha_chars else 0.0

    return avg_word_len, avg_sentence_len, punctuation_ratio, uppercase_ratio, char_len


def clean_numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)

    series = pd.to_numeric(df[column], errors="coerce")
    return series.fillna(default).astype(float)


def normalize_label(value: object) -> int:
    """Best-effort conversion for labels like True/False/1/0 strings."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value > 0)
    if isinstance(value, float):
        return int(value >= 0.5)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "bot", "likely_bot"}:
        return 1
    return 0
