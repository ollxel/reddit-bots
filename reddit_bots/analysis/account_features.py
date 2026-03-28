"""Build per-account aggregated behavior features from parsed Reddit comments."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .behavior_metrics import (
    activity_span_days,
    clean_numeric_column,
    compute_burstiness,
    normalize_label,
    shannon_entropy,
    text_metrics,
)


ACCOUNT_FEATURE_COLUMNS: List[str] = [
    "username",
    "comments_count",
    "comments_per_post",
    "avg_comment_score",
    "comment_score_std",
    "avg_reply_delay",
    "reply_delay_std",
    "sentiment_mean",
    "sentiment_std",
    "avg_comment_length",
    "comment_length_std",
    "activity_span_days",
    "posts_per_day",
    "burstiness_score",
    "avg_word_length",
    "avg_sentence_length",
    "punctuation_ratio",
    "uppercase_ratio",
    "keyword_salad_ratio",
    "unavailable_profile_ratio",
    "unique_subreddits",
    "subreddit_entropy",
]


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    if "username" not in working.columns:
        working["username"] = [f"user_{idx}" for idx in range(len(working))]

    if "comment_text" not in working.columns:
        working["comment_text"] = ""

    if "post_id" not in working.columns:
        if "comment_id" in working.columns:
            working["post_id"] = working["comment_id"].fillna("unknown_post")
        else:
            working["post_id"] = "unknown_post"

    if "subreddit" not in working.columns:
        working["subreddit"] = "unknown_subreddit"

    working["comment_score"] = clean_numeric_column(working, "comment_score", default=0.0)
    working["reply_delay_seconds"] = clean_numeric_column(working, "reply_delay_seconds", default=0.0)
    working["sentiment_score"] = clean_numeric_column(working, "sentiment_score", default=0.0)

    if "created_utc" in working.columns:
        working["_timestamp_numeric"] = clean_numeric_column(working, "created_utc", default=np.nan)
    elif "timestamp" in working.columns:
        parsed = pd.to_datetime(working["timestamp"], errors="coerce", utc=True)
        working["_timestamp_numeric"] = parsed.map(
            lambda ts: ts.timestamp() if pd.notna(ts) else np.nan
        )
    else:
        working["_timestamp_numeric"] = np.nan

    stylometry = working["comment_text"].fillna("").apply(text_metrics)
    style_df = pd.DataFrame(
        stylometry.tolist(),
        columns=[
            "_avg_word_length",
            "_avg_sentence_length",
            "_punctuation_ratio",
            "_uppercase_ratio",
            "_comment_length",
            "_word_count",
        ],
        index=working.index,
    )
    working = pd.concat([working, style_df], axis=1)

    if "is_account_unavailable" in working.columns:
        working["_account_unavailable_flag"] = (
            pd.to_numeric(working["is_account_unavailable"], errors="coerce")
            .fillna(0.0)
            .astype(int)
        )
    else:
        account_age_numeric = pd.to_numeric(working.get("account_age_days"), errors="coerce")
        user_karma_numeric = clean_numeric_column(working, "user_karma", default=0.0)
        comment_karma_numeric = clean_numeric_column(working, "comment_karma", default=0.0)
        working["_account_unavailable_flag"] = (
            account_age_numeric.isna() & (user_karma_numeric <= 0) & (comment_karma_numeric <= 0)
        ).astype(int)

    # Detect short keyword-salad comments like "çanta kuş çay yastık".
    working["_keyword_salad_flag"] = (
        working["_word_count"].between(3, 10)
        & (working["_punctuation_ratio"] <= 0.02)
        & (working["_uppercase_ratio"] <= 0.02)
        & (working["comment_score"] <= 0)
    ).astype(int)

    return working


def _aggregate_group(group: pd.DataFrame) -> pd.Series:
    comments_count = int(len(group))
    unique_posts = int(group["post_id"].nunique()) if "post_id" in group.columns else 0
    unique_posts = unique_posts if unique_posts > 0 else 1

    span_days = activity_span_days(group["_timestamp_numeric"])
    burstiness = compute_burstiness(group["_timestamp_numeric"])

    comments_per_post = float(comments_count / unique_posts)
    posts_per_day = float(comments_count / max(span_days, 1.0))

    aggregated = {
        "comments_count": comments_count,
        "comments_per_post": comments_per_post,
        "avg_comment_score": float(group["comment_score"].mean()),
        "comment_score_std": float(group["comment_score"].std(ddof=0) if comments_count > 1 else 0.0),
        "avg_reply_delay": float(group["reply_delay_seconds"].mean()),
        "reply_delay_std": float(group["reply_delay_seconds"].std(ddof=0) if comments_count > 1 else 0.0),
        "sentiment_mean": float(group["sentiment_score"].mean()),
        "sentiment_std": float(group["sentiment_score"].std(ddof=0) if comments_count > 1 else 0.0),
        "avg_comment_length": float(group["_comment_length"].mean()),
        "comment_length_std": float(group["_comment_length"].std(ddof=0) if comments_count > 1 else 0.0),
        "activity_span_days": float(span_days),
        "posts_per_day": float(posts_per_day),
        "burstiness_score": float(burstiness),
        "avg_word_length": float(group["_avg_word_length"].mean()),
        "avg_sentence_length": float(group["_avg_sentence_length"].mean()),
        "punctuation_ratio": float(group["_punctuation_ratio"].mean()),
        "uppercase_ratio": float(group["_uppercase_ratio"].mean()),
        "keyword_salad_ratio": float(group["_keyword_salad_flag"].mean()),
        "unavailable_profile_ratio": float(group["_account_unavailable_flag"].mean()),
        "unique_subreddits": int(group["subreddit"].nunique()) if "subreddit" in group.columns else 0,
        "subreddit_entropy": float(shannon_entropy(group["subreddit"])) if "subreddit" in group.columns else 0.0,
    }

    for field in ["account_age_days", "user_karma", "comment_karma"]:
        if field in group.columns:
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            aggregated[field] = float(values.mean()) if not values.empty else 0.0

    if "is_bot_flag" in group.columns:
        labels = group["is_bot_flag"].apply(normalize_label)
        aggregated["is_bot_flag"] = int(labels.mean() >= 0.5)

    return pd.Series(aggregated)


def build_account_features(comments_df: pd.DataFrame) -> pd.DataFrame:
    """Group parsed comment data by username and compute behavior profile features."""
    if comments_df is None or comments_df.empty:
        return pd.DataFrame(columns=ACCOUNT_FEATURE_COLUMNS)

    working = _ensure_required_columns(comments_df)

    grouped = (
        working.groupby("username", dropna=False)
        .apply(_aggregate_group)
        .reset_index()
    )

    for column in ACCOUNT_FEATURE_COLUMNS:
        if column not in grouped.columns:
            grouped[column] = 0.0

    numeric_cols = [col for col in grouped.columns if col != "username"]
    for col in numeric_cols:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0)

    grouped = grouped.sort_values(by="comments_count", ascending=False).reset_index(drop=True)
    return grouped


def build_and_save_account_features(
    comments_csv_path: str,
    output_csv_path: str = "account_features.csv",
) -> pd.DataFrame:
    comments_df = pd.read_csv(comments_csv_path)
    account_df = build_account_features(comments_df)
    account_df.to_csv(output_csv_path, index=False, float_format="%.6f")
    print(f"Account-level features saved to '{output_csv_path}'")
    return account_df


def load_account_features(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None
