"""Account-level bot risk model and reporting utilities."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from reddit_bots.analysis.account_features import build_account_features
from reddit_bots.analysis.behavior_metrics import normalize_label


ACCOUNT_FEATURES: List[str] = [
    "comments_count",
    "avg_reply_delay",
    "reply_delay_std",
    "avg_comment_score",
    "comment_score_std",
    "avg_comment_length",
    "sentiment_mean",
    "sentiment_std",
    "activity_span_days",
    "posts_per_day",
    "burstiness_score",
]


class AccountBotClassifier:
    """Train and run bot-risk scoring on account-level features."""

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[RandomForestClassifier] = None

    def _ensure_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        fixed = df.copy()
        for column in ACCOUNT_FEATURES:
            if column not in fixed.columns:
                fixed[column] = 0.0
        return fixed

    def prepare_training_data(self, training_csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        data = pd.read_csv(training_csv_path)

        if "is_bot_flag" not in data.columns:
            raise ValueError("Training file must contain 'is_bot_flag' column.")

        if "username" not in data.columns:
            data["username"] = [f"train_user_{idx}" for idx in range(len(data))]

        if not set(ACCOUNT_FEATURES).issubset(data.columns):
            account_df = build_account_features(data)
        else:
            account_df = data.copy()

        if "is_bot_flag" not in account_df.columns:
            merged = data[["username", "is_bot_flag"]].copy()
            merged["is_bot_flag"] = merged["is_bot_flag"].apply(normalize_label)
            merged = (
                merged.groupby("username", as_index=False)["is_bot_flag"]
                .mean()
                .assign(is_bot_flag=lambda df_: (df_["is_bot_flag"] >= 0.5).astype(int))
            )
            account_df = account_df.merge(merged, on="username", how="left")

        account_df["is_bot_flag"] = account_df["is_bot_flag"].apply(normalize_label)
        account_df = self._ensure_feature_columns(account_df)

        X = account_df[ACCOUNT_FEATURES].fillna(0.0)
        y = account_df["is_bot_flag"].astype(int)
        return X, y

    def train(self, training_csv_path: str) -> RandomForestClassifier:
        X, y = self.prepare_training_data(training_csv_path)

        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=stratify,
        )

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print("\n" + "=" * 60)
        print("ACCOUNT MODEL TRAINING REPORT")
        print("=" * 60)
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        importances = pd.Series(self.model.feature_importances_, index=ACCOUNT_FEATURES)
        print("\nFeature importances:")
        for feature, value in importances.sort_values(ascending=False).items():
            print(f"  {feature:<22} {value:.4f}")

        return self.model

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability < 0.3:
            return "likely human"
        if probability < 0.6:
            return "suspicious"
        return "likely bot"

    @staticmethod
    def _heuristic_bot_probability(row: pd.Series) -> float:
        """
        Rule-based boost for CLEAR bot patterns only.
        Avoid false positives from normal low-activity users.
        """
        score = 0.0
        unavailable_ratio = float(row.get("unavailable_profile_ratio", 0.0) or 0.0)
        keyword_salad_ratio = float(row.get("keyword_salad_ratio", 0.0) or 0.0)
        avg_comment_score = float(row.get("avg_comment_score", 0.0) or 0.0)
        comments_count = float(row.get("comments_count", 0.0) or 0.0)
        avg_sentence_length = float(row.get("avg_sentence_length", 0.0) or 0.0)
        punctuation_ratio = float(row.get("punctuation_ratio", 0.0) or 0.0)
        uppercase_ratio = float(row.get("uppercase_ratio", 0.0) or 0.0)
        avg_comment_length = float(row.get("avg_comment_length", 0.0) or 0.0)
        sentiment_std = float(row.get("sentiment_std", 0.0) or 0.0)
        sentiment_mean = float(row.get("sentiment_mean", 0.0) or 0.0)
        activity_span_days = float(row.get("activity_span_days", 0.0) or 0.0)
        account_age_days = float(row.get("account_age_days", 0.0) or 0.0)
        user_karma = float(row.get("user_karma", 0.0) or 0.0)
        comment_karma = float(row.get("comment_karma", 0.0) or 0.0)

        # PATTERN 1: Keyword salad spam (VERY CLEAR)
        if keyword_salad_ratio >= 0.5 and avg_comment_score <= 0:
            score = max(score, 0.7)
        elif keyword_salad_ratio >= 0.8:
            score = max(score, 0.8)

        # PATTERN 2: Suspended/unavailable profiles
        # Only if combined with other spam signals
        if unavailable_ratio >= 0.9 and keyword_salad_ratio >= 0.5:
            score = max(score, 0.75)
        elif unavailable_ratio >= 0.9 and avg_comment_score <= -0.5:
            score = max(score, 0.60)

        # PATTERN 3: Very short, low-quality spam comments
        if (
            comments_count <= 3
            and avg_comment_score <= 0
            and avg_comment_length <= 20  # Extremely short
            and punctuation_ratio <= 0.02
            and uppercase_ratio <= 0.05
        ):
            score = max(score, 0.65)

        # PATTERN 4: Young account + low karma + rapid posting
        # These are classic bot creation patterns
        if (
            account_age_days > 0  # Account data exists
            and account_age_days < 90  # Very new account
            and user_karma <= 1  # Essentially no karma
            and comment_karma <= 0
            and comments_count >= 2
            and activity_span_days < 0.01  # Tight activity window
        ):
            score = max(score, 0.70)

        return float(min(score, 0.99))

    def predict(self, account_features_df: pd.DataFrame) -> pd.DataFrame:
        data = self._ensure_feature_columns(account_features_df)
        X = data[ACCOUNT_FEATURES].fillna(0.0)

        # Use model predictions if trained, otherwise use zero
        if self.model is not None:
            probabilities = self.model.predict_proba(X)
            model_probability = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        else:
            model_probability = np.zeros(len(data))
        
        heuristic_probability = data.apply(self._heuristic_bot_probability, axis=1).to_numpy(dtype=float)
        bot_probability = np.maximum(model_probability, heuristic_probability)

        result = data.copy()
        result["model_probability"] = model_probability
        result["heuristic_probability"] = heuristic_probability
        result["bot_probability"] = bot_probability
        result["risk_level"] = result["bot_probability"].apply(self._risk_level)

        output_cols = ["username", "comments_count", "bot_probability", "risk_level"]
        for column in output_cols:
            if column not in result.columns:
                result[column] = 0

        ordered = result.sort_values("bot_probability", ascending=False).reset_index(drop=True)
        return ordered

    def run_analysis(
        self,
        account_features_df: pd.DataFrame,
        output_csv_path: str = "account_analysis.csv",
    ) -> pd.DataFrame:
        result = self.predict(account_features_df)
        result.to_csv(output_csv_path, index=False, float_format="%.6f")
        print(f"Account analysis saved to '{output_csv_path}'")
        return result

    @staticmethod
    def show_suspicious_accounts(
        analysis_df: pd.DataFrame,
        min_probability: float = 0.3,
        top_n: int = 20,
    ) -> pd.DataFrame:
        filtered = analysis_df[analysis_df["bot_probability"] >= min_probability].copy()
        filtered = filtered.sort_values("bot_probability", ascending=False).head(top_n)
        return filtered


def plot_distributions(
    comments_df: Optional[pd.DataFrame],
    account_features_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    output_dir: str = ".",
) -> List[str]:
    """Generate optional histograms and return saved image paths."""
    os.makedirs(output_dir, exist_ok=True)
    generated_paths: List[str] = []

    if "comments_count" in account_features_df.columns:
        plt.figure(figsize=(8, 5))
        account_features_df["comments_count"].dropna().hist(bins=30)
        plt.title("Distribution of Comment Counts")
        plt.xlabel("comments_count")
        plt.ylabel("accounts")
        path = os.path.join(output_dir, "comment_count_distribution.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        generated_paths.append(path)

    if comments_df is not None and "reply_delay_seconds" in comments_df.columns:
        plt.figure(figsize=(8, 5))
        comments_df["reply_delay_seconds"].dropna().clip(upper=86400).hist(bins=40)
        plt.title("Reply Delay Histogram")
        plt.xlabel("reply_delay_seconds (capped at 86400)")
        plt.ylabel("comments")
        path = os.path.join(output_dir, "reply_delay_histogram.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        generated_paths.append(path)

    if comments_df is not None and "sentiment_score" in comments_df.columns:
        plt.figure(figsize=(8, 5))
        comments_df["sentiment_score"].dropna().hist(bins=30, range=(-1, 1))
        plt.title("Sentiment Distribution")
        plt.xlabel("sentiment_score")
        plt.ylabel("comments")
        path = os.path.join(output_dir, "sentiment_distribution.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        generated_paths.append(path)

    if "bot_probability" in analysis_df.columns:
        plt.figure(figsize=(8, 5))
        analysis_df["bot_probability"].dropna().hist(bins=30, range=(0, 1))
        plt.title("Bot Probability Histogram")
        plt.xlabel("bot_probability")
        plt.ylabel("accounts")
        path = os.path.join(output_dir, "bot_probability_histogram.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        generated_paths.append(path)

    return generated_paths
