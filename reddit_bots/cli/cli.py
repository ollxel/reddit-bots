"""CLI workflow for parsing, feature aggregation, and account risk analysis."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from reddit_bots.analysis.account_features import build_account_features, build_and_save_account_features
from reddit_bots.models.bot_classifier import AccountBotClassifier, plot_distributions
from reddit_bots.parser.reddit_parser import RedditParser, SentimentAnalyzer


class RedditAccountAnalyzerCLI:
    def __init__(self):
        load_dotenv()
        self.raw_comments_df: Optional[pd.DataFrame] = None
        self.account_features_df: Optional[pd.DataFrame] = None
        self.analysis_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _ask_yes_no(prompt: str, default: bool = False) -> bool:
        default_hint = "Y/n" if default else "y/N"
        value = input(f"{prompt} ({default_hint}): ").strip().lower()
        if not value:
            return default
        return value in {"y", "yes", "д", "да"}

    @staticmethod
    def _safe_int(prompt: str, default: int) -> int:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"Invalid integer, using default {default}.")
            return default

    def _configure_sentiment(self, parser: RedditParser) -> None:
        env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        api_key = env_key

        if self._ask_yes_no("Enable OpenRouter sentiment analysis?", default=bool(env_key)):
            if env_key:
                use_env = self._ask_yes_no("Use OPENROUTER_API_KEY from .env?", default=True)
                if not use_env:
                    api_key = input("Enter OpenRouter API key: ").strip()
            else:
                api_key = input("Enter OpenRouter API key: ").strip()

            model = input(
                "OpenRouter model [arcee-ai/trinity-large-preview:free]: "
            ).strip() or "arcee-ai/trinity-large-preview:free"

            if api_key:
                parser.sentiment = SentimentAnalyzer(api_key=api_key, model=model)
                print(f"SentimentAnalyzer enabled with model: {model}")
            else:
                print("No API key provided. Sentiment analysis disabled.")
        else:
            parser.run_sentiment = False

    def parse_subreddit(self) -> None:
        parser = RedditParser(
            user_agent="RedditDataCollector/2.0 (Educational)",
            run_sentiment=True,
            unique_users_only=False,
        )
        self._configure_sentiment(parser)

        mode = input("Parse mode: 1) Subreddit 2) Post URL [1]: ").strip() or "1"
        target_comments = self._safe_int("Target comment count", 300)

        if mode == "2":
            post_url = input("Post URL: ").strip()
            self.raw_comments_df = parser.parse_post_comments(
                post_url=post_url,
                target_comments=target_comments,
            )
        else:
            subreddit = input("Subreddit (without r/): ").strip()
            category = input("Category [hot/new/top/rising] (default hot): ").strip() or "hot"
            time_filter = "week"
            if category == "top":
                time_filter = (
                    input("Top time filter [hour/day/week/month/year/all] (default week): ").strip()
                    or "week"
                )
            posts_limit = self._safe_int("Posts limit", 10)

            self.raw_comments_df = parser.parse_subreddit_comments(
                subreddit_name=subreddit,
                posts_limit=posts_limit,
                category=category,
                time_filter=time_filter,
                target_comments=target_comments,
            )

        if self.raw_comments_df is None or self.raw_comments_df.empty:
            print("No comments collected.")
            return

        raw_output = "parsed_comments.csv"
        self.raw_comments_df.to_csv(raw_output, index=False)
        print(f"Raw parsed comments saved to '{raw_output}'")
        print(f"Collected comments: {len(self.raw_comments_df)}")
        print(f"Unique users: {self.raw_comments_df['username'].nunique()}")

    def build_account_features(self) -> None:
        if self.raw_comments_df is None or self.raw_comments_df.empty:
            csv_path = input("Path to parsed comments CSV [parsed_comments.csv]: ").strip() or "parsed_comments.csv"
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
                return
            self.account_features_df = build_and_save_account_features(csv_path, "account_features.csv")
        else:
            self.account_features_df = build_account_features(self.raw_comments_df)
            self.account_features_df.to_csv("account_features.csv", index=False)
            print("Account-level features saved to 'account_features.csv'")

        print(f"Accounts profiled: {len(self.account_features_df)}")

    def run_analyzer(self) -> None:
        training_csv = (
            input("Training CSV path [reddit_dead_internet_analysis_2026.csv]: ").strip()
            or "reddit_dead_internet_analysis_2026.csv"
        )
        if not os.path.exists(training_csv):
            print(f"Training file not found: {training_csv}")
            return

        if self.account_features_df is None or self.account_features_df.empty:
            features_path = input("Account features CSV path [account_features.csv]: ").strip() or "account_features.csv"
            if not os.path.exists(features_path):
                print(f"Account features file not found: {features_path}")
                return
            self.account_features_df = pd.read_csv(features_path)

        classifier = AccountBotClassifier()
        classifier.train(training_csv)
        self.analysis_df = classifier.run_analysis(self.account_features_df, "account_analysis.csv")

        print("\nTop suspicious accounts:")
        suspicious = classifier.show_suspicious_accounts(self.analysis_df, min_probability=0.3, top_n=10)
        if suspicious.empty:
            print("No suspicious accounts detected with threshold 0.3")
        else:
            print(suspicious[["username", "comments_count", "bot_probability", "risk_level"]].to_string(index=False))

        if self._ask_yes_no("Generate plots?", default=True):
            comments_df = self.raw_comments_df
            if comments_df is None or comments_df.empty:
                fallback_comments_path = input(
                    "Parsed comments CSV for plots [parsed_comments.csv]: "
                ).strip() or "parsed_comments.csv"
                if os.path.exists(fallback_comments_path):
                    comments_df = pd.read_csv(fallback_comments_path)

            paths = plot_distributions(
                comments_df=comments_df,
                account_features_df=self.account_features_df,
                analysis_df=self.analysis_df,
                output_dir=".",
            )
            if paths:
                print("Generated plot files:")
                for path in paths:
                    print(f"  - {path}")

    def show_suspicious_accounts(self) -> None:
        if self.analysis_df is None or self.analysis_df.empty:
            path = input("Analysis CSV path [account_analysis.csv]: ").strip() or "account_analysis.csv"
            if not os.path.exists(path):
                print(f"File not found: {path}")
                return
            self.analysis_df = pd.read_csv(path)

        threshold_raw = input("Minimum bot probability threshold [0.3]: ").strip() or "0.3"
        try:
            threshold = float(threshold_raw)
        except ValueError:
            threshold = 0.3

        filtered = self.analysis_df[self.analysis_df["bot_probability"] >= threshold]
        filtered = filtered.sort_values("bot_probability", ascending=False)

        if filtered.empty:
            print("No accounts above threshold.")
            return

        print("\nSuspicious accounts:")
        print(filtered[["username", "comments_count", "bot_probability", "risk_level"]].head(50).to_string(index=False))

    def run(self) -> None:
        while True:
            print("\n" + "=" * 60)
            print("REDDIT ACCOUNT ANALYZER")
            print("=" * 60)
            print("1) Parse subreddit")
            print("2) Build account features")
            print("3) Run analyzer")
            print("4) Show suspicious accounts")
            print("5) Exit")

            choice = input("Select option: ").strip()

            if choice == "1":
                self.parse_subreddit()
            elif choice == "2":
                self.build_account_features()
            elif choice == "3":
                self.run_analyzer()
            elif choice == "4":
                self.show_suspicious_accounts()
            elif choice == "5":
                print("Bye.")
                break
            else:
                print("Unknown option. Choose 1-5.")


def run_cli() -> None:
    RedditAccountAnalyzerCLI().run()
