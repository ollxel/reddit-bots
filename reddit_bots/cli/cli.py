"""CLI workflow for parsing, feature aggregation, and account risk analysis."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from reddit_bots.analysis.account_features import build_account_features
from reddit_bots.models.bot_classifier import AccountBotClassifier, plot_distributions
from reddit_bots.parser.reddit_parser import RedditParser, SentimentAnalyzer


class RedditAccountAnalyzerCLI:
    C = {
        "RED": "\033[91m",
        "SAND": "\033[93m",
        "DIM": "\033[90m",
        "BOLD": "\033[1m",
        "END": "\033[0m",
    }

    def __init__(self):
        load_dotenv()
        self.raw_comments_df: Optional[pd.DataFrame] = None
        self.account_features_df: Optional[pd.DataFrame] = None
        self.analysis_df: Optional[pd.DataFrame] = None

    @classmethod
    def _cx(cls, tone: str, text: str) -> str:
        return f"{cls.C.get(tone, '')}{text}{cls.C['END']}"

    @classmethod
    def _section(cls, title: str) -> None:
        line = "=" * 68
        print("\n" + cls._cx("DIM", line))
        print(cls._cx("RED", cls._cx("BOLD", title)))
        print(cls._cx("DIM", line))

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

    @staticmethod
    def _read_csv_flexible(path: str) -> pd.DataFrame:
        """
        Reads CSV with auto delimiter detection (comma/semicolon/tab).
        Protects against locale exports from Excel.
        """
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path)

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

        mode = (
            input(
                "Parse mode: 1) Subreddit (classic) 2) Post URL 3) Subreddit by date range [1]: "
            ).strip()
            or "1"
        )

        if mode == "2":
            post_url = input("Post URL: ").strip()
            parse_all = self._ask_yes_no("Parse ALL comments under this post?", default=True)
            target_comments = None if parse_all else self._safe_int("Comment count limit for this post", 500)
            self.raw_comments_df = parser.parse_post_comments(
                post_url=post_url,
                target_comments=target_comments,
                sort="new",
            )
        elif mode == "3":
            subreddit = input("Subreddit (without r/): ").strip()
            start_date = input("Start date UTC [YYYY-MM-DD]: ").strip()
            end_date = input("End date UTC [YYYY-MM-DD]: ").strip()
            comments_per_post = self._safe_int("Comments per post limit (0 = ALL)", 300)
            comments_per_post_limit = None if comments_per_post <= 0 else comments_per_post
            category = input("Listing category [new/hot/top/rising] (default new): ").strip() or "new"

            try:
                self.raw_comments_df = parser.parse_subreddit_comments_by_date_range(
                    subreddit_name=subreddit,
                    start_date=start_date,
                    end_date=end_date,
                    comments_per_post_limit=comments_per_post_limit,
                    category=category,
                    sort_comments="new",
                )
            except ValueError as exc:
                print(f"Invalid date range: {exc}")
                self.raw_comments_df = pd.DataFrame()
        else:
            target_comments = self._safe_int("Target comment count", 300)
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
            parsed_df = self._read_csv_flexible(csv_path)
            self.account_features_df = build_account_features(parsed_df)
            self.account_features_df.to_csv("account_features.csv", index=False, float_format="%.6f")
            print("Account-level features saved to 'account_features.csv'")
        else:
            self.account_features_df = build_account_features(self.raw_comments_df)
            self.account_features_df.to_csv("account_features.csv", index=False, float_format="%.6f")
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
            self.account_features_df = self._read_csv_flexible(features_path)

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
                    comments_df = self._read_csv_flexible(fallback_comments_path)

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
            self.analysis_df = self._read_csv_flexible(path)

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

    @staticmethod
    def _print_mode_intro() -> None:
        RedditAccountAnalyzerCLI._section("MODE SELECTION")
        print(RedditAccountAnalyzerCLI._cx("RED", "1) Default mode"))
        print(RedditAccountAnalyzerCLI._cx("SAND", "   Full pipeline: parse -> features -> analyzer -> report"))
        print(RedditAccountAnalyzerCLI._cx("RED", "2) Advanced mode"))
        print(RedditAccountAnalyzerCLI._cx("SAND", "   Separate steps with resume via CSV on another device"))
        print(RedditAccountAnalyzerCLI._cx("RED", "3) Exit"))
        print()
        print(RedditAccountAnalyzerCLI._cx("SAND", "Tip: copy CSV output from one machine and continue from step 2/3/4."))

    @staticmethod
    def _print_advanced_help() -> None:
        RedditAccountAnalyzerCLI._section("ADVANCED MODE HELP")
        print(RedditAccountAnalyzerCLI._cx("RED", "Step 1 output:") + " parsed_comments.csv")
        print(RedditAccountAnalyzerCLI._cx("RED", "Step 2 output:") + " account_features.csv")
        print(RedditAccountAnalyzerCLI._cx("RED", "Step 3 output:") + " account_analysis.csv")
        print(RedditAccountAnalyzerCLI._cx("SAND", "You can move these files and continue on another device."))

    def run_default_mode(self) -> None:
        self._section("DEFAULT MODE")
        print(self._cx("SAND", "Full pipeline started."))
        self._section("STAGE 1/4 - PARSE")
        self.parse_subreddit()
        if self.raw_comments_df is None or self.raw_comments_df.empty:
            print("Pipeline stopped: no parsed comments.")
            return

        self._section("STAGE 2/4 - ACCOUNT FEATURES")
        self.build_account_features()
        if self.account_features_df is None or self.account_features_df.empty:
            print("Pipeline stopped: account features were not built.")
            return

        self._section("STAGE 3/4 - ANALYZER")
        self.run_analyzer()
        if self.analysis_df is None or self.analysis_df.empty:
            print("Pipeline stopped: analyzer did not produce output.")
            return

        self._section("STAGE 4/4 - SUSPICIOUS ACCOUNTS")
        self.show_suspicious_accounts()
        self._section("DEFAULT MODE COMPLETED")

    def run_advanced_mode(self) -> None:
        self._print_advanced_help()
        while True:
            self._section("ADVANCED MODE")
            print(self._cx("RED", "1) Parse subreddit") + self._cx("SAND", " (create parsed_comments.csv)"))
            print(self._cx("RED", "2) Build account features") + self._cx("SAND", " (create account_features.csv)"))
            print(self._cx("RED", "3) Run analyzer") + self._cx("SAND", " (create account_analysis.csv)"))
            print(self._cx("RED", "4) Show suspicious accounts"))
            print(self._cx("RED", "5) Help"))
            print(self._cx("RED", "6) Back"))

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
                self._print_advanced_help()
            elif choice == "6":
                break
            else:
                print("Unknown option. Choose 1-6.")

    def run(self) -> None:
        while True:
            self._print_mode_intro()
            mode = input("Select mode: ").strip()
            if mode == "1":
                self.run_default_mode()
            elif mode == "2":
                self.run_advanced_mode()
            elif mode == "3":
                print("Bye.")
                break
            else:
                print("Unknown option. Choose 1-3.")


def run_cli() -> None:
    RedditAccountAnalyzerCLI().run()
