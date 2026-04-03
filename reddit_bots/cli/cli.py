"""CLI workflow for parsing, feature aggregation, and account risk analysis."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from reddit_bots.analysis.account_features import build_account_features
from reddit_bots.models.bot_classifier import AccountBotClassifier, plot_distributions
from reddit_bots.parser.reddit_parser import RedditParser, SentimentAnalyzer
from reddit_bots.web.web_mode import run_web_interface


class RedditAccountAnalyzerCLI:
    I18N = {
        "en": {
            "mode_selection": "MODE SELECTION",
            "default_mode": "1) Default mode",
            "default_desc": "   Full pipeline: parse -> features -> analyzer -> report",
            "advanced_mode": "2) Advanced mode",
            "advanced_desc": "   Separate steps with resume via CSV on another device",
            "web_mode": "3) Web mode",
            "web_desc": "   Localhost terminal-style web interface",
            "exit": "4) Exit",
            "mode_prompt": "Select mode: ",
            "unknown_mode": "Unknown option. Choose 1-4.",
            "bye": "Bye.",
            "web_title": "WEB MODE",
            "host_prompt": "Host [127.0.0.1]: ",
            "port_prompt": "Port",
            "launch_web": "Launching local web server...",
        },
        "ru": {
            "mode_selection": "ВЫБОР РЕЖИМА",
            "default_mode": "1) Режим по умолчанию",
            "default_desc": "   Полный пайплайн: парсинг -> фичи -> анализатор -> отчет",
            "advanced_mode": "2) Продвинутый режим",
            "advanced_desc": "   Пошаговый режим с продолжением через CSV на другом устройстве",
            "web_mode": "3) Веб режим",
            "web_desc": "   Локальный веб-интерфейс в терминальном стиле",
            "exit": "4) Выход",
            "mode_prompt": "Выберите режим: ",
            "unknown_mode": "Неизвестная опция. Выберите 1-4.",
            "bye": "Выход.",
            "web_title": "ВЕБ РЕЖИМ",
            "host_prompt": "Хост [127.0.0.1]: ",
            "port_prompt": "Порт",
            "launch_web": "Запускаю локальный веб-сервер...",
        },
    }
    C = {
        "RED": "\033[91m",
        "SAND": "\033[93m",
        "DIM": "\033[90m",
        "BOLD": "\033[1m",
        "END": "\033[0m",
    }

    def __init__(self, language: str = "en"):
        load_dotenv()
        self.language = language if language in self.I18N else "en"
        self.raw_comments_df: Optional[pd.DataFrame] = None
        self.account_features_df: Optional[pd.DataFrame] = None
        self.analysis_df: Optional[pd.DataFrame] = None

    def _tr(self, key: str) -> str:
        return self.I18N[self.language].get(key, key)

    def _txt(self, en: str, ru: str) -> str:
        return ru if self.language == "ru" else en

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

        if self._ask_yes_no(
            self._txt("Enable OpenRouter sentiment analysis?", "Включить OpenRouter sentiment анализ?"),
            default=bool(env_key),
        ):
            if env_key:
                use_env = self._ask_yes_no(
                    self._txt("Use OPENROUTER_API_KEY from .env?", "Использовать OPENROUTER_API_KEY из .env?"),
                    default=True,
                )
                if not use_env:
                    api_key = input(
                        self._txt("Enter OpenRouter API key: ", "Введите OpenRouter API ключ: ")
                    ).strip()
            else:
                api_key = input(self._txt("Enter OpenRouter API key: ", "Введите OpenRouter API ключ: ")).strip()

            model = input(
                self._txt(
                    "OpenRouter model [arcee-ai/trinity-large-preview:free]: ",
                    "Модель OpenRouter [arcee-ai/trinity-large-preview:free]: ",
                )
            ).strip() or "arcee-ai/trinity-large-preview:free"

            if api_key:
                parser.sentiment = SentimentAnalyzer(api_key=api_key, model=model)
                print(
                    self._txt(
                        f"SentimentAnalyzer enabled with model: {model}",
                        f"SentimentAnalyzer включен с моделью: {model}",
                    )
                )
            else:
                print(self._txt("No API key provided. Sentiment analysis disabled.", "API ключ не указан. Sentiment анализ отключен."))
        else:
            parser.run_sentiment = False

    def parse_subreddit(self) -> None:
        parser = RedditParser(
            user_agent="RedditDataCollector/2.0 (Educational)",
            run_sentiment=True,
            unique_users_only=False,
            language=self.language,
        )
        self._configure_sentiment(parser)

        mode = (
            input(
                self._txt(
                    "Parse mode: 1) Subreddit (classic) 2) Post URL 3) Subreddit by date range [1]: ",
                    "Режим парсинга: 1) Сабреддит (классика) 2) Ссылка на пост 3) Сабреддит по диапазону дат [1]: ",
                )
            ).strip()
            or "1"
        )

        if mode == "2":
            post_url = input(self._txt("Post URL: ", "Ссылка на пост: ")).strip()
            parse_all = self._ask_yes_no(
                self._txt("Parse ALL comments under this post?", "Парсить ВСЕ комментарии под постом?"),
                default=True,
            )
            target_comments = None if parse_all else self._safe_int(
                self._txt("Comment count limit for this post", "Лимит комментариев для этого поста"),
                500,
            )
            self.raw_comments_df = parser.parse_post_comments(
                post_url=post_url,
                target_comments=target_comments,
                sort="new",
            )
        elif mode == "3":
            subreddit = input(self._txt("Subreddit (without r/): ", "Сабреддит (без r/): ")).strip()
            start_date = input(self._txt("Start date UTC [YYYY-MM-DD]: ", "Дата начала UTC [YYYY-MM-DD]: ")).strip()
            end_date = input(self._txt("End date UTC [YYYY-MM-DD]: ", "Дата конца UTC [YYYY-MM-DD]: ")).strip()
            comments_per_post = self._safe_int(
                self._txt("Comments per post limit (0 = ALL)", "Лимит комментариев на пост (0 = ВСЕ)"),
                300,
            )
            comments_per_post_limit = None if comments_per_post <= 0 else comments_per_post
            category = input(
                self._txt(
                    "Listing category [new/hot/top/rising] (default new): ",
                    "Категория листинга [new/hot/top/rising] (по умолчанию new): ",
                )
            ).strip() or "new"

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
                print(self._txt(f"Invalid date range: {exc}", f"Некорректный диапазон дат: {exc}"))
                self.raw_comments_df = pd.DataFrame()
        else:
            target_comments = self._safe_int(self._txt("Target comment count", "Целевое число комментариев"), 300)
            subreddit = input(self._txt("Subreddit (without r/): ", "Сабреддит (без r/): ")).strip()
            category = input(
                self._txt(
                    "Category [hot/new/top/rising] (default hot): ",
                    "Категория [hot/new/top/rising] (по умолчанию hot): ",
                )
            ).strip() or "hot"
            time_filter = "week"
            if category == "top":
                time_filter = (
                    input(
                        self._txt(
                            "Top time filter [hour/day/week/month/year/all] (default week): ",
                            "Фильтр времени для top [hour/day/week/month/year/all] (по умолчанию week): ",
                        )
                    ).strip()
                    or "week"
                )
            posts_limit = self._safe_int(self._txt("Posts limit", "Лимит постов"), 10)

            self.raw_comments_df = parser.parse_subreddit_comments(
                subreddit_name=subreddit,
                posts_limit=posts_limit,
                category=category,
                time_filter=time_filter,
                target_comments=target_comments,
            )

        if self.raw_comments_df is None or self.raw_comments_df.empty:
            print(self._txt("No comments collected.", "Комментарии не собраны."))
            return

        raw_output = "parsed_comments.csv"
        self.raw_comments_df.to_csv(raw_output, index=False)
        print(self._txt(f"Raw parsed comments saved to '{raw_output}'", f"Сырые комментарии сохранены в '{raw_output}'"))
        print(self._txt(f"Collected comments: {len(self.raw_comments_df)}", f"Собрано комментариев: {len(self.raw_comments_df)}"))
        print(self._txt(
            f"Unique users: {self.raw_comments_df['username'].nunique()}",
            f"Уникальных пользователей: {self.raw_comments_df['username'].nunique()}",
        ))

    def build_account_features(self) -> None:
        if self.raw_comments_df is None or self.raw_comments_df.empty:
            csv_path = input(
                self._txt(
                    "Path to parsed comments CSV [parsed_comments.csv]: ",
                    "Путь к CSV с комментариями [parsed_comments.csv]: ",
                )
            ).strip() or "parsed_comments.csv"
            if not os.path.exists(csv_path):
                print(self._txt(f"File not found: {csv_path}", f"Файл не найден: {csv_path}"))
                return
            parsed_df = self._read_csv_flexible(csv_path)
            self.account_features_df = build_account_features(parsed_df)
            self.account_features_df.to_csv("account_features.csv", index=False, float_format="%.6f")
            print(self._txt("Account-level features saved to 'account_features.csv'", "Фичи уровня аккаунтов сохранены в 'account_features.csv'"))
        else:
            self.account_features_df = build_account_features(self.raw_comments_df)
            self.account_features_df.to_csv("account_features.csv", index=False, float_format="%.6f")
            print(self._txt("Account-level features saved to 'account_features.csv'", "Фичи уровня аккаунтов сохранены в 'account_features.csv'"))

        print(self._txt(f"Accounts profiled: {len(self.account_features_df)}", f"Профилей аккаунтов: {len(self.account_features_df)}"))

    def run_analyzer(self) -> None:
        training_csv = (
            input(
                self._txt(
                    "Training CSV path [reddit_dead_internet_analysis_2026.csv]: ",
                    "Путь к обучающему CSV [reddit_dead_internet_analysis_2026.csv]: ",
                )
            ).strip()
            or "reddit_dead_internet_analysis_2026.csv"
        )
        if not os.path.exists(training_csv):
            print(self._txt(f"Training file not found: {training_csv}", f"Обучающий файл не найден: {training_csv}"))
            return

        if self.account_features_df is None or self.account_features_df.empty:
            features_path = input(
                self._txt(
                    "Account features CSV path [account_features.csv]: ",
                    "Путь к CSV с фичами аккаунтов [account_features.csv]: ",
                )
            ).strip() or "account_features.csv"
            if not os.path.exists(features_path):
                print(self._txt(f"Account features file not found: {features_path}", f"Файл фич аккаунтов не найден: {features_path}"))
                return
            self.account_features_df = self._read_csv_flexible(features_path)

        classifier = AccountBotClassifier()
        classifier.train(training_csv)
        self.analysis_df = classifier.run_analysis(self.account_features_df, "account_analysis.csv")

        print(self._txt("\nTop suspicious accounts:", "\nТоп подозрительных аккаунтов:"))
        suspicious = classifier.show_suspicious_accounts(self.analysis_df, min_probability=0.3, top_n=10)
        if suspicious.empty:
            print(self._txt("No suspicious accounts detected with threshold 0.3", "Подозрительные аккаунты с порогом 0.3 не найдены"))
        else:
            print(suspicious[["username", "comments_count", "bot_probability", "risk_level"]].to_string(index=False))

        if self._ask_yes_no(self._txt("Generate plots?", "Сгенерировать графики?"), default=True):
            comments_df = self.raw_comments_df
            if comments_df is None or comments_df.empty:
                fallback_comments_path = input(
                    self._txt(
                        "Parsed comments CSV for plots [parsed_comments.csv]: ",
                        "CSV с комментариями для графиков [parsed_comments.csv]: ",
                    )
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
                print(self._txt("Generated plot files:", "Созданы файлы графиков:"))
                for path in paths:
                    print(f"  - {path}")

    def show_suspicious_accounts(self) -> None:
        if self.analysis_df is None or self.analysis_df.empty:
            path = input(
                self._txt(
                    "Analysis CSV path [account_analysis.csv]: ",
                    "Путь к CSV анализа [account_analysis.csv]: ",
                )
            ).strip() or "account_analysis.csv"
            if not os.path.exists(path):
                print(self._txt(f"File not found: {path}", f"Файл не найден: {path}"))
                return
            self.analysis_df = self._read_csv_flexible(path)

        threshold_raw = input(
            self._txt(
                "Minimum bot probability threshold [0.3]: ",
                "Минимальный порог bot probability [0.3]: ",
            )
        ).strip() or "0.3"
        try:
            threshold = float(threshold_raw)
        except ValueError:
            threshold = 0.3

        filtered = self.analysis_df[self.analysis_df["bot_probability"] >= threshold]
        filtered = filtered.sort_values("bot_probability", ascending=False)

        if filtered.empty:
            print(self._txt("No accounts above threshold.", "Аккаунты выше порога не найдены."))
            return

        print(self._txt("\nSuspicious accounts:", "\nПодозрительные аккаунты:"))
        print(filtered[["username", "comments_count", "bot_probability", "risk_level"]].head(50).to_string(index=False))

    def _print_mode_intro_localized(self) -> None:
        self._section(self._tr("mode_selection"))
        print(self._cx("RED", self._tr("default_mode")))
        print(self._cx("SAND", self._tr("default_desc")))
        print(self._cx("RED", self._tr("advanced_mode")))
        print(self._cx("SAND", self._tr("advanced_desc")))
        print(self._cx("RED", self._tr("web_mode")))
        print(self._cx("SAND", self._tr("web_desc")))
        print(self._cx("RED", self._tr("exit")))

    @staticmethod
    def _print_advanced_help(self) -> None:
        self._section(self._txt("ADVANCED MODE HELP", "СПРАВКА ПО ПРОДВИНУТОМУ РЕЖИМУ"))
        print(self._cx("RED", self._txt("Step 1 output:", "Выход шага 1:")) + " parsed_comments.csv")
        print(self._cx("RED", self._txt("Step 2 output:", "Выход шага 2:")) + " account_features.csv")
        print(self._cx("RED", self._txt("Step 3 output:", "Выход шага 3:")) + " account_analysis.csv")
        print(self._cx("SAND", self._txt("You can move these files and continue on another device.", "Можно перенести эти файлы и продолжить на другом устройстве.")))

    def run_default_mode(self) -> None:
        self._section(self._txt("DEFAULT MODE", "РЕЖИМ ПО УМОЛЧАНИЮ"))
        print(self._cx("SAND", self._txt("Full pipeline started.", "Полный пайплайн запущен.")))
        self._section(self._txt("STAGE 1/4 - PARSE", "ЭТАП 1/4 - ПАРСИНГ"))
        self.parse_subreddit()
        if self.raw_comments_df is None or self.raw_comments_df.empty:
            print(self._txt("Pipeline stopped: no parsed comments.", "Пайплайн остановлен: комментарии не собраны."))
            return

        self._section(self._txt("STAGE 2/4 - ACCOUNT FEATURES", "ЭТАП 2/4 - ФИЧИ АККАУНТОВ"))
        self.build_account_features()
        if self.account_features_df is None or self.account_features_df.empty:
            print(self._txt("Pipeline stopped: account features were not built.", "Пайплайн остановлен: фичи аккаунтов не построены."))
            return

        self._section(self._txt("STAGE 3/4 - ANALYZER", "ЭТАП 3/4 - АНАЛИЗАТОР"))
        self.run_analyzer()
        if self.analysis_df is None or self.analysis_df.empty:
            print(self._txt("Pipeline stopped: analyzer did not produce output.", "Пайплайн остановлен: анализатор не дал результат."))
            return

        self._section(self._txt("STAGE 4/4 - SUSPICIOUS ACCOUNTS", "ЭТАП 4/4 - ПОДОЗРИТЕЛЬНЫЕ АККАУНТЫ"))
        self.show_suspicious_accounts()
        self._section(self._txt("DEFAULT MODE COMPLETED", "РЕЖИМ ПО УМОЛЧАНИЮ ЗАВЕРШЕН"))

    def run_advanced_mode(self) -> None:
        self._print_advanced_help()
        while True:
            self._section(self._txt("ADVANCED MODE", "ПРОДВИНУТЫЙ РЕЖИМ"))
            print(
                self._cx("RED", self._txt("1) Parse subreddit", "1) Парсить сабреддит"))
                + self._cx("SAND", self._txt(" (create parsed_comments.csv)", " (создать parsed_comments.csv)"))
            )
            print(
                self._cx("RED", self._txt("2) Build account features", "2) Построить фичи аккаунтов"))
                + self._cx("SAND", self._txt(" (create account_features.csv)", " (создать account_features.csv)"))
            )
            print(
                self._cx("RED", self._txt("3) Run analyzer", "3) Запустить анализатор"))
                + self._cx("SAND", self._txt(" (create account_analysis.csv)", " (создать account_analysis.csv)"))
            )
            print(self._cx("RED", self._txt("4) Show suspicious accounts", "4) Показать подозрительные аккаунты")))
            print(self._cx("RED", self._txt("5) Help", "5) Справка")))
            print(self._cx("RED", self._txt("6) Back", "6) Назад")))

            choice = input(self._txt("Select option: ", "Выберите опцию: ")).strip()
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
                print(self._txt("Unknown option. Choose 1-6.", "Неизвестная опция. Выберите 1-6."))

    def run_web_mode(self) -> None:
        self._section(self._tr("web_title"))
        host = input(self._tr("host_prompt")).strip() or "127.0.0.1"
        port = self._safe_int(self._tr("port_prompt"), 8080)
        print(self._cx("SAND", self._tr("launch_web")))
        run_web_interface(host=host, port=port, open_browser=True, language=self.language)

    def run(self) -> None:
        while True:
            self._print_mode_intro_localized()
            mode = input(self._tr("mode_prompt")).strip()
            if mode == "1":
                self.run_default_mode()
            elif mode == "2":
                self.run_advanced_mode()
            elif mode == "3":
                self.run_web_mode()
            elif mode == "4":
                print(self._tr("bye"))
                break
            else:
                print(self._tr("unknown_mode"))


def run_cli(language: str = "en") -> None:
    RedditAccountAnalyzerCLI(language=language).run()
