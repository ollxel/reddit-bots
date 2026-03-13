#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reddit Parser - CLI интерфейс для парсинга комментариев Reddit
"""
import os
import sys

# Fix Windows encoding - simplified approach
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')
    except Exception:
        pass

import subprocess
import time
from pathlib import Path


def c(color, text):
    """Apply color to text"""
    colors = {
        'BLUE': '\033[94m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'PURPLE': '\033[95m',
        'CYAN': '\033[96m',
        'BOLD': '\033[1m',
        'END': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['END']}"


def print_header():
    """Выводит заголовок приложения"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print(c('CYAN', c('BOLD', """
                    Reddit Parser v2.0
              Парсинг комментариев Reddit
    """)))


def print_menu():
    """Выводит главное меню"""
    print(c('BLUE', 'Выберите режим работы:'))
    print()
    print(f"    {c('GREEN','1)')}  CLI интерфейс")
    print("           - Работа в терминале")
    print("           - Ввод API ключа и параметров вручную")
    print()
    print(f"    {c('RED','2)')}  Выход")
    print()
    return(int(input("Режим: ")))


def run_cli_interface():
    """Запускает CLI интерфейс для парсинга"""
    print(c('BLUE', '\n' + '='*60))
    print("Запуск CLI интерфейса...")
    print('='*60 + '\n')

    # Импортируем модули парсера
    from main_parser import RedditParser, SentimentAnalyzer
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Шаг 1: Ввод API ключа
    print(c('YELLOW', '\nШаг 1: Настройка OpenRouter API'))
    print("-" * 40)
    
    # Проверяем есть ли ключ в .env
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        print(f"Найден ключ в .env файле: {c('GREEN','OK')}")
        use_existing = input("Использовать существующий ключ? (y/n): ").strip().lower()
        if use_existing in ['y', 'yes', 'д', 'да']:
            api_key = env_key
            model = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")
        else:
            api_key = input("Введите ваш OpenRouter API ключ: ").strip()
            model = input("Введите модель (по умолчанию: arcee-ai/trinity-large-preview:free): ").strip()
            if not model:
                model = "arcee-ai/trinity-large-preview:free"
    else:
        api_key = input("Введите ваш OpenRouter API ключ: ").strip()
        model = input("Введите модель (по умолчанию: arcee-ai/trinity-large-preview:free): ").strip()
        if not model:
            model = "arcee-ai/trinity-large-preview:free"

    # Вопрос об использовании sentiment анализа
    use_sentiment = api_key and input("\nВключить анализ тональности комментариев? (y/n): ").strip().lower() in ['y', 'yes', 'д', 'да']

    # Шаг 2: Выбор режима парсинга
    print(c('YELLOW', '\nШаг 2: Выбор режима парсинга'))
    print("-" * 40)
    print("1) Парсить последние посты сабреддита")
    print("2) Парсить конкретный пост по ссылке")
    
    parse_mode = input("Выберите режим (1/2): ").strip()

    # Шаг 3: Ввод параметров
    print(c('YELLOW', '\nШаг 3: Параметры парсинга'))
    print("-" * 40)
    
    target_comments = int(input("Количество комментариев для сбора (по умолчанию 100): ").strip() or "100")
    
    # Создаём парсер
    parser = RedditParser(
        user_agent="RedditDataCollector/1.0 (Educational)",
        run_sentiment=use_sentiment
    )
    
    # Если есть API ключ, создаём SentimentAnalyzer
    if use_sentiment and api_key:
        parser.sentiment = SentimentAnalyzer(api_key=api_key, model=model)
        print(f"{c('GREEN','OK')} SentimentAnalyzer готов (модель: {model})")

    # Запуск парсинга
    print(c('GREEN', '\n' + '='*60))
    print("Начинаем парсинг...")
    print('='*60 + '\n')

    if parse_mode == "1":
        # Парсинг сабреддита
        subreddit = input("Введите название сабреддита (без r/): ").strip()
        category = input("Категория (hot/new/top/rising, по умолчанию hot): ").strip() or "hot"
        
        if category == "top":
            time_filter = input("Временной фильтр (hour/day/week/month/year/all, по умолчанию week): ").strip() or "week"
        else:
            time_filter = "week"
        
        df = parser.parse_subreddit_comments(
            subreddit,
            posts_limit=10,
            category=category,
            time_filter=time_filter,
            target_comments=target_comments
        )
    else:
        # Парсинг по ссылке
        post_url = input("Введите ссылку на пост (https://reddit.com/r/...): ").strip()
        
        df = parser.parse_post_comments(
            post_url,
            target_comments=target_comments
        )

    # Вывод результатов
    if not df.empty:
        print(c('GREEN', '\n' + '='*60))
        print(f"Парсинг завершён! Собрано {len(df)} уникальных пользователей")
        print('='*60)
        
        stats = parser.get_stats()
        print("\n--- Статистика ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        print(c('RED', '\nДанные не собраны'))

    input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    # Добавляем текущую директорию в путь
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print_header()
    mode = print_menu()
    if mode == 1:
        run_cli_interface()
import pandas as pd
import time
import json
from datetime import datetime
import requests
from typing import List, Dict, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from dotenv import load_dotenv

# === Загружаем переменные из .env ===
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")


# ─────────────────────────────────────────────────────────────
#  OpenRouter Sentiment Analyzer
# ─────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Оценивает тональность комментария через OpenRouter API.
    Возвращает float: <0 — негатив, 0 — нейтрал, >0 — позитив.
    Диапазон: от -1.0 до +1.0
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    PROMPT_TEMPLATE = """You are a sentiment analysis engine.
Given a Reddit comment from user '{username}', output ONLY a single floating-point number
in the range [-1.0, +1.0] representing the sentiment:
  -1.0 = very negative
   0.0 = neutral
  +1.0 = very positive

No explanation, no extra text — just the number.

Comment:
\"\"\"{comment_text}\"\"\"
"""

    def __init__(self, api_key: str = None, model: str = None,
                 min_request_interval: float = 1.5):
        self.api_key  = api_key  or OPENROUTER_API_KEY
        self.model    = model    or OPENROUTER_MODEL
        self.interval = min_request_interval
        self._last_call = 0.0

        if not self.api_key:
            raise ValueError(
                "❌ OPENROUTER_API_KEY не найден. "
                "Добавьте его в .env или передайте явно."
            )

    # ── rate limit ──────────────────────────────────────────
    def _wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_call = time.time()

    # ── основной вызов ───────────────────────────────────────
    def score(self, comment_text: str, username: str = "unknown",
              max_retries: int = 3) -> Optional[float]:
        """
        Возвращает float sentiment_score или None при ошибке.
        """
        prompt = self.PROMPT_TEMPLATE.format(
            username=username,
            comment_text=comment_text[:1500]   # обрезаем очень длинные тексты
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.0,
        }

        for attempt in range(max_retries):
            self._wait()
            try:
                resp = requests.post(
                    self.BASE_URL, headers=headers,
                    json=payload, timeout=20
                )
                resp.raise_for_status()
                raw = (resp.json()["choices"][0]["message"]["content"]
                       .strip().replace(",", "."))
                return max(-1.0, min(1.0, float(raw)))
            except (ValueError, KeyError):
                print(f"  ⚠ Не удалось распарсить ответ: '{raw}'")
                return None
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                print(f"  ⚠ OpenRouter ошибка ({e}), retry in {wait}s")
                time.sleep(wait)

        return None


# ─────────────────────────────────────────────────────────────
#  Reddit Parser
# ─────────────────────────────────────────────────────────────

class RedditParser:
    """
    Reddit comment parser using public JSON endpoints.
    Collects ONE comment per unique user with:
      reply_delay, karma, account_age, comment_karma, comment_text, sentiment_score
    """

    def __init__(self, user_agent: str = "RedditParser/1.0",
                 run_sentiment: bool = True):
        self.collected_data: List[Dict] = []
        self.comment_texts:  List[Dict] = []   # ← НОВЫЙ массив: username + text
        self.processed_users: set = set()
        self.target_comments: Optional[int] = None

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.last_request_time  = 0.0
        self.min_request_interval = 5

        # Sentiment — создаём только если есть ключ
        self.run_sentiment = run_sentiment and bool(OPENROUTER_API_KEY)
        if self.run_sentiment:
            self.sentiment = SentimentAnalyzer()
            print(f"✓ SentimentAnalyzer готов (модель: {self.sentiment.model})")
        else:
            self.sentiment = None
            if run_sentiment:
                print("⚠ OPENROUTER_API_KEY не найден — sentiment отключён")

    # ── HTTP helpers ─────────────────────────────────────────

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Optional[Dict] = None,
                      max_retries: int = 3) -> Optional[Dict]:
        self._rate_limit()
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    print(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Request error for {url}: {e}")
                    return None
                wait = 2 ** attempt
                print(f"Request failed, retrying in {wait}s...")
                time.sleep(wait)
        return None

    # ── User info ────────────────────────────────────────────

    def get_user_info(self, username: str) -> Dict[str, Any]:
        url  = f"https://www.reddit.com/user/{username}/about.json"
        data = self._make_request(url)

        if data and "data" in data:
            ud = data["data"]
            created = ud.get("created_utc")
            age_days = None
            if created:
                try:
                    age_days = round((time.time() - float(created)) / 86400, 2)
                except (TypeError, ValueError):
                    pass

            link_karma    = int(ud.get("link_karma",    0) or 0)
            comment_karma = int(ud.get("comment_karma", 0) or 0)
            return {
                "account_age_days": age_days,
                "user_karma":       link_karma + comment_karma,
                "comment_karma":    comment_karma,
            }

        return {"account_age_days": None, "user_karma": 0, "comment_karma": 0}

    # ── Subreddit / post fetching ────────────────────────────

    def fetch_subreddit_posts(self, subreddit_name: str, limit: int = 25,
                              category: str = "hot",
                              time_filter: str = "week") -> List[Dict]:
        url    = f"https://www.reddit.com/r/{subreddit_name}/{category}.json"
        params = {"limit": min(limit, 100)}
        if category == "top":
            params["t"] = time_filter

        data = self._make_request(url, params)
        if not data or "data" not in data:
            return []

        posts = []
        for child in data["data"].get("children", []):
            if child["kind"] == "t3":
                pd_ = child["data"]
                posts.append({
                    "title":        pd_.get("title", ""),
                    "author":       pd_.get("author", ""),
                    "permalink":    pd_.get("permalink", ""),
                    "created_utc":  pd_.get("created_utc", 0),
                    "score":        pd_.get("score", 0),
                    "num_comments": pd_.get("num_comments", 0),
                    "id":           pd_.get("id", ""),
                })
        return posts

    def _extract_comments(self, comments_data: List,
                          all_comments: List = None) -> List[Dict]:
        if all_comments is None:
            all_comments = []
        for item in comments_data:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind", "")
            data = item.get("data", {})
            if kind == "t1":
                all_comments.append({
                    "id":           data.get("id", ""),
                    "author":       data.get("author", ""),
                    "body":         data.get("body", ""),
                    "score":        data.get("score", 0),
                    "created_utc":  data.get("created_utc", 0),
                })
                replies = data.get("replies", "")
                if isinstance(replies, dict) and "data" in replies:
                    self._extract_comments(
                        replies["data"].get("children", []), all_comments
                    )
            elif kind == "Listing":
                self._extract_comments(data.get("children", []), all_comments)
        return all_comments

    def scrape_post_details(self, permalink: str) -> Optional[Dict]:
        if not permalink.startswith("/"):
            permalink = "/" + permalink
        url  = f"https://www.reddit.com{permalink}.json"
        data = self._make_request(url)
        if not data or len(data) < 2:
            return None

        post_listing     = data[0]
        comments_listing = data[1]

        if ("data" not in post_listing or
                "children" not in post_listing["data"]):
            return None

        pd_ = post_listing["data"]["children"][0]["data"]
        comments = []
        if "data" in comments_listing:
            children = comments_listing["data"].get("children", [])
            comments = self._extract_comments(children)

        return {
            "title":        pd_.get("title", ""),
            "author":       pd_.get("author", ""),
            "created_utc":  pd_.get("created_utc", 0),
            "score":        pd_.get("score", 0),
            "num_comments": pd_.get("num_comments", 0),
            "selftext":     pd_.get("selftext", ""),
            "url":          pd_.get("url", ""),
            "permalink":    pd_.get("permalink", ""),
            "comments":     comments,
        }

    # ── Comment processing ───────────────────────────────────

    def process_comments(self, comments: List[Dict], post_author: str,
                         post_time: float, post_title: Optional[str] = None,
                         post_url: Optional[str] = None,
                         target_comments: int = 700) -> bool:
        """
        Обрабатывает комментарии поста.
        Один комментарий на пользователя.
        Возвращает True — продолжать, False — остановиться.
        """
        for comment in comments:
            author = comment.get("author")
            body   = comment.get("body", "").strip()

            # Фильтры
            if (not author or author == post_author or
                    author == "[deleted]" or not body or
                    body == "[removed]"):
                continue
            if author in self.processed_users:
                continue

            try:
                user_info = self.get_user_info(author)
                account_age_days = user_info["account_age_days"]
                if account_age_days is None:
                    print(f"  Skipping {author}: no account data")
                    continue

                user_karma    = user_info["user_karma"]
                comment_karma = user_info["comment_karma"]
                comment_created   = comment.get("created_utc", 0)
                reply_delay_secs  = int(comment_created - post_time) if comment_created else 0

                # ── Sentiment через OpenRouter ────────────────
                sentiment_score = None
                if self.run_sentiment and self.sentiment:
                    print(f"  🔍 Sentiment для @{author}...")
                    sentiment_score = self.sentiment.score(body, username=author)

                # ── Основная запись (для ML) ──────────────────
                record = {
                    "reply_delay_seconds": reply_delay_secs,
                    "user_karma":          user_karma,
                    "account_age_days":    account_age_days,
                    "comment_karma":       comment_karma,
                    "sentiment_score":     sentiment_score,
                    # метаданные
                    "username":            author,
                    "comment_id":          comment.get("id", ""),
                    "comment_score":       comment.get("score", 0),
                }
                if post_title:
                    record["post_title"] = post_title
                if post_url:
                    record["post_url"] = post_url

                self.collected_data.append(record)

                # ── Текстовый массив (username + text) ───────
                self.comment_texts.append({
                    "username":      author,
                    "comment_text":  body,
                    "sentiment_score": sentiment_score,
                })

                self.processed_users.add(author)

                total = len(self.collected_data)
                if total >= self.target_comments:
                    print(f"\n{'='*60}")
                    print(f"Reached target: {total} unique users")
                    print(f"{'='*60}")
                    self.save_to_csv()
                    if not self.ask_continue():
                        return False
                    else:
                        print("Continuing parsing...")
                        self.target_comments += 700
                        return True

            except Exception as e:
                print(f"Warning: Error processing comment from {author}: {e}")
                continue

        return True

    # ── Сохранение ───────────────────────────────────────────

    def save_to_csv(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"reddit_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(self.collected_data)
        df.to_csv(filename, index=False)
        print(f"\nData saved to '{filename}'")
        return filename

    def save_comment_texts(self, filename: Optional[str] = None) -> str:
        """Сохраняет отдельный CSV с username, comment_text, sentiment_score"""
        if filename is None:
            filename = f"comment_texts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(self.comment_texts)
        df.to_csv(filename, index=False)
        print(f"Comment texts saved to '{filename}'")
        return filename

    def ask_continue(self) -> bool:
        response = input("\nContinue parsing? (yes/no): ").strip().lower()
        return response in ["yes", "y"]

    # ── Публичные методы парсинга ────────────────────────────

    def parse_subreddit_comments(self, subreddit_name: str,
                                 posts_limit: int = 10,
                                 category: str = "hot",
                                 time_filter: str = "week",
                                 target_comments: int = 700) -> pd.DataFrame:
        self.target_comments = target_comments
        print(f"Parsing subreddit: r/{subreddit_name}")
        print(f"Category: {category}, Time filter: {time_filter}")
        print(f"Target: {target_comments} unique users")
        print("-" * 60)

        posts = self.fetch_subreddit_posts(
            subreddit_name, limit=posts_limit,
            category=category, time_filter=time_filter
        )
        if not posts:
            print("No posts found")
            return pd.DataFrame()

        print(f"Found {len(posts)} posts")

        for i, post in enumerate(posts, 1):
            title = post.get("title", "")[:60]
            print(f"\n[{i}/{len(posts)}] {title}...")

            permalink    = post.get("permalink", "")
            post_details = self.scrape_post_details(permalink)
            if not post_details:
                print("  Could not fetch post details")
                continue

            post_author = post_details.get("author")
            post_time   = post_details.get("created_utc", time.time())
            post_title  = post_details.get("title", "")
            post_url    = post.get("permalink", "")
            comments    = post_details.get("comments", [])

            print(f"  Found {len(comments)} total comments")
            initial     = len(self.collected_data)
            should_cont = self.process_comments(
                comments, post_author, post_time,
                post_title, post_url
            )
            print(f"  Added: {len(self.collected_data)-initial} | "
                  f"Total: {len(self.collected_data)}")

            if not should_cont:
                break

        df = pd.DataFrame(self.collected_data)
        print("\n" + "=" * 60)
        print(f"Parsing complete: {len(df)} unique users")
        print("=" * 60 + "\n")
        self.save_to_csv()
        self.save_comment_texts()
        return df

    def parse_post_comments(self, post_url: str,
                            target_comments: int = 700) -> pd.DataFrame:
        self.target_comments = target_comments
        if "reddit.com" in post_url:
            permalink = post_url.split("reddit.com")[1]
        else:
            permalink = post_url

        print(f"Parsing post from: {post_url}")
        post_details = self.scrape_post_details(permalink)
        if not post_details:
            print("Failed to scrape post details")
            return pd.DataFrame()

        self.process_comments(
            post_details.get("comments", []),
            post_details.get("author"),
            post_details.get("created_utc", time.time()),
            post_details.get("title", ""),
            post_url,
            target_comments,
        )
        df = pd.DataFrame(self.collected_data)
        self.save_to_csv()
        self.save_comment_texts()
        return df

    def parse_multiple_posts(self, post_urls: List[str],
                             target_comments: int = 700) -> pd.DataFrame:
        self.target_comments = target_comments
        for i, url in enumerate(post_urls, 1):
            print(f"\n[{i}/{len(post_urls)}] Parsing post...")
            permalink = url.split("reddit.com")[1] if "reddit.com" in url else url
            post_details = self.scrape_post_details(permalink)
            if not post_details:
                print(f"Failed to scrape: {url}")
                continue

            should_cont = self.process_comments(
                post_details.get("comments", []),
                post_details.get("author"),
                post_details.get("created_utc", time.time()),
                post_details.get("title", ""),
                url,
                target_comments,
            )
            if not should_cont:
                break

        df = pd.DataFrame(self.collected_data)
        self.save_to_csv()
        self.save_comment_texts()
        return df

    def get_stats(self) -> Dict[str, Any]:
        if not self.collected_data:
            return {}
        df = pd.DataFrame(self.collected_data)
        stats: Dict[str, Any] = {
            "total_unique_users": len(df),
        }
        for col in ["reply_delay_seconds", "user_karma", "comment_karma",
                    "account_age_days", "sentiment_score"]:
            if col in df.columns:
                stats[f"avg_{col}"]    = df[col].mean()
                stats[f"median_{col}"] = df[col].median()
        return stats


# ─────────────────────────────────────────────────────────────
#  ML Model
# ─────────────────────────────────────────────────────────────

class Model:

    # Признаки, которые использует модель
    FEATURE_COLS = [
        "sentiment_score",
        "account_age_days",
        "user_karma",
        "comment_karma",
        "reply_delay_seconds",
    ]

    @staticmethod
    def prepare_data(filepath: str = "reddit_dead_internet_analysis_2026.csv"):
        """Загружает и подготавливает обучающие данные"""
        data = pd.read_csv(filepath)
        print(f"✓ Загружено {len(data)} строк для обучения")

        # Удаляем ненужные колонки
        cols_to_drop = ["subreddit", "comment_id", "bot_type_label",
                        "bot_probability", "contains_links", "avg_word_length"]
        data = data.drop(columns=[c for c in cols_to_drop if c in data.columns],
                         errors="ignore")

        # Добавляем sentiment_score = 0 если нет (совместимость со старыми CSV)
        if "sentiment_score" not in data.columns:
            print("  ⚠ 'sentiment_score' не найден — заполняем 0")
            data["sentiment_score"] = 0.0

        return data

    @staticmethod
    def prepare_parser_data(df: pd.DataFrame):
        """
        Подготавливает данные из парсера для предсказания.
        """
        data = df.copy()
        missing = [c for c in Model.FEATURE_COLS if c not in data.columns]
        if missing:
            print(f"  ⚠ Колонки отсутствуют, заполняем нулями: {missing}")
            for c in missing:
                data[c] = 0.0

        usernames = data["username"].copy() if "username" in data.columns else None
        X = data[Model.FEATURE_COLS].fillna(0)
        return X, usernames

    @staticmethod
    def train_model(data: pd.DataFrame) -> RandomForestClassifier:
        """Обучает модель RandomForest"""
        if "is_bot_flag" not in data.columns:
            raise ValueError("❌ В данных нет колонки 'is_bot_flag'")

        y = data["is_bot_flag"]

        # Берём только признаки, которые есть в данных
        available = [c for c in Model.FEATURE_COLS if c in data.columns]
        print(f"  Признаки для обучения: {available}")
        X = data[available].fillna(0)

        stratify_param = y if len(y.unique()) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )

        clf = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        clf._feature_names = available   # сохраняем для predict

        y_pred = clf.predict(X_test)
        print("\n" + "=" * 60)
        print("📊 ОТЧЁТ ОБ ОБУЧЕНИИ")
        print("=" * 60)
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Feature importance
        importances = clf.feature_importances_
        print("\nFeature importances:")
        for name, imp in sorted(zip(available, importances),
                                key=lambda x: -x[1]):
            print(f"  {name:<28} {imp:.4f}")

        return clf

    @staticmethod
    def predict(clf: RandomForestClassifier, df: pd.DataFrame) -> pd.DataFrame:
        """
        Делает предсказания на данных парсера.
        Совместим как с новыми (5 признаков), так и со старыми данными.
        """
        # Используем признаки, на которых обучалась конкретная модель
        feature_cols = (getattr(clf, "_feature_names", None)
                        or Model.FEATURE_COLS)

        data = df.copy()
        for c in feature_cols:
            if c not in data.columns:
                data[c] = 0.0

        usernames = data["username"].copy() if "username" in data.columns else None
        X = data[feature_cols].fillna(0)

        predictions   = clf.predict(X)
        probabilities = (clf.predict_proba(X)
                         if hasattr(clf, "predict_proba") else None)

        result = pd.DataFrame({
            "username":   (usernames if usernames is not None
                           else range(len(predictions))),
            "is_bot_flag": predictions,
        })
        if probabilities is not None:
            result["bot_probability"] = (
                probabilities[:, 1] if probabilities.shape[1] > 1
                else probabilities[:, 0]
            )

        print(f"\n✓ Предсказаний: {len(predictions)}")
        print(f"  Боты: {sum(predictions)}, "
              f"Люди: {len(predictions) - sum(predictions)}")
        return result


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # === ЧАСТЬ 1: Парсинг ===
    parse = input("Start parse? y/n: ").strip().lower()
    df_raw = None

    if parse in ["y", "yes"]:
        use_sentiment = input(
            "Enable sentiment analysis via OpenRouter? y/n: "
        ).strip().lower() in ["y", "yes"]

        parser = RedditParser(
            user_agent="RedditDataCollector/1.0 (Educational)",
            run_sentiment=use_sentiment,
        )
        df_raw = parser.parse_subreddit_comments(
            "AskReddit",
            posts_limit=10,
            category="hot",
            time_filter="week",
            target_comments=100,
        )
        if not df_raw.empty:
            print(f"\n✓ Собрано {len(df_raw)} уникальных пользователей")
            print(f"  Текстовых записей: {len(parser.comment_texts)}")
            stats = parser.get_stats()
            print("\n--- Statistics ---")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    # === ЧАСТЬ 2: Модель ===
    run_model = input("\nRun ML model? y/n: ").strip().lower()

    if run_model in ["y", "yes"]:
        try:
            print("\n[1/3] Загрузка обучающих данных...")
            train_data = Model.prepare_data(
                "reddit_dead_internet_analysis_2026.csv"
            )

            print("\n[2/3] Обучение модели...")
            clf = Model.train_model(train_data)

            print("\n[3/3] Предсказание...")
            if df_raw is not None and not df_raw.empty:
                print("  → Используем данные из парсера")
                predict_source = df_raw
            else:
                csv_file = input(
                    "  → Введите путь к CSV (Enter = collected_reddit_data.csv): "
                ).strip() or "collected_reddit_data.csv"
                predict_source = pd.read_csv(csv_file)
                print(f"  → Загружено {len(predict_source)} записей")

            predictions_df = Model.predict(clf, predict_source)
            predictions_df.to_csv("model_predictions.csv", index=False)

            print("\n" + "=" * 60)
            print("📋 РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ")
            print("=" * 60)
            print(predictions_df.head(10))
            print(f"\n✓ Результаты сохранены в 'model_predictions.csv'")
            print(f"  Всего: {len(predictions_df)}")
            print(f"  Ботов: {predictions_df['is_bot_flag'].sum()}")
            print(f"  Людей: {len(predictions_df) - predictions_df['is_bot_flag'].sum()}")

        except FileNotFoundError as e:
            print(f"❌ Файл не найден: {e}")
        except KeyError as e:
            print(f"❌ Ошибка: в данных нет колонки {e}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
