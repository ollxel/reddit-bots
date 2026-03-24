# -*- coding: utf-8 -*-
"""Reddit parser and sentiment integration for account-level analysis."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


def is_bot(
    user_karma: int,
    account_age_days: float,
    reply_delay_seconds: int,
    sentiment_score: Optional[float],
    comment_text: str = "",
    comment_score: int = 0,
) -> bool:
    """Legacy heuristic preserved for backward compatibility."""
    bot_score = 0
    human_score = 0

    if account_age_days and account_age_days < 7:
        bot_score += 5
    elif account_age_days and account_age_days < 30:
        bot_score += 2

    if user_karma < 10:
        bot_score += 4
    elif user_karma < 50:
        bot_score += 2
    elif user_karma < 100:
        bot_score += 1

    if reply_delay_seconds < 5:
        bot_score += 5
    elif reply_delay_seconds < 15:
        bot_score += 3
    elif reply_delay_seconds < 30:
        bot_score += 1

    if comment_score < 0:
        bot_score += 3

    if comment_text and len(comment_text) < 20:
        bot_score += 2

    if account_age_days and account_age_days > 365:
        human_score += 3
    elif account_age_days and account_age_days > 180:
        human_score += 2
    elif account_age_days and account_age_days > 60:
        human_score += 1

    if user_karma > 1000:
        human_score += 3
    elif user_karma > 500:
        human_score += 2
    elif user_karma > 100:
        human_score += 1

    if 60 <= reply_delay_seconds <= 3600:
        human_score += 2
    elif 30 <= reply_delay_seconds <= 86400:
        human_score += 1

    if comment_score > 10:
        human_score += 2
    elif comment_score > 0:
        human_score += 1

    if comment_text and len(comment_text) > 100:
        human_score += 2
    elif comment_text and len(comment_text) > 50:
        human_score += 1

    return bot_score > human_score + 2


class SentimentAnalyzer:
    """OpenRouter-backed sentiment scoring."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    PROMPT_TEMPLATE = """You are a sentiment analysis engine.
Given a Reddit comment from user '{username}', output ONLY a single floating-point number
in the range [-1.0, +1.0] representing the sentiment:
  -1.0 = very negative
   0.0 = neutral
  +1.0 = very positive

No explanation, no extra text - just the number.

Comment:
\"\"\"{comment_text}\"\"\"
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "arcee-ai/trinity-large-preview:free",
        min_request_interval: float = 1.5,
    ):
        self.api_key = api_key
        self.model = model
        self.interval = min_request_interval
        self._last_call = 0.0
        self.request_count = 0

    def _wait(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_call = time.time()

    def score(
        self,
        comment_text: str,
        username: str = "unknown",
        max_retries: int = 3,
    ) -> Optional[float]:
        if not self.api_key:
            return None

        self.request_count += 1
        if self.request_count % 10 == 0:
            print(f"  [SENTIMENT] Processed {self.request_count} comments...")

        prompt = self.PROMPT_TEMPLATE.format(
            username=username,
            comment_text=comment_text[:1500],
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
                response = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=20,
                )
                response.raise_for_status()
                raw = (
                    response.json()["choices"][0]["message"]["content"]
                    .strip()
                    .replace(",", ".")
                )
                return max(-1.0, min(1.0, float(raw)))
            except (ValueError, KeyError, requests.exceptions.RequestException):
                if attempt == max_retries - 1:
                    return None
                time.sleep(2**attempt)

        return None


class RedditParser:
    """Collects Reddit comments with metadata for account-level analysis."""

    def __init__(
        self,
        user_agent: str = "RedditParserCLI/1.0",
        run_sentiment: bool = True,
        unique_users_only: bool = False,
    ):
        self.collected_data: List[Dict[str, Any]] = []
        self.comment_texts: List[Dict[str, Any]] = []
        self.processed_users: set[str] = set()
        self.user_cache: Dict[str, Dict[str, Any]] = {}
        self.target_comments: Optional[int] = None

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.last_request_time = 0.0
        self.min_request_interval = 5.0

        self.run_sentiment = run_sentiment
        self.sentiment: Optional[SentimentAnalyzer] = None
        self.unique_users_only = unique_users_only

    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Optional[Dict[str, Any]]:
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
            except requests.exceptions.RequestException as exc:
                if attempt == max_retries - 1:
                    print(f"Request error for {url}: {exc}")
                    return None
                wait = 2**attempt
                print(f"Request failed, retrying in {wait}s...")
                time.sleep(wait)
        return None

    def get_user_info(self, username: str) -> Dict[str, Any]:
        if username in self.user_cache:
            return self.user_cache[username]

        url = f"https://www.reddit.com/user/{username}/about.json"
        data = self._make_request(url)

        if data and "data" in data:
            user_data = data["data"]
            created = user_data.get("created_utc")
            age_days = None
            if created:
                try:
                    age_days = round((time.time() - float(created)) / 86400, 2)
                except (TypeError, ValueError):
                    age_days = None

            link_karma = int(user_data.get("link_karma", 0) or 0)
            comment_karma = int(user_data.get("comment_karma", 0) or 0)

            info = {
                "account_age_days": age_days,
                "user_karma": link_karma + comment_karma,
                "comment_karma": comment_karma,
            }
            self.user_cache[username] = info
            return info

        info = {"account_age_days": None, "user_karma": 0, "comment_karma": 0}
        self.user_cache[username] = info
        return info

    def fetch_subreddit_posts(
        self,
        subreddit_name: str,
        limit: int = 25,
        category: str = "hot",
        time_filter: str = "week",
    ) -> List[Dict[str, Any]]:
        url = f"https://www.reddit.com/r/{subreddit_name}/{category}.json"
        params: Dict[str, Any] = {"limit": min(limit, 100)}
        if category == "top":
            params["t"] = time_filter

        data = self._make_request(url, params)
        if not data or "data" not in data:
            return []

        posts: List[Dict[str, Any]] = []
        for child in data["data"].get("children", []):
            if child.get("kind") != "t3":
                continue
            post_data = child.get("data", {})
            posts.append(
                {
                    "title": post_data.get("title", ""),
                    "author": post_data.get("author", ""),
                    "permalink": post_data.get("permalink", ""),
                    "created_utc": post_data.get("created_utc", 0),
                    "score": post_data.get("score", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "id": post_data.get("id", ""),
                    "subreddit": post_data.get("subreddit", subreddit_name),
                }
            )
        return posts

    def _extract_comments(
        self,
        comments_data: List[Dict[str, Any]],
        all_comments: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if all_comments is None:
            all_comments = []

        for item in comments_data:
            if not isinstance(item, dict):
                continue

            kind = item.get("kind", "")
            data = item.get("data", {})

            if kind == "t1":
                all_comments.append(
                    {
                        "id": data.get("id", ""),
                        "author": data.get("author", ""),
                        "body": data.get("body", ""),
                        "score": data.get("score", 0),
                        "created_utc": data.get("created_utc", 0),
                        "link_id": data.get("link_id", ""),
                    }
                )

                replies = data.get("replies", "")
                if isinstance(replies, dict) and "data" in replies:
                    self._extract_comments(
                        replies["data"].get("children", []),
                        all_comments,
                    )

            elif kind == "Listing":
                self._extract_comments(data.get("children", []), all_comments)

        return all_comments

    def scrape_post_details(self, permalink: str) -> Optional[Dict[str, Any]]:
        if not permalink.startswith("/"):
            permalink = "/" + permalink

        url = f"https://www.reddit.com{permalink}.json"
        data = self._make_request(url)
        if not data or len(data) < 2:
            return None

        post_listing = data[0]
        comments_listing = data[1]

        if "data" not in post_listing or "children" not in post_listing["data"]:
            return None

        post_data = post_listing["data"]["children"][0]["data"]

        comments: List[Dict[str, Any]] = []
        if "data" in comments_listing:
            children = comments_listing["data"].get("children", [])
            comments = self._extract_comments(children)

        return {
            "title": post_data.get("title", ""),
            "author": post_data.get("author", ""),
            "created_utc": post_data.get("created_utc", 0),
            "score": post_data.get("score", 0),
            "num_comments": post_data.get("num_comments", 0),
            "selftext": post_data.get("selftext", ""),
            "url": post_data.get("url", ""),
            "permalink": post_data.get("permalink", ""),
            "id": post_data.get("id", ""),
            "subreddit": post_data.get("subreddit", ""),
            "comments": comments,
        }

    def process_comments(
        self,
        comments: List[Dict[str, Any]],
        post_author: str,
        post_time: float,
        post_title: Optional[str] = None,
        post_url: Optional[str] = None,
        post_id: Optional[str] = None,
        subreddit: Optional[str] = None,
    ) -> bool:
        for comment in comments:
            author = comment.get("author")
            body = comment.get("body", "").strip()
            comment_score = int(comment.get("score", 0) or 0)

            if not author or author == post_author or author == "[deleted]":
                continue
            if not body or body == "[removed]":
                continue
            if self.unique_users_only and author in self.processed_users:
                continue

            try:
                user_info = self.get_user_info(author)
                account_age_days = user_info.get("account_age_days")
                user_karma = int(user_info.get("user_karma", 0) or 0)
                comment_karma = int(user_info.get("comment_karma", 0) or 0)

                comment_created = comment.get("created_utc", 0)
                if comment_created:
                    reply_delay_secs = max(0, int(float(comment_created) - float(post_time)))
                else:
                    reply_delay_secs = 0

                sentiment_score = None
                if self.run_sentiment and self.sentiment:
                    print(f"  [SENTIMENT] @{author}...")
                    sentiment_score = self.sentiment.score(body, username=author)

                comment_timestamp = float(comment_created or 0)
                record = {
                    "username": author,
                    "comment_id": comment.get("id", ""),
                    "comment_text": body,
                    "comment_score": comment_score,
                    "reply_delay_seconds": reply_delay_secs,
                    "sentiment_score": sentiment_score,
                    "post_id": (post_id or comment.get("link_id", "").replace("t3_", "")),
                    "subreddit": subreddit or "",
                    "timestamp": datetime.utcfromtimestamp(comment_timestamp).isoformat() if comment_timestamp else "",
                    "created_utc": comment_timestamp,
                    "user_karma": user_karma,
                    "account_age_days": account_age_days,
                    "comment_karma": comment_karma,
                    "is_bot": is_bot(
                        user_karma,
                        account_age_days or 0,
                        reply_delay_secs,
                        sentiment_score,
                        comment_text=body,
                        comment_score=comment_score,
                    ),
                }

                if post_title:
                    record["post_title"] = post_title
                if post_url:
                    record["post_url"] = post_url

                self.collected_data.append(record)
                self.comment_texts.append(
                    {
                        "username": author,
                        "comment_text": body,
                        "sentiment_score": sentiment_score,
                        "comment_score": comment_score,
                        "post_id": record["post_id"],
                        "subreddit": record["subreddit"],
                        "timestamp": record["timestamp"],
                    }
                )

                self.processed_users.add(author)

                if self.target_comments and len(self.collected_data) >= self.target_comments:
                    print("\n" + "=" * 60)
                    print(f"Reached target: {len(self.collected_data)} comments")
                    print("=" * 60)
                    self.save_to_csv()
                    if not self.ask_continue():
                        return False
                    self.target_comments += max(100, self.target_comments)
                    print(f"Continuing parsing. New target: {self.target_comments}")
                    return True

            except Exception as exc:
                print(f"Warning: Error processing comment from {author}: {exc}")
                continue

        return True

    def save_to_csv(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"reddit_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(self.collected_data)
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"\nData saved to '{filename}'")
        return filename

    def save_comment_texts(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"comment_texts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(self.comment_texts)
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"Comment texts saved to '{filename}'")
        return filename

    def ask_continue(self) -> bool:
        response = input("\nContinue parsing? (yes/no): ").strip().lower()
        return response in {"yes", "y"}

    def parse_subreddit_comments(
        self,
        subreddit_name: str,
        posts_limit: int = 10,
        category: str = "hot",
        time_filter: str = "week",
        target_comments: int = 700,
    ) -> pd.DataFrame:
        self.target_comments = target_comments
        print(f"Parsing subreddit: r/{subreddit_name}")
        print(f"Category: {category}, Time filter: {time_filter}")
        print(f"Target comments: {target_comments}")
        print("-" * 60)

        posts = self.fetch_subreddit_posts(
            subreddit_name,
            limit=posts_limit,
            category=category,
            time_filter=time_filter,
        )
        if not posts:
            print("No posts found")
            return pd.DataFrame()

        print(f"Found {len(posts)} posts")

        for i, post in enumerate(posts, start=1):
            title = post.get("title", "")[:60]
            print(f"\n[{i}/{len(posts)}] {title}...")

            permalink = post.get("permalink", "")
            post_details = self.scrape_post_details(permalink)
            if not post_details:
                print("  Could not fetch post details")
                continue

            comments = post_details.get("comments", [])
            print(f"  Found {len(comments)} total comments")

            initial = len(self.collected_data)
            should_continue = self.process_comments(
                comments=comments,
                post_author=post_details.get("author", ""),
                post_time=post_details.get("created_utc", time.time()),
                post_title=post_details.get("title", ""),
                post_url=post.get("permalink", ""),
                post_id=post_details.get("id", ""),
                subreddit=post_details.get("subreddit", subreddit_name),
            )
            print(
                f"  Added: {len(self.collected_data) - initial} | "
                f"Total comments: {len(self.collected_data)}"
            )

            if not should_continue:
                break

        df = pd.DataFrame(self.collected_data)
        print("\n" + "=" * 60)
        print(f"Parsing complete: {len(df)} comments from {df['username'].nunique() if not df.empty else 0} users")
        print("=" * 60 + "\n")

        self.save_to_csv()
        self.save_comment_texts()
        return df

    def parse_post_comments(
        self,
        post_url: str,
        target_comments: int = 700,
    ) -> pd.DataFrame:
        self.target_comments = target_comments
        permalink = post_url.split("reddit.com")[1] if "reddit.com" in post_url else post_url

        print(f"Parsing post from: {post_url}")
        post_details = self.scrape_post_details(permalink)
        if not post_details:
            print("Failed to scrape post details")
            return pd.DataFrame()

        self.process_comments(
            comments=post_details.get("comments", []),
            post_author=post_details.get("author", ""),
            post_time=post_details.get("created_utc", time.time()),
            post_title=post_details.get("title", ""),
            post_url=post_url,
            post_id=post_details.get("id", ""),
            subreddit=post_details.get("subreddit", ""),
        )

        df = pd.DataFrame(self.collected_data)
        self.save_to_csv()
        self.save_comment_texts()
        return df

    def parse_multiple_posts(
        self,
        post_urls: List[str],
        target_comments: int = 700,
    ) -> pd.DataFrame:
        self.target_comments = target_comments

        for i, post_url in enumerate(post_urls, start=1):
            print(f"\n[{i}/{len(post_urls)}] Parsing post...")
            permalink = post_url.split("reddit.com")[1] if "reddit.com" in post_url else post_url
            post_details = self.scrape_post_details(permalink)
            if not post_details:
                print(f"Failed to scrape: {post_url}")
                continue

            should_continue = self.process_comments(
                comments=post_details.get("comments", []),
                post_author=post_details.get("author", ""),
                post_time=post_details.get("created_utc", time.time()),
                post_title=post_details.get("title", ""),
                post_url=post_url,
                post_id=post_details.get("id", ""),
                subreddit=post_details.get("subreddit", ""),
            )
            if not should_continue:
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
            "total_comments": int(len(df)),
            "total_unique_users": int(df["username"].nunique()) if "username" in df.columns else 0,
        }

        if "is_bot" in df.columns:
            bots = int(df["is_bot"].sum())
            stats["heuristic_bots_detected"] = bots
            stats["heuristic_humans_detected"] = int(len(df) - bots)

        for column in [
            "reply_delay_seconds",
            "user_karma",
            "comment_karma",
            "account_age_days",
            "sentiment_score",
            "comment_score",
        ]:
            if column in df.columns:
                stats[f"avg_{column}"] = float(df[column].fillna(0).mean())
                stats[f"median_{column}"] = float(df[column].fillna(0).median())

        return stats
