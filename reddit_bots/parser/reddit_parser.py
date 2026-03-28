# -*- coding: utf-8 -*-
"""Reddit parser and sentiment integration for account-level analysis."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

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
                if response.status_code in {403, 404}:
                    return None
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

    @staticmethod
    def _parse_date_boundary(date_text: str, end_of_day: bool = False) -> float:
        parsed = datetime.strptime(date_text.strip(), "%Y-%m-%d")
        if end_of_day:
            parsed = parsed + timedelta(days=1) - timedelta(seconds=1)
        return parsed.replace(tzinfo=timezone.utc).timestamp()

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

    def fetch_subreddit_posts_by_date_range(
        self,
        subreddit_name: str,
        start_date: str,
        end_date: str,
        category: str = "new",
        max_batches: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Fetches posts in a subreddit for the inclusive date range [start_date, end_date].
        Date format: YYYY-MM-DD (UTC).
        """
        start_ts = self._parse_date_boundary(start_date, end_of_day=False)
        end_ts = self._parse_date_boundary(end_date, end_of_day=True)
        if end_ts < start_ts:
            raise ValueError("end_date must be >= start_date")

        if category not in {"new", "hot", "top", "rising"}:
            category = "new"

        url = f"https://www.reddit.com/r/{subreddit_name}/{category}.json"
        posts: List[Dict[str, Any]] = []
        after: Optional[str] = None

        for _ in range(max_batches):
            params: Dict[str, Any] = {"limit": 100}
            if after:
                params["after"] = after
            if category == "top":
                params["t"] = "all"

            data = self._make_request(url, params=params)
            if not data or "data" not in data:
                break

            children = data["data"].get("children", [])
            if not children:
                break

            stop_due_to_old = False
            for child in children:
                if child.get("kind") != "t3":
                    continue

                post_data = child.get("data", {})
                created_utc = float(post_data.get("created_utc", 0) or 0)

                if created_utc > end_ts:
                    continue
                if created_utc < start_ts:
                    if category == "new":
                        stop_due_to_old = True
                    continue

                posts.append(
                    {
                        "title": post_data.get("title", ""),
                        "author": post_data.get("author", ""),
                        "permalink": post_data.get("permalink", ""),
                        "created_utc": created_utc,
                        "score": post_data.get("score", 0),
                        "num_comments": post_data.get("num_comments", 0),
                        "id": post_data.get("id", ""),
                        "subreddit": post_data.get("subreddit", subreddit_name),
                    }
                )

            after = data["data"].get("after")
            if not after or (stop_due_to_old and category == "new"):
                break

        posts.sort(key=lambda item: item.get("created_utc", 0))
        return posts

    def _extract_comments(
        self,
        comments_data: List[Dict[str, Any]],
        all_comments: Optional[List[Dict[str, Any]]] = None,
        more_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        if all_comments is None:
            all_comments = []
        if more_ids is None:
            more_ids = []

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
                        more_ids,
                    )

            elif kind == "more":
                children_ids = data.get("children", [])
                if isinstance(children_ids, list):
                    more_ids.extend([item_id for item_id in children_ids if isinstance(item_id, str)])

            elif kind == "Listing":
                self._extract_comments(data.get("children", []), all_comments, more_ids)

        return all_comments, more_ids

    def _fetch_morechildren_comments(
        self,
        post_id: str,
        child_ids: List[str],
        sort: str = "new",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not post_id or not child_ids:
            return []

        url = "https://www.reddit.com/api/morechildren.json"
        queue: List[str] = list(child_ids)
        seen: Set[str] = set(queue)
        collected: List[Dict[str, Any]] = []

        while queue:
            if limit is not None and len(collected) >= limit:
                break

            chunk = queue[:100]
            queue = queue[100:]

            params = {
                "api_type": "json",
                "link_id": f"t3_{post_id}",
                "children": ",".join(chunk),
                "sort": sort,
            }
            data = self._make_request(url, params=params, max_retries=2)
            if not data:
                continue

            things = data.get("json", {}).get("data", {}).get("things", [])
            comments, nested_more = self._extract_comments(things, all_comments=[], more_ids=[])
            collected.extend(comments)

            for more_id in nested_more:
                if more_id not in seen:
                    seen.add(more_id)
                    queue.append(more_id)

        if limit is not None:
            return collected[:limit]
        return collected

    def _collect_post_comments(
        self,
        post_details: Dict[str, Any],
        comments_limit: Optional[int] = None,
        sort: str = "new",
    ) -> List[Dict[str, Any]]:
        base_comments = list(post_details.get("comments", []))
        more_ids = list(post_details.get("more_comment_ids", []))
        post_id = post_details.get("id", "")

        if comments_limit is not None and comments_limit <= 0:
            return []

        if comments_limit is None:
            remaining = None
        else:
            remaining = max(0, comments_limit - len(base_comments))

        if more_ids and (remaining is None or remaining > 0):
            extra = self._fetch_morechildren_comments(
                post_id=post_id,
                child_ids=more_ids,
                sort=sort,
                limit=remaining,
            )
            base_comments.extend(extra)

        # Deduplicate by comment id while preserving order.
        deduped: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        for comment in base_comments:
            comment_id = comment.get("id", "")
            if comment_id and comment_id in seen_ids:
                continue
            if comment_id:
                seen_ids.add(comment_id)
            deduped.append(comment)

        if comments_limit is not None:
            return deduped[:comments_limit]
        return deduped

    def scrape_post_details(
        self,
        permalink: str,
        sort: str = "new",
        top_level_limit: int = 500,
    ) -> Optional[Dict[str, Any]]:
        if not permalink.startswith("/"):
            permalink = "/" + permalink

        url = f"https://www.reddit.com{permalink}.json"
        params = {"sort": sort, "limit": min(max(top_level_limit, 1), 500), "depth": 10}
        data = self._make_request(url, params=params)
        if not data or len(data) < 2:
            return None

        post_listing = data[0]
        comments_listing = data[1]

        if "data" not in post_listing or "children" not in post_listing["data"]:
            return None

        post_data = post_listing["data"]["children"][0]["data"]

        comments: List[Dict[str, Any]] = []
        more_comment_ids: List[str] = []
        if "data" in comments_listing:
            children = comments_listing["data"].get("children", [])
            comments, more_comment_ids = self._extract_comments(children)

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
            "more_comment_ids": more_comment_ids,
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
        enable_continue_prompt: bool = True,
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
                account_unavailable = int(account_age_days is None and user_karma <= 0 and comment_karma <= 0)

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
                    "is_account_unavailable": account_unavailable,
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
                    if not enable_continue_prompt:
                        return False
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
        target_comments: Optional[int] = 700,
        sort: str = "new",
    ) -> pd.DataFrame:
        self.target_comments = target_comments
        permalink = post_url.split("reddit.com")[1] if "reddit.com" in post_url else post_url

        print(f"Parsing post from: {post_url}")
        post_details = self.scrape_post_details(permalink, sort=sort)
        if not post_details:
            print("Failed to scrape post details")
            return pd.DataFrame()

        comments = self._collect_post_comments(
            post_details=post_details,
            comments_limit=target_comments,
            sort=sort,
        )
        print(f"Collected {len(comments)} comments from post listing.")

        self.process_comments(
            comments=comments,
            post_author=post_details.get("author", ""),
            post_time=post_details.get("created_utc", time.time()),
            post_title=post_details.get("title", ""),
            post_url=post_url,
            post_id=post_details.get("id", ""),
            subreddit=post_details.get("subreddit", ""),
            enable_continue_prompt=False,
        )

        df = pd.DataFrame(self.collected_data)
        self.save_to_csv()
        self.save_comment_texts()
        return df

    def parse_subreddit_comments_by_date_range(
        self,
        subreddit_name: str,
        start_date: str,
        end_date: str,
        comments_per_post_limit: Optional[int] = 300,
        category: str = "new",
        sort_comments: str = "new",
    ) -> pd.DataFrame:
        self.target_comments = None
        print(f"Parsing subreddit by date range: r/{subreddit_name}")
        print(f"Date range: {start_date} .. {end_date} (UTC)")
        print(f"Comments per post limit: {'ALL' if comments_per_post_limit is None else comments_per_post_limit}")
        print("-" * 60)

        posts = self.fetch_subreddit_posts_by_date_range(
            subreddit_name=subreddit_name,
            start_date=start_date,
            end_date=end_date,
            category=category,
        )
        if not posts:
            print("No posts found in that range.")
            return pd.DataFrame()

        print(f"Found {len(posts)} posts in range.")

        for index, post in enumerate(posts, start=1):
            title = post.get("title", "")[:70]
            print(f"\n[{index}/{len(posts)}] {title}...")
            post_details = self.scrape_post_details(post.get("permalink", ""), sort=sort_comments)
            if not post_details:
                print("  Could not fetch post details")
                continue

            comments = self._collect_post_comments(
                post_details=post_details,
                comments_limit=comments_per_post_limit,
                sort=sort_comments,
            )
            print(f"  Collected comments from post: {len(comments)}")

            initial = len(self.collected_data)
            self.process_comments(
                comments=comments,
                post_author=post_details.get("author", ""),
                post_time=post_details.get("created_utc", time.time()),
                post_title=post_details.get("title", ""),
                post_url=post.get("permalink", ""),
                post_id=post_details.get("id", ""),
                subreddit=post_details.get("subreddit", subreddit_name),
                enable_continue_prompt=False,
            )
            added = len(self.collected_data) - initial
            print(f"  Added valid comments: {added}")

        df = pd.DataFrame(self.collected_data)
        print("\n" + "=" * 60)
        print(f"Date-range parsing complete: {len(df)} comments from {df['username'].nunique() if not df.empty else 0} users")
        print("=" * 60 + "\n")

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
