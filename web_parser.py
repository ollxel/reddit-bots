#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Parser - запускается из Node.js сервера для парсинга Reddit
"""
import sys
import os

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import json
import argparse
import pandas as pd
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional, Any

# === Bot Detection Logic (Improved) ===
def is_bot(user_karma: int, account_age_days: float, reply_delay_seconds: int, 
           sentiment_score: Optional[float], comment_text: str = "", comment_score: int = 0) -> bool:
    """
    Improved bot detection based on multiple factors
    """
    bot_score = 0
    human_score = 0
    
    # SUSPICIOUS FACTORS
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
    
    # HUMAN FACTORS
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


# === OpenRouter Sentiment Analyzer ===

class SentimentAnalyzer:
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

    def __init__(self, api_key: str = None, model: str = "arcee-ai/trinity-large-preview:free",
                 min_request_interval: float = 1.5):
        self.api_key = api_key
        self.model = model
        self.interval = min_request_interval
        self._last_call = 0.0
        self.request_count = 0

    def _wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_call = time.time()

    def score(self, comment_text: str, username: str = "unknown", max_retries: int = 3) -> Optional[float]:
        if not self.api_key:
            return None
        
        self.request_count += 1
        if self.request_count % 10 == 0:
            print(f"  [SENTIMENT] Processed {self.request_count} comments...")
            
        prompt = self.PROMPT_TEMPLATE.format(
            username=username,
            comment_text=comment_text[:1500]
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
                resp = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=20)
                resp.raise_for_status()
                raw = (resp.json()["choices"][0]["message"]["content"].strip().replace(",", "."))
                return max(-1.0, min(1.0, float(raw)))
            except (ValueError, KeyError, Exception):
                if attempt == max_retries - 1:
                    pass
                wait = 2 ** attempt
                time.sleep(wait)

        return None


# === Reddit Parser ===

class RedditParser:
    def __init__(self, user_agent: str = "RedditParserWeb/1.0", run_sentiment: bool = True):
        self.collected_data: List[Dict] = []
        self.processed_users: set = set()
        self.target_comments: Optional[int] = None
        
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.last_request_time = 0.0
        self.min_request_interval = 5
        
        self.run_sentiment = run_sentiment
        self.sentiment = None

    def set_sentiment(self, api_key: str, model: str):
        if api_key:
            self.sentiment = SentimentAnalyzer(api_key=api_key, model=model)
            print(f"[OK] SentimentAnalyzer ready (model: {model})")

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
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
                time.sleep(wait)
        return None

    def get_user_info(self, username: str) -> Dict[str, Any]:
        url = f"https://www.reddit.com/user/{username}/about.json"
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

            link_karma = int(ud.get("link_karma", 0) or 0)
            comment_karma = int(ud.get("comment_karma", 0) or 0)
            return {
                "account_age_days": age_days,
                "user_karma": link_karma + comment_karma,
                "comment_karma": comment_karma,
            }

        return {"account_age_days": None, "user_karma": 0, "comment_karma": 0}

    def fetch_subreddit_posts(self, subreddit_name: str, limit: int = 25,
                              category: str = "hot", time_filter: str = "week") -> List[Dict]:
        url = f"https://www.reddit.com/r/{subreddit_name}/{category}.json"
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
                    "title": pd_.get("title", ""),
                    "author": pd_.get("author", ""),
                    "permalink": pd_.get("permalink", ""),
                    "created_utc": pd_.get("created_utc", 0),
                    "score": pd_.get("score", 0),
                    "num_comments": pd_.get("num_comments", 0),
                    "id": pd_.get("id", ""),
                })
        return posts

    def _extract_comments(self, comments_data: List, all_comments: List = None) -> List[Dict]:
        if all_comments is None:
            all_comments = []
        for item in comments_data:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind", "")
            data = item.get("data", {})
            if kind == "t1":
                all_comments.append({
                    "id": data.get("id", ""),
                    "author": data.get("author", ""),
                    "body": data.get("body", ""),
                    "score": data.get("score", 0),
                    "created_utc": data.get("created_utc", 0),
                })
                replies = data.get("replies", "")
                if isinstance(replies, dict) and "data" in replies:
                    self._extract_comments(replies["data"].get("children", []), all_comments)
            elif kind == "Listing":
                self._extract_comments(data.get("children", []), all_comments)
        return all_comments

    def scrape_post_details(self, permalink: str) -> Optional[Dict]:
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

        pd_ = post_listing["data"]["children"][0]["data"]
        comments = []
        if "data" in comments_listing:
            children = comments_listing["data"].get("children", [])
            comments = self._extract_comments(children)

        return {
            "title": pd_.get("title", ""),
            "author": pd_.get("author", ""),
            "created_utc": pd_.get("created_utc", 0),
            "score": pd_.get("score", 0),
            "num_comments": pd_.get("num_comments", 0),
            "selftext": pd_.get("selftext", ""),
            "url": pd_.get("url", ""),
            "permalink": pd_.get("permalink", ""),
            "comments": comments,
        }

    def process_comments(self, comments: List[Dict], post_author: str,
                         post_time: float, post_title: Optional[str] = None,
                         post_url: Optional[str] = None,
                         target_comments: int = 700) -> bool:
        for comment in comments:
            author = comment.get("author")
            body = comment.get("body", "").strip()
            comment_score = comment.get("score", 0)

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
                    continue

                user_karma = user_info["user_karma"]
                comment_karma = user_info["comment_karma"]
                comment_created = comment.get("created_utc", 0)
                reply_delay_secs = int(comment_created - post_time) if comment_created else 0

                # Sentiment
                sentiment_score = None
                if self.run_sentiment and self.sentiment:
                    sentiment_score = self.sentiment.score(body, username=author)

                # Bot detection
                is_bot_user = is_bot(
                    user_karma, 
                    account_age_days, 
                    reply_delay_secs, 
                    sentiment_score,
                    comment_text=body,
                    comment_score=comment_score
                )

                record = {
                    "reply_delay_seconds": reply_delay_secs,
                    "user_karma": user_karma,
                    "account_age_days": account_age_days,
                    "comment_karma": comment_karma,
                    "sentiment_score": sentiment_score,
                    "username": author,
                    "comment_id": comment.get("id", ""),
                    "comment_score": comment_score,
                    "is_bot": is_bot_user,
                }
                if post_title:
                    record["post_title"] = post_title
                if post_url:
                    record["post_url"] = post_url

                self.collected_data.append(record)
                self.processed_users.add(author)

                total = len(self.collected_data)
                if total >= self.target_comments:
                    return False

            except Exception as e:
                continue

        return True

    def parse_subreddit(self, subreddit_name: str, posts_limit: int = 10,
                       category: str = "hot", time_filter: str = "week",
                       target_comments: int = 100) -> pd.DataFrame:
        self.target_comments = target_comments
        print("Sending POST requests to Reddit API...")
        print(f"Parsing subreddit: r/{subreddit_name}")
        
        posts = self.fetch_subreddit_posts(
            subreddit_name, limit=posts_limit,
            category=category, time_filter=time_filter
        )
        if not posts:
            print("No posts found")
            return pd.DataFrame()

        print(f"Found {len(posts)} posts")
        print("Parsing comments...")

        for i, post in enumerate(posts, 1):
            title = post.get("title", "")[:60]
            print(f"\n[{i}/{len(posts)}] {title}...")

            permalink = post.get("permalink", "")
            post_details = self.scrape_post_details(permalink)
            if not post_details:
                continue

            post_author = post_details.get("author")
            post_time = post_details.get("created_utc", time.time())
            post_title = post_details.get("title", "")
            post_url = post.get("permalink", "")
            comments = post_details.get("comments", [])

            print(f"  Found {len(comments)} comments")
            
            if self.sentiment:
                print("Analyzing sentiment...")
            
            should_cont = self.process_comments(
                comments, post_author, post_time,
                post_title, post_url
            )
            print(f"  Added: {len(self.collected_data)}")

            if not should_cont:
                break

        df = pd.DataFrame(self.collected_data)
        print(f"\nParsing complete! Collected {len(df)} unique users")
        
        if not df.empty and "is_bot" in df.columns:
            bots = df["is_bot"].sum()
            humans = len(df) - bots
            print(f"Bots detected: {bots}")
            print(f"Humans: {humans}")
        
        return df

    def parse_post(self, post_url: str, target_comments: int = 100) -> pd.DataFrame:
        self.target_comments = target_comments
        print("Sending POST requests to Reddit API...")
        
        if "reddit.com" in post_url:
            permalink = post_url.split("reddit.com")[1]
        else:
            permalink = post_url

        print(f"Parsing post: {post_url}")
        print("Parsing comments...")
        
        post_details = self.scrape_post_details(permalink)
        if not post_details:
            print("Failed to scrape post details")
            return pd.DataFrame()

        if self.sentiment:
            print("Analyzing sentiment...")
            
        self.process_comments(
            post_details.get("comments", []),
            post_details.get("author"),
            post_details.get("created_utc", time.time()),
            post_details.get("title", ""),
            post_url,
            target_comments,
        )
        
        df = pd.DataFrame(self.collected_data)
        print(f"\nParsing complete! Collected {len(df)} unique users")
        
        if not df.empty and "is_bot" in df.columns:
            bots = df["is_bot"].sum()
            humans = len(df) - bots
            print(f"Bots detected: {bots}")
            print(f"Humans: {humans}")
        
        return df

    def save_results(self, filename: str = "web_results.json") -> str:
        result = {
            "total_users": len(self.collected_data),
            "timestamp": datetime.now().isoformat(),
            "users": self.collected_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Saving data to {filename}")
        return filename


def main():
    parser = argparse.ArgumentParser(description='Web Parser for Reddit')
    parser.add_argument('--api-key', type=str, default='', help='OpenRouter API Key')
    parser.add_argument('--model', type=str, default='arcee-ai/trinity-large-preview:free', help='Model name')
    parser.add_argument('--mode', type=str, required=True, choices=['subreddit', 'post'], help='Parse mode')
    parser.add_argument('--subreddit', type=str, default='', help='Subreddit name')
    parser.add_argument('--post-url', type=str, default='', help='Post URL')
    parser.add_argument('--target', type=int, default=100, help='Target number of comments')
    
    args = parser.parse_args()
    
    p = RedditParser(run_sentiment=bool(args.api_key))
    
    if args.api_key:
        p.set_sentiment(args.api_key, args.model)
    
    if args.mode == 'subreddit':
        df = p.parse_subreddit(
            args.subreddit,
            target_comments=args.target
        )
    else:
        df = p.parse_post(
            args.post_url,
            target_comments=args.target
        )
    
    if not df.empty:
        p.save_results()


if __name__ == "__main__":
    main()
