# -*- coding: utf-8 -*-
"""Compatibility wrapper for legacy imports.

Legacy code importing from `main_parser` continues to work, while the
implementation now lives in `reddit_bots.parser.reddit_parser`.
"""

from reddit_bots.parser.reddit_parser import RedditParser, SentimentAnalyzer, is_bot

__all__ = ["RedditParser", "SentimentAnalyzer", "is_bot"]
