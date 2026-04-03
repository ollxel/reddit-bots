"""Retro terminal web interface for reddit-bots."""

from __future__ import annotations

import html
import io
import json
import os
import threading
import traceback
import webbrowser
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import pandas as pd

from reddit_bots.analysis.account_features import build_account_features
from reddit_bots.models.bot_classifier import AccountBotClassifier
from reddit_bots.parser.reddit_parser import RedditParser, SentimentAnalyzer


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


WEB_TEXT: Dict[str, Dict[str, str]] = {
    "en": {
        "title": "REDDIT-BOTS | Web Mode",
        "menu_title": "[ MODE SELECTION ]",
        "menu_1": "1) Default mode",
        "menu_2": "2) Advanced mode",
        "menu_3": "3) Web mode (current)",
        "panel_title": "[ DEFAULT PIPELINE ]",
        "panel_desc": "Single run: parse -> account features -> analyzer -> suspicious accounts",
        "parse_mode": "parse mode",
        "mode_classic": "subreddit (classic)",
        "mode_post": "post URL",
        "mode_range": "subreddit by date range",
        "configure": "CONFIGURE PARSE INPUT",
        "sentiment": "enable sentiment",
        "api_key": "OpenRouter API key (optional)",
        "model": "OpenRouter model",
        "training_csv": "training CSV",
        "threshold": "suspicious threshold",
        "run": "RUN DEFAULT PIPELINE",
        "flow_hint": "Flow: 1) select parse mode  2) configure parse input  3) run pipeline",
        "dialog_title": "[ PARSE INPUT CONFIGURATION ]",
        "save": "SAVE",
        "close": "CLOSE",
        "classic_subreddit": "subreddit",
        "classic_category": "category",
        "classic_time_filter": "time_filter (for top)",
        "classic_posts_limit": "posts_limit",
        "classic_target_comments": "target_comments",
        "post_url": "post URL",
        "post_parse_all": "parse all comments",
        "post_target_comments": "target_comments (if parse all = false)",
        "range_subreddit": "subreddit",
        "range_start": "start_date",
        "range_end": "end_date",
        "range_comments_per_post": "comments_per_post_limit (0 = all)",
        "range_category": "category",
        "status_label": "[ STATUS ]",
        "ready": "ready",
        "running": "running",
        "done": "done",
        "error": "error",
    },
    "ru": {
        "title": "REDDIT-BOTS | Веб режим",
        "menu_title": "[ ВЫБОР РЕЖИМА ]",
        "menu_1": "1) Режим по умолчанию",
        "menu_2": "2) Продвинутый режим",
        "menu_3": "3) Веб режим (текущий)",
        "panel_title": "[ ПАЙПЛАЙН ПО УМОЛЧАНИЮ ]",
        "panel_desc": "Один запуск: парсинг -> фичи аккаунтов -> анализатор -> подозрительные аккаунты",
        "parse_mode": "режим парсинга",
        "mode_classic": "сабреддит (классика)",
        "mode_post": "ссылка на пост",
        "mode_range": "сабреддит по диапазону дат",
        "configure": "НАСТРОИТЬ ВВОД ПАРСИНГА",
        "sentiment": "включить sentiment",
        "api_key": "OpenRouter API ключ (необязательно)",
        "model": "модель OpenRouter",
        "training_csv": "обучающий CSV",
        "threshold": "порог подозрительности",
        "run": "ЗАПУСТИТЬ ПАЙПЛАЙН",
        "flow_hint": "Порядок: 1) выберите режим парсинга  2) настройте ввод  3) запустите пайплайн",
        "dialog_title": "[ НАСТРОЙКА ПАРСИНГА ]",
        "save": "СОХРАНИТЬ",
        "close": "ЗАКРЫТЬ",
        "classic_subreddit": "сабреддит",
        "classic_category": "категория",
        "classic_time_filter": "time_filter (для top)",
        "classic_posts_limit": "лимит постов",
        "classic_target_comments": "лимит комментариев",
        "post_url": "ссылка на пост",
        "post_parse_all": "парсить все комментарии",
        "post_target_comments": "лимит комментариев (если не все)",
        "range_subreddit": "сабреддит",
        "range_start": "дата начала",
        "range_end": "дата конца",
        "range_comments_per_post": "лимит комментариев на пост (0 = все)",
        "range_category": "категория",
        "status_label": "[ СТАТУС ]",
        "ready": "готово",
        "running": "выполняется",
        "done": "завершено",
        "error": "ошибка",
    },
}


class WebModeServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8080, language: str = "en"):
        self.host = host
        self.port = port
        self.language = language if language in WEB_TEXT else "en"
        self._action_lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None

    def _t(self, key: str) -> str:
        return WEB_TEXT[self.language].get(key, key)

    @staticmethod
    def _read_csv_flexible(path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path)

    @staticmethod
    def _preview_df(df: pd.DataFrame, limit: int = 25) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"columns": [], "rows": [], "total": 0}

        sample = df.head(limit).copy()
        for col in sample.columns:
            if pd.api.types.is_float_dtype(sample[col]):
                sample[col] = sample[col].round(6)

        sample = sample.fillna("")
        return {
            "columns": list(sample.columns),
            "rows": sample.to_dict(orient="records"),
            "total": int(len(df)),
        }

    def _build_parser(self, payload: Dict[str, Any]) -> RedditParser:
        parser = RedditParser(
            user_agent="RedditDataCollector/2.0 (WebMode)",
            run_sentiment=_to_bool(payload.get("enable_sentiment"), default=False),
            unique_users_only=False,
            language=self.language,
        )

        if parser.run_sentiment:
            api_key = str(payload.get("api_key") or os.getenv("OPENROUTER_API_KEY", "")).strip()
            model = str(payload.get("model") or "arcee-ai/trinity-large-preview:free").strip()
            if api_key:
                parser.sentiment = SentimentAnalyzer(api_key=api_key, model=model)
            else:
                parser.run_sentiment = False
                print("OpenRouter API key not provided. Sentiment disabled.")

        return parser

    def _parse_by_mode(self, parser: RedditParser, payload: Dict[str, Any]) -> pd.DataFrame:
        parse_mode = str(payload.get("parse_mode") or "subreddit_classic").strip()

        if parse_mode == "post_url":
            post_url = str(payload.get("post_url") or "").strip()
            if not post_url:
                raise ValueError("post_url is required")
            parse_all = _to_bool(payload.get("post_parse_all"), default=True)
            target_comments = None if parse_all else max(1, _to_int(payload.get("post_target_comments"), 500))
            return parser.parse_post_comments(post_url=post_url, target_comments=target_comments, sort="new")

        if parse_mode == "subreddit_range":
            subreddit = str(payload.get("range_subreddit") or "").strip()
            start_date = str(payload.get("range_start") or "").strip()
            end_date = str(payload.get("range_end") or "").strip()
            if not subreddit or not start_date or not end_date:
                raise ValueError("range_subreddit, range_start, range_end are required")

            comments_per_post = _to_int(payload.get("range_comments_per_post"), 300)
            comments_per_post_limit = None if comments_per_post <= 0 else comments_per_post
            category = str(payload.get("range_category") or "new").strip() or "new"

            return parser.parse_subreddit_comments_by_date_range(
                subreddit_name=subreddit,
                start_date=start_date,
                end_date=end_date,
                comments_per_post_limit=comments_per_post_limit,
                category=category,
                sort_comments="new",
            )

        subreddit = str(payload.get("classic_subreddit") or "AskReddit").strip()
        category = str(payload.get("classic_category") or "hot").strip() or "hot"
        time_filter = str(payload.get("classic_time_filter") or "week").strip() or "week"
        posts_limit = max(1, _to_int(payload.get("classic_posts_limit"), 10))
        target_comments = max(1, _to_int(payload.get("classic_target_comments"), 300))

        return parser.parse_subreddit_comments(
            subreddit_name=subreddit,
            posts_limit=posts_limit,
            category=category,
            time_filter=time_filter,
            target_comments=target_comments,
            enable_continue_prompt=False,
        )

    def _action_run_default_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        parser = self._build_parser(payload)
        parsed_df = self._parse_by_mode(parser, payload)
        if parsed_df is None or parsed_df.empty:
            raise ValueError("No comments parsed. Check parse input settings.")

        parsed_df.to_csv("parsed_comments.csv", index=False, float_format="%.6f")

        features_df = build_account_features(parsed_df)
        features_df.to_csv("account_features.csv", index=False, float_format="%.6f")

        training_csv = str(payload.get("training_csv") or "reddit_dead_internet_analysis_2026.csv").strip()
        if not os.path.exists(training_csv):
            raise FileNotFoundError(f"Training CSV not found: {training_csv}")

        classifier = AccountBotClassifier()
        classifier.train(training_csv)
        analysis_df = classifier.run_analysis(features_df, "account_analysis.csv")

        threshold = _to_float(payload.get("suspicious_threshold"), 0.3)
        suspicious_df = analysis_df[analysis_df["bot_probability"] >= threshold].copy()
        suspicious_df = suspicious_df.sort_values("bot_probability", ascending=False)

        if suspicious_df.empty:
            preview_df = analysis_df[["username", "bot_probability", "risk_level"]].head(25)
        else:
            preview_df = suspicious_df[["username", "bot_probability", "risk_level"]].head(25)

        users = int(parsed_df["username"].nunique()) if "username" in parsed_df.columns else 0
        message = (
            f"Pipeline done: comments={len(parsed_df)}, users={users}, "
            f"accounts={len(analysis_df)}, suspicious={len(suspicious_df)}. "
            "Files: parsed_comments.csv, account_features.csv, account_analysis.csv"
        )

        return {
            "message": message,
            "preview": self._preview_df(preview_df),
        }

    def _run_action(self, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        acquired = self._action_lock.acquire(blocking=False)
        if not acquired:
            return {
                "ok": False,
                "message": "Another action is already running. Wait for completion.",
                "log": "",
                "preview": {"columns": [], "rows": [], "total": 0},
            }

        log_stream = io.StringIO()
        try:
            with redirect_stdout(log_stream), redirect_stderr(log_stream):
                data = fn() or {}
            return {
                "ok": True,
                "message": data.get("message", "Action completed."),
                "log": log_stream.getvalue(),
                "preview": data.get("preview", {"columns": [], "rows": [], "total": 0}),
            }
        except Exception as exc:
            traceback.print_exc(file=log_stream)
            return {
                "ok": False,
                "message": f"{type(exc).__name__}: {exc}",
                "log": log_stream.getvalue(),
                "preview": {"columns": [], "rows": [], "total": 0},
            }
        finally:
            self._action_lock.release()

    def handle_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action = str(payload.get("action") or "").strip()
        if action != "run_default_pipeline":
            return {
                "ok": False,
                "message": f"Unknown action: {action}",
                "log": "",
                "preview": {"columns": [], "rows": [], "total": 0},
            }

        return self._run_action(lambda: self._action_run_default_pipeline(payload))

    def _build_handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, body_text: str, status: int = 200) -> None:
                body = body_text.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/":
                    self._send_html(_render_index_html(app.host, app.port, app.language))
                    return
                if path == "/health":
                    self._send_json({"ok": True, "status": "up"})
                    return
                if path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                self._send_json({"ok": False, "message": "Not found"}, status=404)

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path != "/api/action":
                    self._send_json({"ok": False, "message": "Not found"}, status=404)
                    return

                length = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                except json.JSONDecodeError:
                    self._send_json({"ok": False, "message": "Invalid JSON payload"}, status=400)
                    return

                response = app.handle_action(payload)
                status = 200 if response.get("ok") else 400
                self._send_json(response, status=status)

            def log_message(self, _format: str, *args: Any) -> None:
                _ = args
                return

        return Handler

    def run(self, open_browser: bool = True) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        url = f"http://{self.host}:{self.port}"
        print("=" * 68)
        print("WEB MODE")
        print("=" * 68)
        print(f"Server started at: {url}")
        print("Stop with Ctrl+C")

        if open_browser:
            webbrowser.open(url)

        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nWeb mode stopped.")
        finally:
            self._httpd.server_close()


def run_web_interface(
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = True,
    language: str = "en",
) -> None:
    server = WebModeServer(host=host, port=port, language=language)
    server.run(open_browser=open_browser)


def _render_index_html(host: str, port: int, language: str = "en") -> str:
    lang = language if language in WEB_TEXT else "en"
    t = WEB_TEXT[lang]

    template = """<!doctype html>
<html lang="__LANG_HTML__">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg0: #0b0f19;
      --bg1: #111827;
      --text: #e5e7eb;
      --red: #d55a2a;
      --amber: #e2b04f;
      --cyan: #6aa1b8;
      --line: #334155;
      --danger: #c2412f;
      --ok: #9ad57a;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "JetBrains Mono", "Fira Code", "IBM Plex Mono", Consolas, "Courier New", monospace;
      color: var(--text);
      background: linear-gradient(180deg, var(--bg0), var(--bg1));
      min-height: 100vh;
      letter-spacing: 0.02em;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: repeating-linear-gradient(
        to bottom,
        rgba(255, 255, 255, 0.025),
        rgba(255, 255, 255, 0.025) 1px,
        transparent 2px,
        transparent 4px
      );
      opacity: 0.2;
    }

    .wrap {
      width: min(1200px, 96vw);
      margin: 0 auto;
      padding: 16px;
      position: relative;
      z-index: 1;
    }

    .banner {
      color: var(--red);
      white-space: pre;
      font-size: 12px;
      line-height: 1.2;
      margin: 0 0 12px 0;
      text-shadow: 0 0 10px rgba(213, 90, 42, 0.25);
      overflow-x: auto;
    }

    .sep {
      color: var(--cyan);
      white-space: pre;
      margin: 10px 0;
      overflow-x: auto;
    }

    .menu {
      color: var(--amber);
      margin-bottom: 16px;
      white-space: pre;
      overflow-x: auto;
    }

    .panel {
      border: 1px solid var(--line);
      background: rgba(3, 8, 20, 0.55);
      padding: 12px;
    }

    .panel h3 {
      margin: 0 0 8px 0;
      color: var(--red);
      font-size: 14px;
      font-weight: 700;
    }

    .panel p {
      margin: 0 0 10px 0;
      color: var(--amber);
      font-size: 12px;
    }

    .hint {
      color: var(--cyan);
      font-size: 12px;
      margin: 0 0 12px 0;
      white-space: pre-wrap;
    }

    label {
      display: block;
      margin-bottom: 6px;
      color: var(--cyan);
      font-size: 12px;
    }

    input, select {
      width: 100%;
      border: 1px solid #334155;
      background: #0a1222;
      color: var(--text);
      padding: 6px 8px;
      border-radius: 0;
      font: inherit;
      margin-top: 2px;
      margin-bottom: 8px;
    }

    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(180px, 1fr));
      gap: 8px;
    }

    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 4px 0 8px;
      color: var(--amber);
      font-size: 12px;
    }

    .check input {
      width: auto;
      margin: 0;
    }

    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    button {
      border: 1px solid var(--red);
      background: #1c0f0a;
      color: var(--text);
      padding: 8px 10px;
      border-radius: 0;
      font: inherit;
      cursor: pointer;
      min-width: 180px;
    }

    button:hover { background: #2a140d; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }

    dialog {
      border: 1px solid var(--line);
      background: #050a16;
      color: var(--text);
      padding: 12px;
      width: min(760px, 96vw);
    }

    dialog::backdrop { background: rgba(0,0,0,0.65); }

    .terminal {
      margin-top: 14px;
      border: 1px solid var(--line);
      background: #050a16;
      padding: 10px;
    }

    .status { color: var(--amber); margin-bottom: 8px; }
    .status-ok { color: var(--ok); }
    .status-err { color: var(--danger); }

    .cursor::after {
      content: "_";
      animation: blink 0.85s steps(1) infinite;
    }

    @keyframes blink { 50% { opacity: 0; } }

    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #d1d5db;
      max-height: 320px;
      overflow: auto;
      border: 1px solid #1f2937;
      background: #020611;
      padding: 8px;
    }

    .table-wrap {
      margin-top: 10px;
      border: 1px solid #1f2937;
      overflow: auto;
      max-height: 360px;
      background: #030917;
    }

    table {
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
    }

    th, td {
      border: 1px solid #1f2937;
      padding: 6px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }

    th {
      color: var(--red);
      background: #0c1427;
      position: sticky;
      top: 0;
    }

    .mode-box { display: none; }
    .mode-box.active { display: block; }

    @media (max-width: 980px) {
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
<pre class="banner">╦═╗╔═╗╔╦╗╔╦╗╦╔╦╗  ╔╗ ╔═╗╔╦╗╔═╗
╠╦╝║╣  ║║ ║║║ ║───╠╩╗║ ║ ║ ╚═╗
╩╚═╚═╝═╩╝═╩╝╩ ╩   ╚═╝╚═╝ ╩ ╚═╝

REDDIT-BOTS | WEB MODE</pre>

<div class="sep">====================================================================</div>
<div class="menu">__MENU_TITLE__
__MENU_1__
__MENU_2__
__MENU_3__

host: __HOST__
port: __PORT__</div>
<div class="sep">--------------------------------------------------------------------</div>

<form id="pipelineForm" class="panel">
  <h3>__PANEL_TITLE__</h3>
  <p>__PANEL_DESC__</p>
  <div class="hint">> __FLOW_HINT__</div>

  <div class="row">
    <label>__PARSE_MODE__
      <select name="parse_mode" id="parseMode">
        <option value="subreddit_classic">__MODE_CLASSIC__</option>
        <option value="post_url">__MODE_POST__</option>
        <option value="subreddit_range">__MODE_RANGE__</option>
      </select>
    </label>
    <label>__TRAINING_CSV__
      <input name="training_csv" value="reddit_dead_internet_analysis_2026.csv" />
    </label>
  </div>

  <div class="row">
    <label>__THRESHOLD__
      <input name="suspicious_threshold" value="0.30" />
    </label>
    <label>__MODEL__
      <input name="model" value="arcee-ai/trinity-large-preview:free" />
    </label>
  </div>

  <label>__API_KEY__
    <input name="api_key" autocomplete="off" />
  </label>

  <div class="check"><input type="checkbox" name="enable_sentiment" /> __SENTIMENT__</div>

  <div class="actions">
    <button type="button" id="openConfigBtn">__CONFIGURE__</button>
    <button type="submit">__RUN__</button>
  </div>

  <dialog id="modeDialog">
    <h3>__DIALOG_TITLE__</h3>

    <div id="modeClassic" class="mode-box">
      <div class="row">
        <label>__CLASSIC_SUBREDDIT__ <input name="classic_subreddit" value="AskReddit" /></label>
        <label>__CLASSIC_CATEGORY__
          <select name="classic_category">
            <option>hot</option><option>new</option><option>top</option><option>rising</option>
          </select>
        </label>
      </div>
      <div class="row">
        <label>__CLASSIC_TIME_FILTER__ <input name="classic_time_filter" value="week" /></label>
        <label>__CLASSIC_POSTS_LIMIT__ <input name="classic_posts_limit" value="10" /></label>
      </div>
      <label>__CLASSIC_TARGET_COMMENTS__ <input name="classic_target_comments" value="300" /></label>
    </div>

    <div id="modePost" class="mode-box">
      <label>__POST_URL__ <input name="post_url" placeholder="https://www.reddit.com/r/..." /></label>
      <div class="check"><input type="checkbox" name="post_parse_all" checked /> __POST_PARSE_ALL__</div>
      <label>__POST_TARGET_COMMENTS__ <input name="post_target_comments" value="500" /></label>
    </div>

    <div id="modeRange" class="mode-box">
      <div class="row">
        <label>__RANGE_SUBREDDIT__ <input name="range_subreddit" value="AskReddit" /></label>
        <label>__RANGE_CATEGORY__
          <select name="range_category">
            <option selected>new</option><option>hot</option><option>top</option><option>rising</option>
          </select>
        </label>
      </div>
      <div class="row">
        <label>__RANGE_START__ <input type="date" name="range_start" /></label>
        <label>__RANGE_END__ <input type="date" name="range_end" /></label>
      </div>
      <label>__RANGE_COMMENTS_PER_POST__ <input name="range_comments_per_post" value="300" /></label>
    </div>

    <div class="actions">
      <button type="button" id="saveConfigBtn">__SAVE__</button>
      <button type="button" id="closeConfigBtn">__CLOSE__</button>
    </div>
  </dialog>
</form>

<div class="sep">~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~</div>
<div class="terminal">
  <div class="status" id="statusLine">__STATUS_LABEL__ <span id="statusText">__READY__</span> <span class="cursor"></span></div>
  <pre id="log">$ __READY__\n</pre>
  <div id="preview" class="table-wrap"></div>
</div>

  </div>

<script>
(function () {
  var form = document.getElementById('pipelineForm');
  var parseMode = document.getElementById('parseMode');
  var openBtn = document.getElementById('openConfigBtn');
  var saveBtn = document.getElementById('saveConfigBtn');
  var closeBtn = document.getElementById('closeConfigBtn');
  var dialog = document.getElementById('modeDialog');
  var statusText = document.getElementById('statusText');
  var statusLine = document.getElementById('statusLine');
  var logEl = document.getElementById('log');
  var previewEl = document.getElementById('preview');

  var TXT = {
    running: __JS_RUNNING__,
    ready: __JS_READY__,
    done: __JS_DONE__,
    error: __JS_ERROR__
  };

  function setActiveMode() {
    var mode = parseMode.value;
    var boxes = [
      { id: 'modeClassic', key: 'subreddit_classic' },
      { id: 'modePost', key: 'post_url' },
      { id: 'modeRange', key: 'subreddit_range' }
    ];

    boxes.forEach(function (item) {
      var box = document.getElementById(item.id);
      var active = mode === item.key;
      box.classList.toggle('active', active);
      Array.prototype.slice.call(box.querySelectorAll('input, select')).forEach(function (el) {
        el.disabled = !active;
      });
    });
  }

  function setStatus(kind, message) {
    statusLine.classList.remove('status-ok', 'status-err');
    if (kind === 'ok') {
      statusLine.classList.add('status-ok');
      statusText.textContent = TXT.done;
    } else if (kind === 'err') {
      statusLine.classList.add('status-err');
      statusText.textContent = TXT.error;
    } else if (kind === 'run') {
      statusText.textContent = TXT.running;
    } else {
      statusText.textContent = TXT.ready;
    }
    if (message) {
      logEl.textContent = message;
    }
  }

  function collectPayload() {
    var payload = { action: 'run_default_pipeline' };
    var elements = Array.prototype.slice.call(form.elements);
    elements.forEach(function (el) {
      if (!el.name || el.disabled) return;
      if (el.type === 'checkbox') {
        payload[el.name] = el.checked;
      } else {
        payload[el.name] = el.value;
      }
    });
    return payload;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderPreview(preview) {
    if (!preview || !preview.columns || preview.columns.length === 0 || !preview.rows) {
      previewEl.innerHTML = '';
      return;
    }

    var cols = preview.columns;
    var rows = preview.rows;
    var html = '<table><thead><tr>';
    cols.forEach(function (col) { html += '<th>' + escapeHtml(col) + '</th>'; });
    html += '</tr></thead><tbody>';

    rows.forEach(function (row) {
      html += '<tr>';
      cols.forEach(function (col) {
        var value = row[col] === null || row[col] === undefined ? '' : row[col];
        html += '<td>' + escapeHtml(value) + '</td>';
      });
      html += '</tr>';
    });

    html += '</tbody></table>';
    previewEl.innerHTML = html;
  }

  function setBusy(busy) {
    Array.prototype.slice.call(form.querySelectorAll('button')).forEach(function (btn) {
      btn.disabled = busy;
    });
  }

  parseMode.addEventListener('change', function () {
    setActiveMode();
    if (!dialog.open) {
      dialog.showModal();
    }
  });
  openBtn.addEventListener('click', function () {
    setActiveMode();
    dialog.showModal();
  });
  saveBtn.addEventListener('click', function () { dialog.close(); });
  closeBtn.addEventListener('click', function () { dialog.close(); });

  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    var payload = collectPayload();

    setBusy(true);
    setStatus('run', '$ running: ' + payload.action + '\\n');

    fetch('/api/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(function (resp) {
        return resp.json().catch(function () {
          return { ok: false, message: 'Invalid JSON response', log: '', preview: {} };
        });
      })
      .then(function (data) {
        var prefix = data.ok ? '[OK] ' : '[ERR] ';
        var message = prefix + (data.message || '') + '\\n\\n' + (data.log || '');
        logEl.textContent = message;
        renderPreview(data.preview);
        setStatus(data.ok ? 'ok' : 'err');
      })
      .catch(function (err) {
        logEl.textContent = '[ERR] request failed\\n\\n' + String(err);
        renderPreview({});
        setStatus('err');
      })
      .finally(function () {
        setBusy(false);
      });
  });

  setActiveMode();
  if (!dialog.open) {
    dialog.showModal();
  }
})();
</script>
</body>
</html>
"""

    replacements = {
        "__LANG_HTML__": "ru" if lang == "ru" else "en",
        "__TITLE__": html.escape(t["title"]),
        "__MENU_TITLE__": html.escape(t["menu_title"]),
        "__MENU_1__": html.escape(t["menu_1"]),
        "__MENU_2__": html.escape(t["menu_2"]),
        "__MENU_3__": html.escape(t["menu_3"]),
        "__HOST__": html.escape(host),
        "__PORT__": str(port),
        "__PANEL_TITLE__": html.escape(t["panel_title"]),
        "__PANEL_DESC__": html.escape(t["panel_desc"]),
        "__PARSE_MODE__": html.escape(t["parse_mode"]),
        "__MODE_CLASSIC__": html.escape(t["mode_classic"]),
        "__MODE_POST__": html.escape(t["mode_post"]),
        "__MODE_RANGE__": html.escape(t["mode_range"]),
        "__CONFIGURE__": html.escape(t["configure"]),
        "__SENTIMENT__": html.escape(t["sentiment"]),
        "__API_KEY__": html.escape(t["api_key"]),
        "__MODEL__": html.escape(t["model"]),
        "__TRAINING_CSV__": html.escape(t["training_csv"]),
        "__THRESHOLD__": html.escape(t["threshold"]),
        "__RUN__": html.escape(t["run"]),
        "__FLOW_HINT__": html.escape(t["flow_hint"]),
        "__DIALOG_TITLE__": html.escape(t["dialog_title"]),
        "__SAVE__": html.escape(t["save"]),
        "__CLOSE__": html.escape(t["close"]),
        "__CLASSIC_SUBREDDIT__": html.escape(t["classic_subreddit"]),
        "__CLASSIC_CATEGORY__": html.escape(t["classic_category"]),
        "__CLASSIC_TIME_FILTER__": html.escape(t["classic_time_filter"]),
        "__CLASSIC_POSTS_LIMIT__": html.escape(t["classic_posts_limit"]),
        "__CLASSIC_TARGET_COMMENTS__": html.escape(t["classic_target_comments"]),
        "__POST_URL__": html.escape(t["post_url"]),
        "__POST_PARSE_ALL__": html.escape(t["post_parse_all"]),
        "__POST_TARGET_COMMENTS__": html.escape(t["post_target_comments"]),
        "__RANGE_SUBREDDIT__": html.escape(t["range_subreddit"]),
        "__RANGE_START__": html.escape(t["range_start"]),
        "__RANGE_END__": html.escape(t["range_end"]),
        "__RANGE_COMMENTS_PER_POST__": html.escape(t["range_comments_per_post"]),
        "__RANGE_CATEGORY__": html.escape(t["range_category"]),
        "__STATUS_LABEL__": html.escape(t["status_label"]),
        "__READY__": html.escape(t["ready"]),
        "__RUNNING__": html.escape(t["running"]),
        "__DONE__": html.escape(t["done"]),
        "__ERROR__": html.escape(t["error"]),
        "__JS_READY__": json.dumps(t["ready"], ensure_ascii=False),
        "__JS_RUNNING__": json.dumps(t["running"], ensure_ascii=False),
        "__JS_DONE__": json.dumps(t["done"], ensure_ascii=False),
        "__JS_ERROR__": json.dumps(t["error"], ensure_ascii=False),
    }

    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)

    return rendered
