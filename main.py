#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point for Reddit Account Analyzer CLI."""

import os
import sys

from reddit_bots.cli.cli import run_cli


if sys.platform == "win32":
    try:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8", errors="replace")
    except Exception:
        pass


def c(color: str, text: str) -> str:
    colors = {
        "BLUE": "\033[94m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "CYAN": "\033[96m",
        "BOLD": "\033[1m",
        "END": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['END']}"


def print_header() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    print(
        c(
            "CYAN",
            c(
                "BOLD",
                """
                Reddit Account Analyzer
      Behavioral Analysis of Reddit Accounts
""",
            ),
        )
    )


def print_menu() -> int:
    print(c("BLUE", "Select mode:"))
    print()
    print(f"  {c('GREEN', '1)')} CLI interface")
    print(f"  {c('RED', '2)')} Exit")
    print()

    raw = input("Mode: ").strip() or "1"
    try:
        return int(raw)
    except ValueError:
        return 1


def run_cli_interface() -> None:
    run_cli()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print_header()
    mode = print_menu()
    if mode == 1:
        run_cli_interface()
