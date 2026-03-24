# REDDIT-BOTS

[English README](README.md) | [Русский README](README_RU.md)

![](image1.jpg)

`REDDIT-BOTS` is a compact CLI tool for Reddit account risk analysis.
It parses comments, builds account-level behavior features, and estimates suspiciousness (`bot_probability`).

## Project Tree

```text
reddit-bots/
├── main.py
├── main_parser.py
├── reddit_bots/
│   ├── parser/
│   │   └── reddit_parser.py
│   ├── analysis/
│   │   ├── behavior_metrics.py
│   │   └── account_features.py
│   ├── models/
│   │   └── bot_classifier.py
│   └── cli/
│       └── cli.py
├── requirements.txt
├── reddit_dead_internet_analysis_2026.csv
├── image1.jpg
├── README.md
└── README_RU.md
```

## Download

```bash
git clone https://github.com/ollxel/reddit-bots
cd reddit-bots
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```
