"""
reddit_bots_plots.py
────────────────────
Визуализация датасета и метода детекции ботов REDDIT-BOTS.
Использует ту же модель (RandomForestClassifier), что и bot_classifier.py.

Запуск:
    python reddit_bots_plots.py
    python reddit_bots_plots.py --csv path/to/dataset.csv --out ./plots

Зависимости: pandas, numpy, scikit-learn, matplotlib, seaborn
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── palette ────────────────────────────────────────────────────────────────
PALETTE = {
    "None (Human)":     "#4CAF50",
    "AI Summarizer":    "#2196F3",
    "Reprint Bot":      "#FF9800",
    "Engagement Farmer":"#F44336",
}
BG      = "#0F1117"
CARD    = "#1A1D27"
TEXT    = "#E8EAF0"
ACCENT  = "#E53935"
GRID    = "#2A2D3A"
SUBTEXT = "#8A8EA8"

BOT_TYPES = ["None (Human)", "AI Summarizer", "Reprint Bot", "Engagement Farmer"]
TYPE_COLORS = [PALETTE[t] for t in BOT_TYPES]

# ── model features (same as ACCOUNT_FEATURES in bot_classifier.py) ─────────
FEATURES = [
    "account_age_days",
    "user_karma",
    "reply_delay_seconds",
    "sentiment_score",
    "avg_word_length",
]
LABEL_COL  = "is_bot_flag"
PROB_COL   = "bot_probability"
TYPE_COL   = "bot_type_label"


# ── helpers ────────────────────────────────────────────────────────────────
def apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    CARD,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "axes.grid":         True,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "xtick.color":       SUBTEXT,
        "ytick.color":       SUBTEXT,
        "text.color":        TEXT,
        "legend.facecolor":  CARD,
        "legend.edgecolor":  GRID,
        "legend.labelcolor": TEXT,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
    })


def save(fig, path, title=""):
    fig.patch.set_facecolor(BG)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  {os.path.basename(path)}")


def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)
    df["is_bot_flag"] = df["is_bot_flag"].map(
        lambda v: 1 if str(v).lower() in {"true", "1", "yes"} else 0
    )
    df["contains_links"] = df["contains_links"].map(
        lambda v: 1 if str(v).lower() in {"true", "1"} else 0
    )
    df[TYPE_COL] = df[TYPE_COL].fillna("None (Human)")
    return df


def train_model(df: pd.DataFrame):
    X = df[FEATURES].fillna(0)
    y = df[LABEL_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Dataset overview (2×2)
# ══════════════════════════════════════════════════════════════════════════
def plot_dataset_overview(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)

    # ── 1a: bot type distribution (donut) ──────────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor(CARD)
    counts = df[TYPE_COL].value_counts()
    labels = [t for t in BOT_TYPES if t in counts.index]
    sizes  = [counts[t] for t in labels]
    colors = [PALETTE[t] for t in labels]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.75,
        wedgeprops={"linewidth": 2, "edgecolor": BG, "width": 0.55},
    )
    for at in autotexts:
        at.set(color=TEXT, fontsize=9, fontweight="bold")
    handles = [mpatches.Patch(color=c, label=l) for l, c in zip(labels, colors)]
    ax.legend(handles=handles, labels=labels,
              loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    ax.set_title("Распределение типов аккаунтов", color=TEXT, pad=12)
    ax.text(0, 0, f"{len(df)}\nзаписей", ha="center", va="center",
            color=TEXT, fontsize=11, fontweight="bold")

    # ── 1b: subreddit bar ──────────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_facecolor(CARD)
    sub_bot  = df[df[LABEL_COL] == 1]["subreddit"].value_counts()
    sub_hum  = df[df[LABEL_COL] == 0]["subreddit"].value_counts()
    subs     = sorted(set(sub_bot.index) | set(sub_hum.index))
    x        = np.arange(len(subs))
    w        = 0.38
    ax.bar(x - w/2, [sub_bot.get(s, 0) for s in subs], w,
           color=ACCENT, alpha=0.88, label="Боты", zorder=3)
    ax.bar(x + w/2, [sub_hum.get(s, 0) for s in subs], w,
           color="#4CAF50", alpha=0.88, label="Люди", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Кол-во комментариев")
    ax.set_title("Боты vs Люди по сабреддитам", color=TEXT, pad=12)
    ax.legend()
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    # ── 1c: bot_probability histogram ──────────────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor(CARD)
    for btype, color in PALETTE.items():
        sub = df[df[TYPE_COL] == btype][PROB_COL]
        ax.hist(sub, bins=20, range=(0, 1), alpha=0.75,
                color=color, label=btype, edgecolor=BG, linewidth=0.4, zorder=3)
    ax.axvline(0.3, color="#FFD600", lw=1.4, ls="--", label="Порог 0.3")
    ax.axvline(0.6, color=ACCENT, lw=1.4, ls="--", label="Порог 0.6")
    ax.set_xlabel("bot_probability")
    ax.set_ylabel("Количество комментариев")
    ax.set_title("Распределение bot_probability по типам", color=TEXT, pad=12)
    ax.legend(fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    # ── 1d: account age vs reply delay scatter ─────────────────────────
    ax = axes[1, 1]
    ax.set_facecolor(CARD)
    for btype, color in PALETTE.items():
        sub = df[df[TYPE_COL] == btype]
        ax.scatter(sub["account_age_days"], np.log1p(sub["reply_delay_seconds"]),
                   c=color, alpha=0.55, s=22, label=btype,
                   edgecolors="none", zorder=3)
    ax.set_xlabel("account_age_days")
    ax.set_ylabel("log(reply_delay_seconds + 1)")
    ax.set_title("Возраст аккаунта vs Задержка ответа", color=TEXT, pad=12)
    ax.legend(fontsize=8, markerscale=1.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    save(fig, os.path.join(out_dir, "1_dataset_overview.png"),
         "Обзор датасета REDDIT-BOTS")


# ══════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Feature distributions (violins)
# ══════════════════════════════════════════════════════════════════════════
def plot_feature_distributions(df: pd.DataFrame, out_dir: str):
    feat_config = [
        ("account_age_days",     "Возраст аккаунта (дни)",    False),
        ("user_karma",           "Карма пользователя",         False),
        ("reply_delay_seconds",  "Задержка ответа (сек)",      True),
        ("sentiment_score",      "Тональность (sentiment)",    False),
        ("avg_word_length",      "Средняя длина слова",        False),
        ("bot_probability",      "bot_probability",             False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor(BG)

    for ax, (col, title, log_scale) in zip(axes.flat, feat_config):
        ax.set_facecolor(CARD)
        plot_df = df[[TYPE_COL, col]].copy()
        if log_scale:
            plot_df[col] = np.log1p(plot_df[col])
            title = f"log({title}+1)"

        order = BOT_TYPES
        sns.violinplot(
            data=plot_df, x=TYPE_COL, y=col, order=order,
            palette=PALETTE, inner="quartile", linewidth=0.8,
            ax=ax, cut=0,
        )
        ax.set_xticklabels(
            [t.replace(" ", "\n") for t in order],
            fontsize=8.5, color=SUBTEXT,
        )
        ax.set_xlabel("")
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, color=TEXT, pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    save(fig, os.path.join(out_dir, "2_feature_distributions.png"),
         "Распределение признаков по типам аккаунтов")


# ══════════════════════════════════════════════════════════════════════════
#  PLOT 3 — Correlation heatmap
# ══════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str):
    cols = [
        "account_age_days", "user_karma", "reply_delay_seconds",
        "sentiment_score", "avg_word_length", "contains_links",
        "bot_probability", "is_bot_flag",
    ]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 9, "color": TEXT},
        linewidths=0.5, linecolor=BG,
        square=True, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right",
                       fontsize=9, color=SUBTEXT)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=9, color=SUBTEXT)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=SUBTEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=SUBTEXT)

    save(fig, os.path.join(out_dir, "3_correlation_heatmap.png"),
         "Тепловая карта корреляций признаков")


# ══════════════════════════════════════════════════════════════════════════
#  PLOT 4 — Model: feature importance + confusion matrix + ROC
# ══════════════════════════════════════════════════════════════════════════
def plot_model_results(clf, X_test, y_test, out_dir: str):
    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # ── 4a: feature importance ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(CARD)
    importances = pd.Series(clf.feature_importances_, index=FEATURES)
    importances = importances.sort_values()
    colors_bar  = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(importances)))
    bars = ax1.barh(importances.index, importances.values,
                    color=colors_bar, edgecolor=BG, height=0.6, zorder=3)
    for bar, val in zip(bars, importances.values):
        ax1.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", color=TEXT, fontsize=8.5)
    ax1.set_xlabel("Важность признака (Gini)", fontsize=10)
    ax1.set_title("Важность признаков\n(Random Forest)", color=TEXT, pad=10)
    ax1.set_xlim(0, importances.max() * 1.22)
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID)

    # ── 4b: confusion matrix ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(CARD)
    y_pred = clf.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    cmap_cm = sns.light_palette(ACCENT, as_cmap=True)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Человек", "Бот"])
    disp.plot(ax=ax2, cmap=cmap_cm, colorbar=False)
    ax2.set_title("Матрица ошибок", color=TEXT, pad=10)
    ax2.tick_params(colors=SUBTEXT)
    ax2.xaxis.label.set_color(TEXT)
    ax2.yaxis.label.set_color(TEXT)
    for text in disp.text_.ravel():
        text.set_color(TEXT)
        text.set_fontsize(14)

    # ── 4c: ROC curve ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(CARD)
    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc      = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color=ACCENT, lw=2,
             label=f"ROC (AUC = {roc_auc:.3f})")
    ax3.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
    ax3.plot([0, 1], [0, 1], color=SUBTEXT, lw=1, ls="--", label="Случайный")
    ax3.set_xlim([-0.02, 1.02])
    ax3.set_ylim([-0.02, 1.05])
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC-кривая", color=TEXT, pad=10)
    ax3.legend(loc="lower right")
    for spine in ax3.spines.values():
        spine.set_edgecolor(GRID)

    save(fig, os.path.join(out_dir, "4_model_results.png"),
         "Результаты модели Random Forest")


# ══════════════════════════════════════════════════════════════════════════
#  PLOT 5 — Bot type profile radar
# ══════════════════════════════════════════════════════════════════════════
def plot_radar_profiles(df: pd.DataFrame, out_dir: str):
    radar_features = [
        "account_age_days", "user_karma",
        "reply_delay_seconds", "sentiment_score", "avg_word_length",
    ]
    labels = [
        "Возраст\nаккаунта", "Карма",
        "Задержка\nответа", "Тональность", "Длина\nслова",
    ]

    # normalise per-column to [0,1]
    normed = df.copy()
    for f in radar_features:
        lo, hi = df[f].min(), df[f].max()
        normed[f] = (df[f] - lo) / (hi - lo + 1e-9)
    # shift sentiment to [0,1]
    normed["sentiment_score"] = (normed["sentiment_score"] + 1) / 2

    profiles = {
        btype: normed[normed[TYPE_COL] == btype][radar_features].mean().values
        for btype in BOT_TYPES
    }

    N     = len(radar_features)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])

    fig, axes = plt.subplots(1, len(BOT_TYPES), figsize=(16, 4.5),
                             subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(BG)

    for ax, btype in zip(axes, BOT_TYPES):
        ax.set_facecolor(CARD)
        vals = np.concatenate([profiles[btype], [profiles[btype][0]]])
        color = PALETTE[btype]
        ax.plot(theta, vals, color=color, lw=2, zorder=4)
        ax.fill(theta, vals, color=color, alpha=0.25, zorder=3)
        ax.set_xticks(theta[:-1])
        ax.set_xticklabels(labels, size=8, color=TEXT)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, linewidth=0.6)
        ax.spines["polar"].set_color(GRID)
        ax.set_title(btype.replace(" ", "\n"), color=color,
                     fontsize=11, fontweight="bold", pad=14)

    save(fig, os.path.join(out_dir, "5_bot_type_radar.png"),
         "Поведенческие профили типов ботов (радар)")


# ══════════════════════════════════════════════════════════════════════════
#  PLOT 6 — Detection method: reply delay analysis
# ══════════════════════════════════════════════════════════════════════════
def plot_reply_delay_analysis(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG)

    # ── 6a: KDE of reply delay by bot class ────────────────────────────
    ax = axes[0]
    ax.set_facecolor(CARD)
    for is_bot, label, color in [(0, "Человек", "#4CAF50"), (1, "Бот", ACCENT)]:
        vals = np.log1p(df[df[LABEL_COL] == is_bot]["reply_delay_seconds"])
        sns.kdeplot(vals, ax=ax, color=color, fill=True, alpha=0.4,
                    label=label, linewidth=2)
    ax.set_xlabel("log(reply_delay_seconds + 1)")
    ax.set_ylabel("Плотность")
    ax.set_title("KDE: задержка ответа\n(Боты vs Люди)", color=TEXT, pad=10)
    ax.legend()
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    # ── 6b: account age bins: bot ratio ───────────────────────────────
    ax = axes[1]
    ax.set_facecolor(CARD)
    df2   = df.copy()
    bins  = [0, 7, 30, 180, 365, 730, 3000]
    blabels = ["<7д", "7-30д", "1-6м", "6-12м", "1-2г", ">2г"]
    df2["age_bin"] = pd.cut(df2["account_age_days"], bins=bins, labels=blabels)
    ratio = df2.groupby("age_bin", observed=False)[LABEL_COL].mean() * 100
    bar_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.85, len(ratio)))
    bars = ax.bar(ratio.index.astype(str), ratio.values,
                  color=bar_colors, edgecolor=BG, zorder=3)
    for bar, val in zip(bars, ratio.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", color=TEXT, fontsize=9)
    ax.set_ylabel("Доля ботов, %")
    ax.set_xlabel("Возраст аккаунта")
    ax.set_title("Доля ботов по возрасту\nаккаунта", color=TEXT, pad=10)
    ax.set_ylim(0, ratio.max() * 1.25)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    # ── 6c: bot_probability vs reply_delay scatter ─────────────────────
    ax = axes[2]
    ax.set_facecolor(CARD)
    sc = ax.scatter(
        np.log1p(df["reply_delay_seconds"]),
        df["bot_probability"],
        c=df[LABEL_COL], cmap="RdYlGn_r",
        alpha=0.55, s=20, edgecolors="none", zorder=3,
    )
    ax.axhline(0.3, color="#FFD600", lw=1.3, ls="--", label="Порог 0.3")
    ax.axhline(0.6, color=ACCENT, lw=1.3, ls="--", label="Порог 0.6")
    ax.set_xlabel("log(reply_delay_seconds + 1)")
    ax.set_ylabel("bot_probability")
    ax.set_title("bot_probability vs\nЗадержка ответа", color=TEXT, pad=10)
    ax.legend(fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    save(fig, os.path.join(out_dir, "6_reply_delay_analysis.png"),
         "Анализ задержки ответа как маркера детекции ботов")


# ══════════════════════════════════════════════════════════════════════════
#  PLOT 7 — Sentiment heatmap by subreddit × bot_type
# ══════════════════════════════════════════════════════════════════════════
def plot_sentiment_heatmap(df: pd.DataFrame, out_dir: str):
    pivot = df.pivot_table(
        values="sentiment_score",
        index="subreddit",
        columns=TYPE_COL,
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=BOT_TYPES)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    sns.heatmap(
        pivot, cmap=cmap, center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 10, "color": TEXT},
        linewidths=0.6, linecolor=BG, square=False,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xticklabels(ax.get_xticklabels(), color=SUBTEXT, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color=SUBTEXT, fontsize=9)
    ax.set_xlabel("Тип аккаунта", color=TEXT)
    ax.set_ylabel("Сабреддит", color=TEXT)

    cbar = ax.collections[0].colorbar
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=SUBTEXT)

    save(fig, os.path.join(out_dir, "7_sentiment_heatmap.png"),
         "Средняя тональность: Сабреддит × Тип бота")


# ══════════════════════════════════════════════════════════════════════════
#  PLOT 8 — Learning curve
# ══════════════════════════════════════════════════════════════════════════
def plot_learning_curve(df: pd.DataFrame, out_dir: str):
    X = df[FEATURES].fillna(0)
    y = df[LABEL_COL]

    train_sizes, train_scores, val_scores = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        X, y, cv=5, scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 8),
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    ax.plot(train_sizes, train_mean, color="#4CAF50", lw=2, marker="o",
            markersize=5, label="Train F1")
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.15, color="#4CAF50")

    ax.plot(train_sizes, val_mean, color=ACCENT, lw=2, marker="s",
            markersize=5, label="Validation F1")
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.15, color=ACCENT)

    ax.set_xlabel("Размер обучающей выборки")
    ax.set_ylabel("F1-score")
    ax.set_title("Кривая обучения (Random Forest, CV=5)", color=TEXT, pad=12)
    ax.legend()
    ax.set_ylim(0, 1.05)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    save(fig, os.path.join(out_dir, "8_learning_curve.png"),
         "Кривая обучения модели")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="REDDIT-BOTS visualization")
    parser.add_argument("--csv", default="reddit_dead_internet_analysis_2026.csv",
                        help="Путь к датасету CSV")
    parser.add_argument("--out", default="plots",
                        help="Папка для сохранения графиков")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    apply_dark_theme()

    print(f"\n📂  Датасет: {args.csv}")
    df = load_and_prepare(args.csv)
    print(f"    Загружено строк: {len(df)}")
    print(f"    Доля ботов: {df[LABEL_COL].mean():.1%}\n")

    print("🤖  Обучение модели...")
    clf, X_train, X_test, y_train, y_test = train_model(df)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred,
                                 target_names=["Человек", "Бот"], zero_division=0))

    print("🎨  Генерация графиков:")
    plot_dataset_overview(df, args.out)
    plot_feature_distributions(df, args.out)
    plot_correlation_heatmap(df, args.out)
    plot_model_results(clf, X_test, y_test, args.out)
    plot_radar_profiles(df, args.out)
    plot_reply_delay_analysis(df, args.out)
    plot_sentiment_heatmap(df, args.out)
    plot_learning_curve(df, args.out)

    print(f"\n✅  Все графики сохранены в: {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
