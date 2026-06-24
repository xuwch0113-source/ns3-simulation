#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Draw presentation-ready figures from lightweight model training outputs.

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / "python-packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ns3-final-submission-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = ["TCP_CUBIC", "TCP_BBR", "QUIC_BBR"]
LEARNED_PREFIXES = ("Static-",)
METRIC_COLUMNS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "macro_auc",
    "weighted_auc",
]
PERFORMANCE_COLUMNS = ["avg_selected_mbps", "oracle_gap_mbps"]
MODEL_COLORS = {
    "LightGBM": "#4c78a8",
    "XGBoost": "#f58518",
    "RandomForest": "#54a24b",
    "TinyMLP": "#b279a2",
    "CatBoost": "#e45756",
    "LinearSVM": "#72b7b2",
    "ContextualBandit-Ridge": "#9d755d",
    "Static-TCP_CUBIC": "#8f8f8f",
    "Static-TCP_BBR": "#a6a6a6",
    "Static-QUIC_BBR": "#c7c7c7",
}


def is_learned_model(name):
    return not name.startswith(LEARNED_PREFIXES)


def read_metrics(input_dir):
    metrics_path = input_dir / "model-metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path)
    for column in METRIC_COLUMNS + PERFORMANCE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def draw_metric_heatmap(df, output_dir):
    metrics = ["accuracy", "macro_f1", "macro_auc", "avg_selected_mbps"]
    plot_df = df.set_index("model")[metrics].copy()
    normalized = plot_df.copy()
    for column in metrics:
        low = plot_df[column].min()
        high = plot_df[column].max()
        normalized[column] = (plot_df[column] - low) / (high - low) if high > low else 1.0

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    image = ax.imshow(normalized.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["Accuracy", "Macro F1", "Macro AUC", "Avg Mbps"], rotation=0)
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Model metric overview")
    for row_index, model in enumerate(plot_df.index):
        for col_index, metric in enumerate(metrics):
            value = plot_df.loc[model, metric]
            ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.82, label="Normalized score")
    savefig(output_dir / "model-metric-overview-heatmap.png")


def draw_grouped_metrics(df, output_dir):
    learned = df[df["model"].map(is_learned_model)].copy()
    learned = learned.sort_values("macro_f1", ascending=False)
    metrics = ["accuracy", "macro_f1", "macro_auc", "weighted_auc"]
    x = np.arange(len(learned))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for index, metric in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2.0) * width
        ax.bar(x + offset, learned[metric], width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(learned["model"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Learned model classification metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    savefig(output_dir / "learned-model-classification-metrics.png")


def draw_gap_ranking(df, output_dir):
    plot_df = df.sort_values("oracle_gap_mbps", ascending=True)
    colors = [MODEL_COLORS.get(model, "#4c78a8") for model in plot_df["model"]]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    bars = ax.barh(plot_df["model"], plot_df["oracle_gap_mbps"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Oracle gap (Mbit/s, lower is better)")
    ax.set_title("Gap to oracle ranking")
    ax.grid(axis="x", alpha=0.25)
    for bar in bars:
        value = bar.get_width()
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    savefig(output_dir / "oracle-gap-ranking.png")


def draw_scatter(df, output_dir):
    plot_df = df[df["model"].map(is_learned_model)].copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    offsets = {
        "LightGBM": (10, 10),
        "XGBoost": (8, -14),
        "TinyMLP": (-68, 10),
        "CatBoost": (8, 18),
        "RandomForest": (8, -12),
        "LinearSVM": (8, 8),
        "ContextualBandit-Ridge": (8, 8),
    }
    for _index, row in plot_df.iterrows():
        color = MODEL_COLORS.get(row["model"], "#4c78a8")
        ax.scatter(row["accuracy"], row["avg_selected_mbps"], s=115, color=color, marker="o", edgecolor="white", linewidth=1.2)
        offset = offsets.get(row["model"], (8, 8))
        ax.annotate(
            row["model"],
            (row["accuracy"], row["avg_selected_mbps"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.7} if row["model"] in {"LightGBM", "XGBoost", "TinyMLP", "CatBoost"} else None,
        )
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Average selected throughput (Mbit/s)")
    ax.set_title("Learned model accuracy vs selected throughput")
    y_min = plot_df["avg_selected_mbps"].min()
    y_max = plot_df["avg_selected_mbps"].max()
    ax.set_ylim(y_min - 0.006, y_max + 0.010)
    ax.grid(alpha=0.25)
    savefig(output_dir / "accuracy-vs-throughput-scatter.png")


def draw_static_vs_best(df, output_dir):
    learned = df[df["model"].map(is_learned_model)].copy()
    statics = df[~df["model"].map(is_learned_model)].copy()
    best_f1 = learned.sort_values(["macro_f1", "avg_selected_mbps"], ascending=False).iloc[0]
    best_gap = learned.sort_values(["oracle_gap_mbps", "macro_f1"], ascending=[True, False]).iloc[0]
    selected = pd.concat([best_f1.to_frame().T, best_gap.to_frame().T, statics], ignore_index=True)
    selected = selected.drop_duplicates(subset=["model"], keep="first")

    metrics = ["macro_f1", "macro_auc", "avg_selected_mbps"]
    x = np.arange(len(selected))
    width = 0.23
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for index, metric in enumerate(metrics):
        offset = (index - 1) * width
        ax.bar(x + offset, selected[metric], width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(selected["model"], rotation=24, ha="right")
    ax.set_title("Adaptive learned models vs static protocol baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3)
    savefig(output_dir / "learned-vs-static-protocol-comparison.png")


def draw_radar(df, output_dir):
    learned = df[df["model"].map(is_learned_model)].copy()
    selected = learned.sort_values("macro_f1", ascending=False).head(5)
    metrics = ["accuracy", "macro_f1", "macro_auc", "avg_selected_mbps"]
    values = selected[metrics].copy()
    for metric in metrics:
        low = df[metric].min()
        high = df[metric].max()
        values[metric] = (selected[metric] - low) / (high - low) if high > low else 1.0

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8.2, 8.2))
    ax = plt.subplot(111, polar=True)
    for row_index, row in selected.iterrows():
        model_values = values.loc[row_index].tolist()
        model_values += model_values[:1]
        color = MODEL_COLORS.get(row["model"], None)
        ax.plot(angles, model_values, label=row["model"], linewidth=2, color=color)
        ax.fill(angles, model_values, alpha=0.10, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Accuracy", "Macro F1", "Macro AUC", "Avg Mbps"])
    ax.set_ylim(0, 1)
    ax.set_title("Top learned models radar view", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
    savefig(output_dir / "learned-model-radar-chart.png")


def draw_confusion_matrices(input_dir, output_dir):
    matrix_files = sorted(input_dir.glob("confusion-matrix-*.csv"))
    learned_files = [
        path for path in matrix_files
        if is_learned_model(path.stem.replace("confusion-matrix-", ""))
    ]
    if not learned_files:
        return
    cols = 2
    rows = int(np.ceil(len(learned_files) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.1, rows * 4.1))
    fig.subplots_adjust(top=0.91, bottom=0.06, left=0.08, right=0.86, hspace=0.55, wspace=0.30)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for index, path in enumerate(learned_files):
        model = path.stem.replace("confusion-matrix-", "")
        matrix = pd.read_csv(path, index_col=0).loc[LABELS, LABELS]
        normalized = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
        ax = axes.ravel()[index]
        ax.axis("on")
        im = ax.imshow(normalized.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(model)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(LABELS)))
        ax.set_yticklabels(LABELS, fontsize=8)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{normalized.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    colorbar_axis = fig.add_axes([0.89, 0.18, 0.025, 0.64])
    fig.colorbar(im, cax=colorbar_axis, label="Row-normalized ratio")
    fig.suptitle("Learned model confusion matrices", y=0.975, fontsize=15)
    plt.savefig(output_dir / "learned-model-confusion-matrices.png", dpi=220)
    plt.close()


def write_summary(df, output_dir):
    learned = df[df["model"].map(is_learned_model)].copy()
    best_f1 = learned.sort_values(["macro_f1", "avg_selected_mbps"], ascending=False).iloc[0]
    best_auc = learned.sort_values(["macro_auc", "macro_f1"], ascending=False).iloc[0]
    best_gap = learned.sort_values(["oracle_gap_mbps", "macro_f1"], ascending=[True, False]).iloc[0]
    with (output_dir / "figure-notes.md").open("w") as stream:
        stream.write("# Lightweight Model Training Figure Notes\n\n")
        stream.write("These figures are generated from `model-metrics.csv` and model confusion matrices. The script does not retrain models.\n\n")
        stream.write("## Suggested Figures for Presentation\n\n")
        stream.write("- `model-metric-overview-heatmap.png`: overview of major metrics for all models.\n")
        stream.write("- `learned-model-classification-metrics.png`: compares Accuracy, F1, and AUC for learned models.\n")
        stream.write("- `accuracy-vs-throughput-scatter.png`: shows that classification accuracy and throughput gain are not identical.\n")
        stream.write("- `oracle-gap-ranking.png`: shows the gap to the oracle selector.\n")
        stream.write("- `learned-model-confusion-matrices.png`: shows where protocol labels are misclassified.\n\n")
        stream.write("## Current Findings\n\n")
        stream.write(f"- Best Macro F1: {best_f1['model']} ({best_f1['macro_f1']:.3f}).\n")
        stream.write(f"- Best Macro AUC: {best_auc['model']} ({best_auc['macro_auc']:.3f}).\n")
        stream.write(f"- Smallest Oracle gap: {best_gap['model']} ({best_gap['oracle_gap_mbps']:.3f} Mbit/s).\n\n")
        stream.write("## Figure Files\n\n")
        for path in sorted(output_dir.glob("*.png")):
            stream.write(f"- `{path.name}`\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="08_algorithm_docs/model-results")
    parser.add_argument("--output-dir", default="08_algorithm_docs/model-results/figures")
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_metrics(input_dir)

    draw_metric_heatmap(df, output_dir)
    draw_grouped_metrics(df, output_dir)
    draw_gap_ranking(df, output_dir)
    draw_scatter(df, output_dir)
    draw_static_vs_best(df, output_dir)
    draw_radar(df, output_dir)
    draw_confusion_matrices(input_dir, output_dir)
    write_summary(df, output_dir)

    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
