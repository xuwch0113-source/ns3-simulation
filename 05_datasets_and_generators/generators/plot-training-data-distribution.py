#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Plot basic distributions for the adaptive protocol training dataset.

import argparse
import csv
import gzip
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_bar_svg(path, title, counts, ylabel="Samples"):
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    width, height = 980, 560
    left, top, right, bottom = 90, 80, 40, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max(counts.values()) if counts else 1
    gap = 24
    bar_w = max(24, (plot_w - gap * (len(items) + 1)) / max(1, len(items)))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="40" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>',
        f'<text x="28" y="{top + plot_h / 2}" transform="rotate(-90 28 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="15">{ylabel}</text>',
    ]
    for index, (label, value) in enumerate(items):
        x = left + gap + index * (bar_w + gap)
        h = value / max_value * plot_h
        y = top + plot_h - h
        color = colors[index % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="Arial" font-size="12" transform="rotate(20 {x + bar_w / 2:.1f} {top + plot_h + 25})">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_histogram_svg(path, title, values, xlabel, bins=18):
    if not values:
        return
    low, high = min(values), max(values)
    step = (high - low) / bins if high > low else 1.0
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(bins - 1, int((value - low) / step)) if high > low else 0
        counts[index] += 1
    labels = [f"{low + i * step:.3g}" for i in range(bins)]
    write_bar_svg(path, title, dict(zip(labels, counts)), ylabel=xlabel)


def write_summary(path, rows, output_dir):
    label_counts = Counter(row.get("best_protocol", "unknown") for row in rows)
    state_counts = Counter(row.get("state_name", "unknown") for row in rows)
    with path.open("w") as stream:
        stream.write("# Training Data Distribution Figures\n\n")
        stream.write(f"Input samples: {len(rows)}\n\n")
        stream.write("## Figures\n\n")
        for figure in sorted(output_dir.glob("*.svg")):
            stream.write(f"- `{figure.name}`\n")
        stream.write("\n## Best Protocol Label Counts\n\n")
        for label, count in label_counts.most_common():
            stream.write(f"- {label}: {count}\n")
        stream.write("\n## Dynamic Link State Counts\n\n")
        for state, count in state_counts.most_common():
            stream.write(f"- {state}: {count}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="05_datasets_and_generators/training-data/adaptive-protocol-large-training-dataset.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="05_datasets_and_generators/training-data/distribution-figures",
    )
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(input_path)
    if not rows:
        raise RuntimeError("No input rows found.")

    write_bar_svg(
        output_dir / "best-protocol-label-distribution.svg",
        "Best Protocol Label Distribution",
        Counter(row.get("best_protocol", "unknown") for row in rows),
    )
    write_bar_svg(
        output_dir / "dynamic-link-state-distribution.svg",
        "Dynamic Link State Distribution",
        Counter(row.get("state_name", "unknown") for row in rows),
    )
    write_bar_svg(
        output_dir / "ocean-data-type-distribution.svg",
        "Ocean Data Type Distribution",
        Counter(row.get("data_type", "unknown") for row in rows),
    )
    write_histogram_svg(
        output_dir / "packet-loss-rate-histogram.svg",
        "Packet Loss Rate Distribution",
        [safe_float(row.get("packet_loss_rate")) for row in rows],
        "Packet loss rate",
    )
    write_histogram_svg(
        output_dir / "capacity-histogram.svg",
        "Capacity Distribution",
        [safe_float(row.get("capacity_mbps")) for row in rows],
        "Capacity Mbps",
    )
    write_summary(output_dir / "figure-notes.md", rows, output_dir)
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
