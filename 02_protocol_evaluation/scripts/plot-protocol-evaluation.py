#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Plot reliable transport comparison figures from protocol-evaluation-results.csv.

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ns3")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


PROTOCOLS = ["TCP_CUBIC", "TCP_BBR", "QUIC_BBR"]
PLOT_LABELS = ["TCP_CUBIC", "TCP_BBR", "QUIC_BBR", "NO_DELIVERY"]
PROTOCOL_COLORS = {
    "TCP_CUBIC": "#4C78A8",
    "TCP_BBR": "#F58518",
    "QUIC_BBR": "#54A24B",
    "NO_DELIVERY": "#9CA3AF",
}


def parse_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def read_rows(path):
    rows = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row.get("status", "ok") != "ok" or row.get("throughput_mbps", "") == "":
                continue
            row["rtt_ms"] = parse_float(row, "rtt_ms")
            row["capacity_mbps"] = parse_float(row, "capacity_mbps")
            row["packet_loss_rate"] = parse_float(row, "packet_loss_rate")
            row["throughput_mbps"] = parse_float(row, "throughput_mbps")
            row["file_size_mb"] = parse_float(row, "file_size_mb")
            row["estimated_file_transfer_time_s"] = parse_float(
                row,
                "estimated_file_transfer_time_s",
                default=0.0,
            )
            row["min_send_interval_ms"] = parse_float(row, "min_send_interval_ms")
            row["max_send_interval_ms"] = parse_float(row, "max_send_interval_ms")
            rows.append(row)
    return rows


def condition_key(row):
    return (
        row.get("scenario", ""),
        row["rtt_ms"],
        row["capacity_mbps"],
        row["packet_loss_rate"],
        row["min_send_interval_ms"],
        row["max_send_interval_ms"],
    )


def winner_for(rows):
    if not any(row["throughput_mbps"] > 0 for row in rows):
        winner = rows[0].copy()
        winner["protocol"] = "NO_DELIVERY"
        winner["throughput_mbps"] = 0.0
        winner["estimated_file_transfer_time_s"] = 0.0
        return winner
    with_time = [row for row in rows if row["estimated_file_transfer_time_s"] > 0]
    if with_time:
        return min(with_time, key=lambda row: row["estimated_file_transfer_time_s"])
    return max(rows, key=lambda row: row["throughput_mbps"])


def group_winners(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[condition_key(row)].append(row)

    winners = []
    for key, group in sorted(grouped.items()):
        protocols = {row["protocol"] for row in group}
        if not all(protocol in protocols for protocol in PROTOCOLS):
            continue
        winner = winner_for(group).copy()
        if winner["protocol"] == "NO_DELIVERY":
            winner["winner_metric"] = "no_delivery"
        elif winner["estimated_file_transfer_time_s"] > 0:
            winner["winner_metric"] = "min_file_time"
        else:
            winner["winner_metric"] = "max_throughput"
        winners.append(winner)
    return winners


def write_winner_csv(path, winners):
    fields = [
        "scenario",
        "rtt_ms",
        "capacity_mbps",
        "packet_loss_rate",
        "min_send_interval_ms",
        "max_send_interval_ms",
        "winner",
        "winner_throughput_mbps",
        "winner_file_time_s",
        "winner_metric",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in winners:
            writer.writerow(
                {
                    "scenario": row["scenario"],
                    "rtt_ms": f"{row['rtt_ms']:.6f}",
                    "capacity_mbps": f"{row['capacity_mbps']:.6f}",
                    "packet_loss_rate": f"{row['packet_loss_rate']:.6f}",
                    "min_send_interval_ms": f"{row['min_send_interval_ms']:.6f}",
                    "max_send_interval_ms": f"{row['max_send_interval_ms']:.6f}",
                    "winner": row["protocol"],
                    "winner_throughput_mbps": f"{row['throughput_mbps']:.6f}",
                    "winner_file_time_s": (
                        f"{row['estimated_file_transfer_time_s']:.6f}"
                        if row["estimated_file_transfer_time_s"] > 0
                        else ""
                    ),
                    "winner_metric": row["winner_metric"],
                }
            )


def plot_winner_map(rows, scenario, interval_label, interval_rows, output_dir):
    capacities = sorted({row["capacity_mbps"] for row in interval_rows})
    losses = sorted({row["packet_loss_rate"] for row in interval_rows})
    protocol_index = {protocol: index for index, protocol in enumerate(PLOT_LABELS)}
    color_list = [PROTOCOL_COLORS[protocol] for protocol in PLOT_LABELS]

    matrix = []
    labels = []
    for loss in losses:
        matrix_row = []
        label_row = []
        for capacity in capacities:
            cells = [
                row
                for row in interval_rows
                if row["capacity_mbps"] == capacity and row["packet_loss_rate"] == loss
            ]
            if not cells:
                matrix_row.append(-1)
                label_row.append("")
                continue
            winner = cells[0]
            matrix_row.append(protocol_index[winner["protocol"]])
            if winner["protocol"] == "NO_DELIVERY":
                label_row.append("NO\nDELIVERY")
            elif winner["estimated_file_transfer_time_s"] > 0:
                label_row.append(
                    f"{winner['protocol']}\n{winner['estimated_file_transfer_time_s']:.1f}s"
                )
            else:
                label_row.append(f"{winner['protocol']}\n{winner['throughput_mbps']:.3f}")
        matrix.append(matrix_row)
        labels.append(label_row)

    plt.figure(figsize=(max(7, len(capacities) * 1.6), max(4.5, len(losses) * 0.8)))
    plt.imshow(matrix, cmap=ListedColormap(color_list), vmin=0, vmax=len(PLOT_LABELS) - 1)
    plt.xticks(range(len(capacities)), [f"{capacity:g}" for capacity in capacities])
    plt.yticks(range(len(losses)), [f"{loss:g}" for loss in losses])
    plt.xlabel("Capacity (Mbps)")
    plt.ylabel("Packet loss rate")
    plt.title(f"{scenario} winner map ({interval_label})")

    for y, _loss in enumerate(losses):
        for x, _capacity in enumerate(capacities):
            plt.text(
                x,
                y,
                labels[y][x],
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

    patches = [
        mpatches.Patch(color=PROTOCOL_COLORS[protocol], label=protocol)
        for protocol in PLOT_LABELS
    ]
    plt.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    plt.tight_layout()
    safe_interval = interval_label.replace(" ", "").replace("-", "_").replace("ms", "ms")
    plt.savefig(output_dir / f"protocol-winner-map-{scenario}-{safe_interval}.png", dpi=180)
    plt.close()


def plot_all_winner_maps(winners, output_dir):
    by_scenario_interval = defaultdict(list)
    for row in winners:
        interval_label = f"{row['min_send_interval_ms']:g}-{row['max_send_interval_ms']:g}ms"
        by_scenario_interval[(row["scenario"], interval_label)].append(row)

    for (scenario, interval_label), rows in sorted(by_scenario_interval.items()):
        plot_winner_map(winners, scenario, interval_label, rows, output_dir)


def plot_winner_counts(winners, output_dir):
    counts = Counter(row["protocol"] for row in winners)
    labels = PLOT_LABELS
    values = [counts.get(protocol, 0) for protocol in labels]

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(labels, values, color=[PROTOCOL_COLORS[label] for label in labels])
    plt.ylabel("Winning conditions")
    plt.title("Protocol winner count")
    plt.grid(True, axis="y", alpha=0.3)
    for index, value in enumerate(values):
        plt.text(index, value, str(value), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "protocol-winner-counts.png", dpi=180)
    plt.close()


def plot_protocol_averages(rows, output_dir):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["protocol"]].append(row)

    labels = [protocol for protocol in PROTOCOLS if protocol in grouped]
    throughputs = [
        sum(row["throughput_mbps"] for row in grouped[protocol]) / len(grouped[protocol])
        for protocol in labels
    ]
    file_times = []
    has_file_time = False
    for protocol in labels:
        values = [
            row["estimated_file_transfer_time_s"]
            for row in grouped[protocol]
            if row["estimated_file_transfer_time_s"] > 0
        ]
        has_file_time = has_file_time or bool(values)
        file_times.append(sum(values) / len(values) if values else 0.0)

    plt.figure(figsize=(8, 4.6))
    plt.bar(labels, throughputs, color=[PROTOCOL_COLORS[label] for label in labels])
    plt.ylabel("Average throughput (Mbit/s)")
    plt.title("Average throughput by protocol")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "average-throughput-by-protocol.png", dpi=180)
    plt.close()

    if has_file_time:
        plt.figure(figsize=(8, 4.6))
        plt.bar(labels, file_times, color=[PROTOCOL_COLORS[label] for label in labels])
        plt.ylabel("Estimated file transfer time (s)")
        plt.title("Estimated file transfer time by protocol")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "average-file-transfer-time-by-protocol.png", dpi=180)
        plt.close()


def write_summary(path, rows, winners):
    counts = Counter(row["protocol"] for row in winners)
    metric = "estimated file transfer time" if any(
        row["estimated_file_transfer_time_s"] > 0 for row in winners
    ) else "throughput"
    with path.open("w") as stream:
        stream.write("# Protocol Evaluation Figure Summary\n\n")
        stream.write(f"Input samples: {len(rows)}\n\n")
        stream.write(f"Complete comparison conditions: {len(winners)}\n\n")
        stream.write(f"Winner metric: {metric}\n\n")
        stream.write("## Winner counts\n\n")
        for protocol in PLOT_LABELS:
            stream.write(f"- {protocol}: {counts.get(protocol, 0)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scratch/protocol-evaluation-results/protocol-evaluation-results.csv")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_path)
    winners = group_winners(rows)
    write_winner_csv(output_dir / "protocol-winner-statistics.csv", winners)
    plot_all_winner_maps(winners, output_dir)
    plot_winner_counts(winners, output_dir)
    plot_protocol_averages(rows, output_dir)
    write_summary(output_dir / "protocol-figure-summary.md", rows, winners)
    print(f"Protocol comparison figures written to {output_dir}")


if __name__ == "__main__":
    main()
