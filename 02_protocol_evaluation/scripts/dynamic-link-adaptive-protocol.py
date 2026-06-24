#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Decision-tree adaptive transport selection experiment.

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ns3")

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def find_ns3_root():
    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "ns3").exists() and (candidate / "scratch").exists():
            return candidate
    return Path.cwd().resolve()


ROOT = find_ns3_root()
QUIC_ROOT = Path("/Users/xuweicheng/ns-allinone-3.32/ns-3.32")
TCP_EXP = ROOT / "build" / "scratch" / "ns3-dev-tcp-exp-default"
TCP_SOURCE = ROOT / "scratch" / "tcp-exp.cc"
QUIC_EXP = (
    QUIC_ROOT
    / "build"
    / "contrib"
    / "quic"
    / "examples"
    / "ns3.32-quic-variants-comparison-bulksend-debug"
)

THROUGHPUT_RE = re.compile(r"averageThroughput=([0-9.]+) Mbit/s")

PROTOCOLS = ["TCP_CUBIC", "TCP_BBR", "QUIC_BBR"]
TCP_TYPES = {
    "TCP_CUBIC": "ns3::TcpCubic",
    "TCP_BBR": "ns3::TcpBbr",
}
COLORS = {
    "TCP_CUBIC": "#4C78A8",
    "TCP_BBR": "#F58518",
    "QUIC_BBR": "#54A24B",
    "ADAPTIVE_TREE": "#7F3C8D",
}

DEFAULT_TREE_MODEL = ROOT / "scratch" / "expanded-learning-results" / "model-evaluation" / "decision-tree-model.json"
DEFAULT_MLP_MODEL = ROOT / "scratch" / "protocol-selector-results" / "mlp-model.json"


DEFAULT_PHASES = [
    {
        "name": "normal",
        "duration": 5.0,
        "rtt_ms": 237.886,
        "capacity_mbps": 4.0,
        "loss": 0.005,
    },
    {
        "name": "moderate_loss",
        "duration": 5.0,
        "rtt_ms": 237.886,
        "capacity_mbps": 2.5,
        "loss": 0.01,
    },
    {
        "name": "severe_loss",
        "duration": 5.0,
        "rtt_ms": 237.886,
        "capacity_mbps": 2.0,
        "loss": 0.03,
    },
    {
        "name": "recovered",
        "duration": 5.0,
        "rtt_ms": 237.886,
        "capacity_mbps": 4.0,
        "loss": 0.005,
    },
]


def sample_interval(rng, ranges):
    low, high = rng.choice(ranges)
    center = rng.uniform(low, high)
    spread = max(center * 0.6, 0.1)
    min_interval = max(0.1, center - spread / 2.0)
    max_interval = center + spread / 2.0
    return round(min_interval, 3), round(max_interval, 3)


def sample_range(rng, low, high):
    lower = min(low, high)
    upper = max(low, high)
    return rng.uniform(lower, upper)


def clamp(value, low, high):
    return max(low, min(high, value))


def lerp(start, stop, ratio):
    return start + (stop - start) * ratio


def parse_interval_ranges(value):
    ranges = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        low, high = [float(part.strip()) for part in item.split(":", 1)]
        if low <= 0 or high <= 0 or high < low:
            raise ValueError(f"Invalid interval range: {item}")
        ranges.append((low, high))
    if not ranges:
        raise ValueError("At least one interval range is required.")
    return ranges


def build_continuous_bad_heavy_phases(args):
    rng = random.Random(args.seed)
    interval_ranges = parse_interval_ranges(args.send_interval_ranges)
    phases = []
    normal_target = max(0.0, min(args.normal_ratio, 1.0))
    severe_target = max(0.0, min(args.severe_ratio, 1.0))

    for index in range(args.phase_count):
        draw = rng.random()
        if draw < normal_target:
            condition = "normal"
            loss = rng.uniform(0.001, 0.008)
            capacity = sample_range(rng,
                                    max(args.capacity_min_mbps, args.capacity_max_mbps * 0.65),
                                    args.capacity_max_mbps)
        elif draw < normal_target + severe_target:
            condition = "severe"
            loss = rng.uniform(args.severe_loss_min, args.loss_max)
            capacity = sample_range(rng,
                                    args.capacity_min_mbps,
                                    min(args.capacity_max_mbps, args.capacity_min_mbps * 2.5))
        else:
            condition = "degraded"
            loss = rng.uniform(0.008, args.severe_loss_min)
            capacity = sample_range(rng,
                                    args.capacity_min_mbps * 1.5,
                                    args.capacity_max_mbps * 0.75)

        rtt = rng.uniform(args.rtt_min_ms, args.rtt_max_ms)
        min_interval, max_interval = sample_interval(rng, interval_ranges)
        phases.append(
            {
                "name": f"{condition}-{index + 1:02d}",
                "condition": condition,
                "duration": args.phase_duration,
                "rtt_ms": round(rtt, 3),
                "capacity_mbps": round(capacity, 3),
                "loss": round(loss, 6),
                "traffic_mode": args.traffic_mode,
                "min_send_interval_ms": min_interval,
                "max_send_interval_ms": max_interval,
                "packet_size_bytes": args.packet_size,
            }
        )

    return phases


def allocate_progressive_counts(phase_count, normal_ratio, severe_ratio):
    if phase_count < 5:
        raise ValueError("progressive-bad-recover mode requires phase-count >= 5")

    normal_count = max(2, round(phase_count * normal_ratio))
    bad_count = max(1, round(phase_count * severe_ratio))
    if normal_count + bad_count > phase_count - 2:
        bad_count = max(1, phase_count - normal_count - 2)

    transition_count = phase_count - normal_count - bad_count
    degrade_count = max(1, transition_count // 2)
    recover_count = max(1, transition_count - degrade_count)

    while normal_count + bad_count + degrade_count + recover_count > phase_count:
        if bad_count > 1:
            bad_count -= 1
        elif normal_count > 2:
            normal_count -= 1
        else:
            recover_count -= 1

    start_normal_count = max(1, normal_count // 2)
    end_normal_count = max(1, normal_count - start_normal_count)
    return start_normal_count, degrade_count, bad_count, recover_count, end_normal_count


def make_phase(args, rng, interval_ranges, index, condition, severity):
    normal_loss_low = 0.001
    normal_loss_high = 0.006
    good_capacity_low = max(args.capacity_min_mbps, args.capacity_max_mbps * 0.75)
    good_capacity_high = args.capacity_max_mbps
    bad_capacity_high = min(args.capacity_max_mbps, args.capacity_min_mbps * 3.0)

    if condition in {"normal-start", "normal-end"}:
        loss = rng.uniform(normal_loss_low, normal_loss_high)
        capacity = sample_range(rng, good_capacity_low, good_capacity_high)
        rtt = sample_range(rng, args.rtt_min_ms, args.rtt_min_ms + (args.rtt_max_ms - args.rtt_min_ms) * 0.25)
    else:
        jitter = rng.uniform(-0.12, 0.12)
        effective_severity = clamp(severity + jitter, 0.0, 1.0)
        loss_center = lerp(normal_loss_high, args.loss_max, effective_severity)
        loss = clamp(
            loss_center + rng.uniform(-0.012, 0.012),
            normal_loss_low,
            args.loss_max,
        )
        if condition == "bad-variable":
            loss = sample_range(rng, args.severe_loss_min, args.loss_max)
            capacity = sample_range(rng, args.capacity_min_mbps, bad_capacity_high)
            rtt = sample_range(rng,
                               args.rtt_min_ms + (args.rtt_max_ms - args.rtt_min_ms) * 0.45,
                               args.rtt_max_ms)
        else:
            capacity_center = lerp(good_capacity_high, args.capacity_min_mbps, effective_severity)
            capacity = clamp(
                capacity_center + rng.uniform(-0.25, 0.25),
                args.capacity_min_mbps,
                args.capacity_max_mbps,
            )
            rtt_center = lerp(args.rtt_min_ms, args.rtt_max_ms, effective_severity)
            rtt = clamp(
                rtt_center + rng.uniform(-12.0, 12.0),
                args.rtt_min_ms,
                args.rtt_max_ms,
            )

    min_interval, max_interval = sample_interval(rng, interval_ranges)
    return {
        "name": f"{condition}-{index + 1:02d}",
        "condition": condition,
        "duration": args.phase_duration,
        "rtt_ms": round(rtt, 3),
        "capacity_mbps": round(capacity, 3),
        "loss": round(loss, 6),
        "traffic_mode": args.traffic_mode,
        "min_send_interval_ms": min_interval,
        "max_send_interval_ms": max_interval,
        "packet_size_bytes": args.packet_size,
    }


def build_progressive_bad_recover_phases(args):
    rng = random.Random(args.seed)
    interval_ranges = parse_interval_ranges(args.send_interval_ranges)
    counts = allocate_progressive_counts(
        args.phase_count,
        max(0.0, min(args.normal_ratio, 1.0)),
        max(0.0, min(args.severe_ratio, 1.0)),
    )
    start_normal_count, degrade_count, bad_count, recover_count, end_normal_count = counts

    phases = []
    index = 0
    for _ in range(start_normal_count):
        phases.append(make_phase(args, rng, interval_ranges, index, "normal-start", 0.0))
        index += 1

    for step in range(degrade_count):
        ratio = (step + 1) / (degrade_count + 1)
        phases.append(make_phase(args, rng, interval_ranges, index, "degrading", ratio))
        index += 1

    for _ in range(bad_count):
        severity = rng.uniform(0.65, 1.0)
        phases.append(make_phase(args, rng, interval_ranges, index, "bad-variable", severity))
        index += 1

    for step in range(recover_count):
        ratio = 1.0 - ((step + 1) / (recover_count + 1))
        phases.append(make_phase(args, rng, interval_ranges, index, "recovering", ratio))
        index += 1

    for _ in range(end_normal_count):
        phases.append(make_phase(args, rng, interval_ranges, index, "normal-end", 0.0))
        index += 1

    return phases


def apply_fixed_phase_traffic(args):
    phases = []
    for phase in DEFAULT_PHASES:
        updated = phase.copy()
        updated.update(
            {
                "condition": phase["name"],
                "traffic_mode": args.traffic_mode,
                "min_send_interval_ms": args.fixed_min_send_interval_ms,
                "max_send_interval_ms": args.fixed_max_send_interval_ms,
                "packet_size_bytes": args.packet_size,
            }
        )
        phases.append(updated)
    return phases


def run_command(command, cwd, timeout):
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {completed.returncode}:\n"
            f"{' '.join(command)}\n{completed.stdout}"
        )
    return completed.stdout


def parse_throughput(output):
    match = THROUGHPUT_RE.search(output)
    if not match:
        raise RuntimeError(f"Cannot parse throughput from output:\n{output}")
    return float(match.group(1))


def build_programs(skip_quic):
    if not TCP_EXP.exists():
        run_command(["./ns3", "build", "tcp-exp"], ROOT, timeout=300)
    elif TCP_SOURCE.exists() and TCP_SOURCE.stat().st_mtime > TCP_EXP.stat().st_mtime:
        run_command(["./ns3", "build", "tcp-exp"], ROOT, timeout=300)
    if skip_quic:
        return
    if not QUIC_ROOT.exists():
        raise FileNotFoundError(f"QUIC ns-3.32 directory not found: {QUIC_ROOT}")
    if not QUIC_EXP.exists():
        run_command(
            [
                "/usr/bin/python3",
                "./waf",
                "configure",
                "--enable-examples",
                "--disable-python",
                "--disable-werror",
            ],
            QUIC_ROOT,
            timeout=120,
        )
        run_command(["/usr/bin/python3", "./waf", "build"], QUIC_ROOT, timeout=600)


def run_tcp(protocol,
            rtt_ms,
            capacity_mbps,
            loss_rate,
            duration,
            traffic_mode,
            min_interval_ms,
            max_interval_ms,
            packet_size):
    one_way_delay_ms = rtt_ms / 2.0
    stop_time = 1.0 + duration
    command = [
        str(TCP_EXP),
        f"--tcpType={TCP_TYPES[protocol]}",
        "--accessDelay=0ms",
        "--egressDelay=0ms",
        f"--bottleneckRate={capacity_mbps:g}Mbps",
        f"--bottleneckDelay={one_way_delay_ms:.6f}ms",
        f"--bottleneckLoss={loss_rate:g}",
        f"--trafficMode={traffic_mode}",
        f"--segmentSize={packet_size}",
        f"--minSendIntervalMs={min_interval_ms:g}",
        f"--maxSendIntervalMs={max_interval_ms:g}",
        f"--stopTime={stop_time:g}",
    ]
    output = run_command(command, ROOT, timeout=max(90, int(duration * 12)))
    return parse_throughput(output), " ".join(command)


def run_quic(rtt_ms, capacity_mbps, loss_rate, duration):
    one_way_delay_ms = rtt_ms / 2.0
    command = [
        str(QUIC_EXP),
        "--transport_prot=QuicBbr",
        f"--bandwidth={capacity_mbps:g}Mbps",
        f"--delay={one_way_delay_ms:.6f}ms",
        "--access_delay=0ms",
        f"--error_p={loss_rate:g}",
        f"--duration={duration:g}",
    ]
    output = run_command(command, QUIC_ROOT, timeout=max(120, int(duration * 30)))
    return parse_throughput(output), " ".join(command)


def decide_protocol(rtt_ms, capacity_mbps, loss_rate):
    """A deliberately simple, explainable decision tree."""
    if rtt_ms >= 100 and 2.0 <= capacity_mbps <= 4.0 and 0.01 <= loss_rate <= 0.02:
        return "QUIC_BBR", "high RTT + medium capacity + moderate loss -> QUIC_BBR"
    if loss_rate >= 0.02:
        return "TCP_BBR", "high loss -> TCP_BBR"
    if loss_rate <= 0.005:
        return "TCP_CUBIC", "low loss and stable cable -> TCP_CUBIC"
    if rtt_ms >= 100:
        return "TCP_BBR", "high RTT -> TCP_BBR"
    return "TCP_CUBIC", "low RTT and low loss -> TCP_CUBIC"


def softmax(values):
    high = max(values)
    exps = [math.exp(value - high) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def load_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def predict_trained_tree(model, rtt_ms, capacity_mbps, loss_rate):
    sample = {
        "rtt_ms": rtt_ms,
        "capacity_mbps": capacity_mbps,
        "packet_loss_rate": loss_rate,
    }
    node = model["tree"]
    path_parts = []
    while node["type"] == "split":
        feature = node["feature"]
        threshold = node["threshold"]
        if sample[feature] <= threshold:
            path_parts.append(f"{feature}<={threshold:.6g}")
            node = node["left"]
        else:
            path_parts.append(f"{feature}>{threshold:.6g}")
            node = node["right"]
    return node["prediction"], "trained tree: " + " and ".join(path_parts)


def predict_mlp(model, rtt_ms, capacity_mbps, loss_rate):
    sample = {
        "rtt_ms": rtt_ms,
        "capacity_mbps": capacity_mbps,
        "packet_loss_rate": loss_rate,
    }
    x = []
    for feature in model["features"]:
        low = model["mins"][feature]
        high = model["maxs"][feature]
        x.append((sample[feature] - low) / (high - low) if high > low else 0.0)
    hidden = [
        math.tanh(model["b1"][h] + sum(model["w1"][h][i] * x[i] for i in range(len(x))))
        for h in range(model["hidden_size"])
    ]
    logits = [
        model["b2"][o] + sum(model["w2"][o][h] * hidden[h] for h in range(model["hidden_size"]))
        for o in range(len(model["labels"]))
    ]
    probs = softmax(logits)
    best = max(range(len(probs)), key=lambda index: probs[index])
    probability_text = ", ".join(
        f"{label}={probs[index]:.2f}" for index, label in enumerate(model["labels"])
    )
    return model["labels"][best], "mlp probabilities: " + probability_text


def select_protocol(selector, model, rtt_ms, capacity_mbps, loss_rate):
    if selector == "manual-tree":
        return decide_protocol(rtt_ms, capacity_mbps, loss_rate)
    if selector == "trained-tree":
        return predict_trained_tree(model, rtt_ms, capacity_mbps, loss_rate)
    if selector == "mlp":
        return predict_mlp(model, rtt_ms, capacity_mbps, loss_rate)
    raise ValueError(f"Unknown selector: {selector}")


def evaluate_phase(protocol, phase, cache):
    key = (
        protocol,
        phase["rtt_ms"],
        phase["capacity_mbps"],
        phase["loss"],
        phase["duration"],
        phase.get("traffic_mode", "bulk"),
        phase.get("min_send_interval_ms", 1.0),
        phase.get("max_send_interval_ms", 10.0),
        phase.get("packet_size_bytes", 1448),
    )
    if key in cache:
        return cache[key]

    if protocol == "QUIC_BBR":
        throughput, command = run_quic(
            phase["rtt_ms"],
            phase["capacity_mbps"],
            phase["loss"],
            phase["duration"],
        )
    else:
        throughput, command = run_tcp(
            protocol,
            phase["rtt_ms"],
            phase["capacity_mbps"],
            phase["loss"],
            phase["duration"],
            phase.get("traffic_mode", "bulk"),
            phase.get("min_send_interval_ms", 1.0),
            phase.get("max_send_interval_ms", 10.0),
            phase.get("packet_size_bytes", 1448),
        )

    result = {
        "throughput_mbps": throughput,
        "delivered_mbit": throughput * phase["duration"],
        "utilization": throughput / phase["capacity_mbps"]
        if phase["capacity_mbps"] > 0
        else 0,
        "command": command,
    }
    cache[key] = result
    return result


def aggregate(rows, scheme_name):
    scheme_rows = [row for row in rows if row["scheme"] == scheme_name]
    total_duration = sum(row["duration_s"] for row in scheme_rows)
    total_delivered = sum(row["delivered_mbit"] for row in scheme_rows)
    total_capacity_time = sum(
        row["capacity_mbps"] * row["duration_s"] for row in scheme_rows
    )
    avg_rtt = sum(row["rtt_ms"] * row["duration_s"] for row in scheme_rows) / total_duration
    return {
        "scheme": scheme_name,
        "total_duration_s": total_duration,
        "delivered_mbit": total_delivered,
        "average_throughput_mbps": total_delivered / total_duration,
        "bandwidth_utilization": total_delivered / total_capacity_time
        if total_capacity_time > 0
        else 0,
        "weighted_rtt_ms": avg_rtt,
    }


def traffic_metrics(phase):
    min_interval = phase.get("min_send_interval_ms", 1.0)
    max_interval = phase.get("max_send_interval_ms", 10.0)
    mean_interval = (min_interval + max_interval) / 2.0
    packet_size = phase.get("packet_size_bytes", 1448)
    offered_load = packet_size * 8.0 / (mean_interval / 1000.0) / 1_000_000.0
    return mean_interval, offered_load


def write_phase_csv(path, rows):
    fieldnames = [
        "scheme",
        "phase",
        "condition",
        "selected_protocol",
        "decision_reason",
        "duration_s",
        "rtt_ms",
        "capacity_mbps",
        "packet_loss_rate",
        "traffic_mode",
        "packet_size_bytes",
        "min_send_interval_ms",
        "max_send_interval_ms",
        "mean_send_interval_ms",
        "offered_load_mbps",
        "throughput_mbps",
        "delivered_mbit",
        "bandwidth_utilization",
        "command",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, summaries):
    fieldnames = [
        "scheme",
        "total_duration_s",
        "delivered_mbit",
        "average_throughput_mbps",
        "bandwidth_utilization",
        "weighted_rtt_ms",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_scenario_config(path, args, phases):
    config = {
        "scenario_mode": args.scenario_mode,
        "phase_count": args.phase_count,
        "phase_duration": args.phase_duration,
        "normal_ratio": args.normal_ratio,
        "severe_ratio": args.severe_ratio,
        "rtt_min_ms": args.rtt_min_ms,
        "rtt_max_ms": args.rtt_max_ms,
        "capacity_min_mbps": args.capacity_min_mbps,
        "capacity_max_mbps": args.capacity_max_mbps,
        "severe_loss_min": args.severe_loss_min,
        "loss_max": args.loss_max,
        "traffic_mode": args.traffic_mode,
        "send_interval_ranges": args.send_interval_ranges,
        "packet_size": args.packet_size,
        "seed": args.seed,
        "phases": phases,
    }
    with path.open("w") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)


def write_markdown(path, phase_rows, summaries, selector):
    with path.open("w") as stream:
        stream.write("# Adaptive Protocol Selection Summary\n\n")
        stream.write(f"Selector: `{selector}`\n\n")
        if selector == "manual-tree":
            stream.write("## Decision Tree\n\n")
            stream.write("```text\n")
            stream.write("if RTT >= 100ms and 2 <= capacity <= 4Mbps and 0.01 <= loss <= 0.02: QUIC_BBR\n")
            stream.write("else if loss >= 0.02: TCP_BBR\n")
            stream.write("else if loss <= 0.005: TCP_CUBIC\n")
            stream.write("else if RTT >= 100ms: TCP_BBR\n")
            stream.write("else: TCP_CUBIC\n")
            stream.write("```\n\n")
        stream.write("## Phase Decisions\n\n")
        stream.write("| Phase | Condition | RTT | Capacity | Loss | Send interval | Selected | Reason |\n")
        stream.write("|---|---|---:|---:|---:|---:|---|---|\n")
        for row in phase_rows:
            if row["scheme"] != "ADAPTIVE_TREE":
                continue
            stream.write(
                f"| {row['phase']} | {row['condition']} | {row['rtt_ms']:.3f} | "
                f"{row['capacity_mbps']:g} | {row['packet_loss_rate']:g} | "
                f"{row['min_send_interval_ms']:g}-{row['max_send_interval_ms']:g} ms | "
                f"{row['selected_protocol']} | {row['decision_reason']} |\n"
            )
        stream.write("\n## Overall Results\n\n")
        stream.write("| Scheme | Avg throughput | Utilization | Delivered |\n")
        stream.write("|---|---:|---:|---:|\n")
        for summary in summaries:
            stream.write(
                f"| {summary['scheme']} | "
                f"{summary['average_throughput_mbps']:.6f} Mbit/s | "
                f"{summary['bandwidth_utilization']:.3f} | "
                f"{summary['delivered_mbit']:.6f} Mbit |\n"
            )


def plot_results(output_dir, phase_rows, summaries):
    if plt is None:
        print("matplotlib is not installed; skip plotting adaptive result images.")
        return

    ordered = ["TCP_CUBIC", "TCP_BBR", "QUIC_BBR", "ADAPTIVE_TREE"]
    summary_map = {row["scheme"]: row for row in summaries}
    ordered = [name for name in ordered if name in summary_map]
    values = [summary_map[name]["average_throughput_mbps"] for name in ordered]

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(ordered, values, color=[COLORS[name] for name in ordered])
    plt.ylabel("Average throughput (Mbit/s)")
    plt.title("Fixed protocols vs decision-tree adaptive selection")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "adaptive-throughput-comparison.png", dpi=180)
    plt.close()

    adaptive_rows = [row for row in phase_rows if row["scheme"] == "ADAPTIVE_TREE"]
    starts = []
    cursor = 0.0
    for row in adaptive_rows:
        starts.append(cursor)
        cursor += row["duration_s"]

    plt.figure(figsize=(9, 2.8))
    y = 0
    for start, row in zip(starts, adaptive_rows):
        plt.barh(
            y,
            row["duration_s"],
            left=start,
            color=COLORS[row["selected_protocol"]],
            edgecolor="white",
        )
        plt.text(
            start + row["duration_s"] / 2,
            y,
            f"{row['phase']}\n{row['selected_protocol']}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    plt.yticks([])
    plt.xlabel("Time (s)")
    plt.title("Adaptive protocol timeline")
    plt.tight_layout()
    plt.savefig(output_dir / "adaptive-protocol-timeline.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="scratch/dynamic-link-results")
    parser.add_argument("--skip-quic", action="store_true")
    parser.add_argument(
        "--selector",
        choices=["manual-tree", "trained-tree", "mlp"],
        default="manual-tree",
    )
    parser.add_argument("--tree-model", default=str(DEFAULT_TREE_MODEL))
    parser.add_argument("--mlp-model", default=str(DEFAULT_MLP_MODEL))
    parser.add_argument(
        "--scenario-mode",
        choices=["fixed", "continuous-bad-heavy", "progressive-bad-recover"],
        default="progressive-bad-recover",
    )
    parser.add_argument("--phase-count", type=int, default=18)
    parser.add_argument("--phase-duration", type=float, default=3.0)
    parser.add_argument("--normal-ratio", type=float, default=0.1)
    parser.add_argument("--severe-ratio", type=float, default=0.55)
    parser.add_argument("--rtt-min-ms", type=float, default=80.0)
    parser.add_argument("--rtt-max-ms", type=float, default=260.0)
    parser.add_argument("--capacity-min-mbps", type=float, default=0.3)
    parser.add_argument("--capacity-max-mbps", type=float, default=4.0)
    parser.add_argument("--severe-loss-min", type=float, default=0.05)
    parser.add_argument("--loss-max", type=float, default=0.15)
    parser.add_argument(
        "--traffic-mode",
        choices=["bulk", "random-interval"],
        default="random-interval",
    )
    parser.add_argument("--send-interval-ranges", default="1:8,8:30,30:120")
    parser.add_argument("--fixed-min-send-interval-ms", type=float, default=1.0)
    parser.add_argument("--fixed-max-send-interval-ms", type=float, default=10.0)
    parser.add_argument("--packet-size", type=int, default=1448)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    protocols = [protocol for protocol in PROTOCOLS if protocol != "QUIC_BBR" or not args.skip_quic]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building experiment programs...")
    build_programs(args.skip_quic)
    selector_model = None
    if args.selector == "trained-tree":
        selector_model = load_json(args.tree_model)
    elif args.selector == "mlp":
        selector_model = load_json(args.mlp_model)

    if args.phase_count <= 0 or args.phase_duration <= 0:
        raise ValueError("phase-count and phase-duration must be positive")
    if args.severe_loss_min > args.loss_max:
        raise ValueError("severe-loss-min must be <= loss-max")
    if args.capacity_min_mbps <= 0 or args.capacity_max_mbps <= 0:
        raise ValueError("capacity bounds must be positive")

    if args.scenario_mode == "fixed":
        phases = apply_fixed_phase_traffic(args)
    elif args.scenario_mode == "continuous-bad-heavy":
        phases = build_continuous_bad_heavy_phases(args)
    else:
        phases = build_progressive_bad_recover_phases(args)

    cache = {}
    phase_rows = []

    for phase in phases:
        mean_interval_ms, offered_load_mbps = traffic_metrics(phase)
        print(
            f"Phase {phase['name']}: RTT={phase['rtt_ms']:.3f} ms, "
            f"capacity={phase['capacity_mbps']:g} Mbps, loss={phase['loss']:g}, "
            f"interval={phase.get('min_send_interval_ms', 1.0):g}-"
            f"{phase.get('max_send_interval_ms', 10.0):g} ms",
            flush=True,
        )

        for protocol in protocols:
            result = evaluate_phase(protocol, phase, cache)
            phase_rows.append(
                {
                    "scheme": protocol,
                    "phase": phase["name"],
                    "condition": phase.get("condition", phase["name"]),
                    "selected_protocol": protocol,
                    "decision_reason": "fixed protocol",
                    "duration_s": phase["duration"],
                    "rtt_ms": phase["rtt_ms"],
                    "capacity_mbps": phase["capacity_mbps"],
                    "packet_loss_rate": phase["loss"],
                    "traffic_mode": phase.get("traffic_mode", "bulk") if protocol != "QUIC_BBR" else "quic-bulksend",
                    "packet_size_bytes": phase.get("packet_size_bytes", 1448),
                    "min_send_interval_ms": phase.get("min_send_interval_ms", 1.0),
                    "max_send_interval_ms": phase.get("max_send_interval_ms", 10.0),
                    "mean_send_interval_ms": mean_interval_ms,
                    "offered_load_mbps": offered_load_mbps,
                    "throughput_mbps": result["throughput_mbps"],
                    "delivered_mbit": result["delivered_mbit"],
                    "bandwidth_utilization": result["utilization"],
                    "command": result["command"],
                }
            )
            print(f"  {protocol}: {result['throughput_mbps']:.6f} Mbit/s", flush=True)

        selected_protocol, reason = select_protocol(
            args.selector,
            selector_model,
            phase["rtt_ms"], phase["capacity_mbps"], phase["loss"]
        )
        if selected_protocol == "QUIC_BBR" and args.skip_quic:
            selected_protocol = "TCP_BBR"
            reason = "QUIC skipped; fallback to TCP_BBR"
        result = evaluate_phase(selected_protocol, phase, cache)
        phase_rows.append(
            {
                "scheme": "ADAPTIVE_TREE",
                "phase": phase["name"],
                "condition": phase.get("condition", phase["name"]),
                "selected_protocol": selected_protocol,
                "decision_reason": reason,
                "duration_s": phase["duration"],
                "rtt_ms": phase["rtt_ms"],
                "capacity_mbps": phase["capacity_mbps"],
                "packet_loss_rate": phase["loss"],
                "traffic_mode": phase.get("traffic_mode", "bulk") if selected_protocol != "QUIC_BBR" else "quic-bulksend",
                "packet_size_bytes": phase.get("packet_size_bytes", 1448),
                "min_send_interval_ms": phase.get("min_send_interval_ms", 1.0),
                "max_send_interval_ms": phase.get("max_send_interval_ms", 10.0),
                "mean_send_interval_ms": mean_interval_ms,
                "offered_load_mbps": offered_load_mbps,
                "throughput_mbps": result["throughput_mbps"],
                "delivered_mbit": result["delivered_mbit"],
                "bandwidth_utilization": result["utilization"],
                "command": result["command"],
            }
        )
        print(f"  ADAPTIVE_TREE -> {selected_protocol}: {result['throughput_mbps']:.6f} Mbit/s")

    schemes = protocols + ["ADAPTIVE_TREE"]
    summaries = [aggregate(phase_rows, scheme) for scheme in schemes]

    write_phase_csv(output_dir / "phase-results.csv", phase_rows)
    write_summary_csv(output_dir / "summary-results.csv", summaries)
    write_scenario_config(output_dir / "link-scenario-config.json", args, phases)
    write_markdown(output_dir / "result-summary.md", phase_rows, summaries, args.selector)
    plot_results(output_dir, phase_rows, summaries)

    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
