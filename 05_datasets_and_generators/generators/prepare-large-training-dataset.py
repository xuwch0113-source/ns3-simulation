#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Build a large tabular training dataset for adaptive protocol selection.

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = ["TCP_CUBIC", "TCP_BBR", "QUIC_BBR"]
DEFAULT_INPUTS = "02_protocol_evaluation/sample-results/protocol-evaluation-results.csv"


def as_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def clamp(value, low, high):
    return max(low, min(high, value))


def safe_time(file_size_mb, throughput_mbps):
    if throughput_mbps <= 0:
        return 0.0
    return file_size_mb * 8.0 / throughput_mbps


def read_protocol_anchors(paths, default_file_size_mb):
    grouped = defaultdict(dict)
    metadata = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                if row.get("status", "ok") != "ok" or row.get("throughput_mbps", "") == "":
                    continue
                protocol = row.get("protocol", "")
                if protocol not in PROTOCOLS:
                    continue

                scenario = row.get("scenario", "")
                rtt_ms = as_float(row, "rtt_ms")
                capacity_mbps = as_float(row, "capacity_mbps")
                loss = as_float(row, "packet_loss_rate")
                packet_size = int(as_float(row, "packet_size_bytes", 1448))
                min_interval = as_float(row, "min_send_interval_ms", 1.0)
                max_interval = as_float(row, "max_send_interval_ms", 10.0)
                file_size_mb = as_float(row, "file_size_mb", default_file_size_mb)
                throughput = as_float(row, "throughput_mbps")
                file_time = as_float(row, "estimated_file_transfer_time_s")
                if file_time <= 0:
                    file_time = safe_time(file_size_mb, throughput)

                key = (
                    scenario,
                    round(rtt_ms, 6),
                    round(capacity_mbps, 6),
                    round(loss, 6),
                    packet_size,
                    round(min_interval, 6),
                    round(max_interval, 6),
                )
                grouped[key][protocol] = {
                    "throughput_mbps": throughput,
                    "file_time_s": file_time,
                }
                metadata[key] = {
                    "source_scenario": scenario,
                    "base_rtt_ms": rtt_ms,
                    "base_capacity_mbps": capacity_mbps,
                    "base_packet_loss_rate": loss,
                    "base_packet_size_bytes": packet_size,
                    "base_min_interval_ms": min_interval,
                    "base_max_interval_ms": max_interval,
                    "base_file_size_mb": file_size_mb,
                }

    anchors = []
    for key, values in grouped.items():
        if not all(protocol in values for protocol in PROTOCOLS):
            continue
        deliverable = [
            protocol for protocol in PROTOCOLS if values[protocol]["throughput_mbps"] > 0
        ]
        if deliverable:
            best = min(deliverable, key=lambda protocol: values[protocol]["file_time_s"])
        else:
            best = "NO_DELIVERY"
        anchor = metadata[key].copy()
        for protocol in PROTOCOLS:
            prefix = protocol.lower()
            anchor[f"{prefix}_mbps"] = values[protocol]["throughput_mbps"]
            anchor[f"{prefix}_file_time_s"] = values[protocol]["file_time_s"]
        anchor["base_best_protocol"] = best
        anchors.append(anchor)
    return anchors


def read_profiles(path):
    profiles = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            profiles.append(
                {
                    "data_type": row["data_type"],
                    "data_type_id": int(float(row.get("data_type_id", 0))),
                    "profile_packet_size_bytes": int(float(row.get("packet_size_bytes", 1448))),
                    "sample_bytes": int(float(row.get("sample_bytes", 0))),
                    "compression_ratio": as_float(row, "compression_ratio", 1.0),
                }
            )
    return profiles


def weighted_anchor_pool(anchors, quic_weight, no_delivery_weight):
    pool = []
    for anchor in anchors:
        label = anchor["base_best_protocol"]
        if label == "QUIC_BBR":
            repeats = max(1, int(round(quic_weight)))
        elif label == "NO_DELIVERY":
            repeats = 1 if no_delivery_weight >= 1 else 0
        else:
            repeats = 1
        pool.extend([anchor] * repeats)
    return pool or anchors


def choose_state(rng):
    draw = rng.random()
    if draw < 0.05:
        return "normal-start", rng.uniform(0.0, 0.12)
    if draw < 0.23:
        progress = rng.uniform(0.15, 0.85)
        return "degrading", progress
    if draw < 0.78:
        return "bad-variable", rng.uniform(0.6, 1.0)
    if draw < 0.95:
        progress = rng.uniform(0.15, 0.85)
        return "recovering", progress
    return "normal-end", rng.uniform(0.0, 0.12)


def random_packet_stats(rng, profile_packet_size):
    center = rng.choice([256, 512, 768, 1024, 1200, 1448, 2048, 4096, profile_packet_size])
    spread = rng.uniform(0.05, 0.35)
    low = max(128, int(center * (1.0 - spread)))
    high = min(8192, int(center * (1.0 + spread)))
    mean = rng.uniform(low, high)
    std = max(1.0, (high - low) / rng.uniform(4.0, 8.0))
    return low, high, mean, std


def random_interval_stats(rng):
    mode = rng.choice(["fast", "medium", "slow", "bursty"])
    if mode == "fast":
        low, high = rng.uniform(0.8, 3.0), rng.uniform(5.0, 12.0)
    elif mode == "medium":
        low, high = rng.uniform(5.0, 15.0), rng.uniform(20.0, 45.0)
    elif mode == "slow":
        low, high = rng.uniform(20.0, 60.0), rng.uniform(80.0, 160.0)
    else:
        low, high = rng.uniform(1.0, 10.0), rng.uniform(80.0, 220.0)
    if high < low:
        low, high = high, low
    mean = rng.uniform(low, high)
    std = max(0.1, (high - low) / rng.uniform(3.0, 7.0))
    return low, high, mean, std, mode


def adjust_protocol_metrics(anchor, rng, rtt_ms, capacity_mbps, loss, offered_load_mbps, traffic_mode):
    base_capacity = max(anchor["base_capacity_mbps"], 1e-6)
    base_loss = anchor["base_packet_loss_rate"]
    cap_factor = clamp(capacity_mbps / base_capacity, 0.35, 2.2)
    loss_delta = loss - base_loss
    sensitivities = {
        "TCP_CUBIC": 10.0,
        "TCP_BBR": 3.2,
        "QUIC_BBR": 5.5,
    }
    metrics = {}
    for protocol in PROTOCOLS:
        base = anchor[f"{protocol.lower()}_mbps"]
        if base <= 0:
            throughput = 0.0
        else:
            loss_factor = math.exp(-sensitivities[protocol] * loss_delta)
            rtt_factor = clamp(math.sqrt(max(anchor["base_rtt_ms"], 1.0) / max(rtt_ms, 1.0)), 0.65, 1.25)
            noise = rng.uniform(0.92, 1.08)
            throughput = base * cap_factor * loss_factor * rtt_factor * noise
            if traffic_mode == "random-interval":
                throughput = min(throughput, offered_load_mbps)
            throughput = max(0.0, throughput)
            if loss >= 0.18 and rng.random() < 0.45:
                throughput = 0.0
        metrics[protocol] = throughput
    return metrics


def build_rows(anchors, profiles, sample_count, seed, file_size_mb, quic_weight):
    rng = random.Random(seed)
    pool = weighted_anchor_pool(anchors, quic_weight=quic_weight, no_delivery_weight=0.8)
    rows = []
    prev = None

    for sample_id in range(sample_count):
        anchor = rng.choice(pool)
        profile = rng.choice(profiles)
        state_name, phase_progress = choose_state(rng)

        if state_name in {"normal-start", "normal-end"}:
            severity = rng.uniform(0.0, 0.12)
        elif state_name == "degrading":
            severity = phase_progress
        elif state_name == "recovering":
            severity = 1.0 - phase_progress
        else:
            severity = rng.uniform(0.6, 1.0)

        base_rtt = anchor["base_rtt_ms"]
        base_capacity = anchor["base_capacity_mbps"]
        base_loss = anchor["base_packet_loss_rate"]

        loss = clamp(base_loss + rng.uniform(-0.01, 0.01) + severity * rng.uniform(0.0, 0.08), 0.0, 0.2)
        capacity = clamp(base_capacity * rng.uniform(0.75, 1.25) * (1.15 - 0.65 * severity), 0.1, 5.0)
        rtt = clamp(base_rtt * rng.uniform(0.85, 1.15) * (1.0 + 0.45 * severity), 5.0, 420.0)

        packet_min, packet_max, packet_mean, packet_std = random_packet_stats(
            rng,
            profile["profile_packet_size_bytes"],
        )
        interval_min, interval_max, interval_mean, interval_std, interval_mode = random_interval_stats(rng)
        offered_load = packet_mean * 8.0 / (interval_mean / 1000.0) / 1_000_000.0
        traffic_mode = "random-interval"
        effective_file_size = file_size_mb * profile["compression_ratio"]

        if state_name == "degrading":
            loss_trend = abs(rng.uniform(0.001, 0.02))
            capacity_trend = -abs(rng.uniform(0.01, 0.3))
            rtt_trend = abs(rng.uniform(1.0, 18.0))
        elif state_name == "recovering":
            loss_trend = -abs(rng.uniform(0.001, 0.02))
            capacity_trend = abs(rng.uniform(0.01, 0.3))
            rtt_trend = -abs(rng.uniform(1.0, 18.0))
        elif state_name == "bad-variable":
            loss_trend = rng.uniform(-0.02, 0.02)
            capacity_trend = rng.uniform(-0.25, 0.25)
            rtt_trend = rng.uniform(-15.0, 15.0)
        else:
            loss_trend = rng.uniform(-0.003, 0.003)
            capacity_trend = rng.uniform(-0.05, 0.05)
            rtt_trend = rng.uniform(-3.0, 3.0)

        throughputs = adjust_protocol_metrics(
            anchor,
            rng,
            rtt,
            capacity,
            loss,
            offered_load,
            traffic_mode,
        )
        file_times = {
            protocol: safe_time(effective_file_size, throughputs[protocol])
            for protocol in PROTOCOLS
        }
        deliverable = [protocol for protocol in PROTOCOLS if throughputs[protocol] > 0]
        best_protocol = min(deliverable, key=lambda protocol: file_times[protocol]) if deliverable else "NO_DELIVERY"
        reward = 1.0 / file_times[best_protocol] if best_protocol != "NO_DELIVERY" else 0.0

        row = {
            "sample_id": sample_id,
            "source_scenario": anchor["source_scenario"],
            "state_name": state_name,
            "phase_progress": f"{phase_progress:.6f}",
            "rtt_ms": f"{rtt:.6f}",
            "capacity_mbps": f"{capacity:.6f}",
            "packet_loss_rate": f"{loss:.6f}",
            "loss_trend": f"{loss_trend:.6f}",
            "capacity_trend": f"{capacity_trend:.6f}",
            "rtt_trend": f"{rtt_trend:.6f}",
            "packet_size_min": packet_min,
            "packet_size_max": packet_max,
            "packet_size_mean": f"{packet_mean:.6f}",
            "packet_size_std": f"{packet_std:.6f}",
            "interval_min_ms": f"{interval_min:.6f}",
            "interval_max_ms": f"{interval_max:.6f}",
            "interval_mean_ms": f"{interval_mean:.6f}",
            "interval_std_ms": f"{interval_std:.6f}",
            "interval_mode": interval_mode,
            "offered_load_mbps": f"{offered_load:.6f}",
            "file_size_mb": f"{file_size_mb:.6f}",
            "effective_file_size_mb": f"{effective_file_size:.6f}",
            "traffic_mode": traffic_mode,
            "data_type": profile["data_type"],
            "data_type_id": profile["data_type_id"],
            "compression_ratio": f"{profile['compression_ratio']:.6f}",
            "prev_rtt_ms": f"{prev['rtt_ms']:.6f}" if prev else "",
            "prev_capacity_mbps": f"{prev['capacity_mbps']:.6f}" if prev else "",
            "prev_packet_loss_rate": f"{prev['packet_loss_rate']:.6f}" if prev else "",
            "prev_best_protocol": prev["best_protocol"] if prev else "",
            "tcp_cubic_mbps": f"{throughputs['TCP_CUBIC']:.6f}",
            "tcp_bbr_mbps": f"{throughputs['TCP_BBR']:.6f}",
            "quic_bbr_mbps": f"{throughputs['QUIC_BBR']:.6f}",
            "tcp_cubic_file_time_s": f"{file_times['TCP_CUBIC']:.6f}" if file_times["TCP_CUBIC"] > 0 else "",
            "tcp_bbr_file_time_s": f"{file_times['TCP_BBR']:.6f}" if file_times["TCP_BBR"] > 0 else "",
            "quic_bbr_file_time_s": f"{file_times['QUIC_BBR']:.6f}" if file_times["QUIC_BBR"] > 0 else "",
            "best_protocol": best_protocol,
            "reward": f"{reward:.9f}",
            "bandit_action_tcp_cubic_reward": f"{(1.0 / file_times['TCP_CUBIC']) if file_times['TCP_CUBIC'] > 0 else 0.0:.9f}",
            "bandit_action_tcp_bbr_reward": f"{(1.0 / file_times['TCP_BBR']) if file_times['TCP_BBR'] > 0 else 0.0:.9f}",
            "bandit_action_quic_bbr_reward": f"{(1.0 / file_times['QUIC_BBR']) if file_times['QUIC_BBR'] > 0 else 0.0:.9f}",
            "base_best_protocol": anchor["base_best_protocol"],
            "base_rtt_ms": f"{anchor['base_rtt_ms']:.6f}",
            "base_capacity_mbps": f"{anchor['base_capacity_mbps']:.6f}",
            "base_packet_loss_rate": f"{anchor['base_packet_loss_rate']:.6f}",
        }
        rows.append(row)
        prev = {
            "rtt_ms": rtt,
            "capacity_mbps": capacity,
            "packet_loss_rate": loss,
            "best_protocol": best_protocol,
        }
    return rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("No rows to write")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, anchors, profiles, rows):
    labels = Counter(row["best_protocol"] for row in rows)
    states = Counter(row["state_name"] for row in rows)
    data_types = Counter(row["data_type"] for row in rows)
    with path.open("w") as stream:
        stream.write("# Large Training Dataset Summary\n\n")
        stream.write(f"Protocol anchor conditions: {len(anchors)}\n\n")
        stream.write(f"Ocean data profiles: {len(profiles)}\n\n")
        stream.write(f"Generated training rows: {len(rows)}\n\n")
        stream.write("## Label distribution\n\n")
        for label, count in sorted(labels.items()):
            stream.write(f"- {label}: {count}\n")
        stream.write("\n## Dynamic state distribution\n\n")
        for label, count in sorted(states.items()):
            stream.write(f"- {label}: {count}\n")
        stream.write("\n## Data type distribution\n\n")
        for label, count in sorted(data_types.items()):
            stream.write(f"- {label}: {count}\n")
        stream.write("\n## Notes\n\n")
        stream.write(
            "Rows are generated from measured ns-3 protocol comparison anchors and expanded with "
            "random packet size, random send interval, dynamic link state, previous-window context, "
            "and ocean observation data profiles. This dataset is intended for LightGBM, XGBoost, "
            "Random Forest, TinyMLP, CatBoost, and contextual bandit experiments.\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=DEFAULT_INPUTS)
    parser.add_argument("--profiles", default="05_datasets_and_generators/training-data/ocean-sample-profiles.csv")
    parser.add_argument("--output-dir", default="05_datasets_and_generators/training-data/generated")
    parser.add_argument("--rows", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--file-size-mb", type=float, default=10.0)
    parser.add_argument("--quic-weight", type=float, default=5.0)
    args = parser.parse_args()

    input_paths = [ROOT / item.strip() for item in args.inputs.split(",") if item.strip()]
    profiles_path = ROOT / args.profiles
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    anchors = read_protocol_anchors(input_paths, args.file_size_mb)
    profiles = read_profiles(profiles_path)
    if not anchors:
        raise RuntimeError("No complete protocol anchors found.")
    if not profiles:
        raise RuntimeError("No ocean data profiles found.")

    rows = build_rows(
        anchors,
        profiles,
        sample_count=args.rows,
        seed=args.seed,
        file_size_mb=args.file_size_mb,
        quic_weight=args.quic_weight,
    )

    write_csv(output_dir / "adaptive-protocol-large-training-dataset.csv", rows)
    write_summary(output_dir / "dataset-summary.md", anchors, profiles, rows)
    print(f"Protocol anchor conditions: {len(anchors)}")
    print(f"Ocean data profiles: {len(profiles)}")
    print(f"Generated training rows: {len(rows)}")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
