#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Build an expanded protocol-selector learning dataset from existing reliable
# transport sweep results and ocean observation sample files.

import argparse
import csv
import math
import zlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABELS = ["TCP_CUBIC", "TCP_BBR", "QUIC_BBR"]
DEFAULT_INPUTS = "02_protocol_evaluation/sample-results/protocol-evaluation-results.csv"


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def classify_sample(path):
    name = path.name
    if "current" in name or name.endswith(".log"):
        return "adcp_current_log"
    if "wave" in name:
        return "wave_sensor"
    if name == "weather-data.txt":
        return "weather_tsv"
    if "weather" in name or "buoy" in name:
        return "weather_csv"
    return "other"


def iter_chunks(data, packet_size):
    for offset in range(0, len(data), packet_size):
        chunk = data[offset : offset + packet_size]
        if chunk:
            yield chunk


def profile_data_type(files, packet_size, max_profile_bytes):
    total_bytes = sum(path.stat().st_size for path in files)
    profile_bytes = bytearray()
    for path in files:
        if len(profile_bytes) >= max_profile_bytes:
            break
        remaining = max_profile_bytes - len(profile_bytes)
        with path.open("rb") as stream:
            profile_bytes.extend(stream.read(remaining))

    original_profile_bytes = len(profile_bytes)
    if original_profile_bytes == 0:
        return {
            "sample_bytes": total_bytes,
            "profile_bytes": 0,
            "chunk_count": 0,
            "avg_payload_bytes": 0.0,
            "compression_ratio": 1.0,
        }

    compressed_bytes = 0
    chunk_count = 0
    for chunk in iter_chunks(profile_bytes, packet_size):
        compressed_bytes += len(zlib.compress(bytes(chunk), level=1))
        chunk_count += 1

    full_chunk_count = math.ceil(total_bytes / packet_size) if total_bytes else 0
    avg_payload = total_bytes / full_chunk_count if full_chunk_count else 0.0
    return {
        "sample_bytes": total_bytes,
        "profile_bytes": original_profile_bytes,
        "chunk_count": full_chunk_count,
        "avg_payload_bytes": avg_payload,
        "compression_ratio": compressed_bytes / original_profile_bytes,
    }


def build_sample_profiles(sample_dir, packet_sizes, max_profile_bytes):
    files_by_type = defaultdict(list)
    for path in sorted(sample_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".log", ".csv"}:
            files_by_type[classify_sample(path)].append(path)

    profiles = []
    data_types = sorted(files_by_type)
    data_type_ids = {name: index for index, name in enumerate(data_types)}
    for data_type in data_types:
        files = files_by_type[data_type]
        for packet_size in packet_sizes:
            profile = profile_data_type(files, packet_size, max_profile_bytes)
            profile.update(
                {
                    "data_type": data_type,
                    "data_type_id": data_type_ids[data_type],
                    "packet_size_bytes": packet_size,
                    "files": ";".join(path.name for path in files),
                }
            )
            profiles.append(profile)
    return profiles


def read_protocol_results(paths):
    grouped = defaultdict(dict)
    metadata = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                if row.get("status", "ok") != "ok" or not row.get("throughput_mbps"):
                    continue
                protocol = row.get("protocol", "")
                if protocol not in LABELS:
                    continue
                rtt_ms = float(row["rtt_ms"])
                capacity_mbps = float(row["capacity_mbps"])
                loss = float(row["packet_loss_rate"])
                scenario = row.get("scenario", f"rtt-{rtt_ms:g}ms")
                packet_size = int(float(row.get("packet_size_bytes", 1448) or 1448))
                min_interval = float(row.get("min_send_interval_ms", 0.0) or 0.0)
                max_interval = float(row.get("max_send_interval_ms", 0.0) or 0.0)
                mean_interval = float(row.get("mean_send_interval_ms", 0.0) or 0.0)
                offered_load = float(row.get("offered_load_mbps", 0.0) or 0.0)
                traffic_mode = row.get("traffic_mode", "bulk")
                if traffic_mode == "quic-bulksend" and min_interval > 0:
                    traffic_mode = "random-interval"
                key = (scenario, rtt_ms, capacity_mbps, loss, packet_size, min_interval, max_interval)
                grouped[key][protocol] = float(row["throughput_mbps"])
                metadata[key] = {
                    "scenario": scenario,
                    "rtt_ms": rtt_ms,
                    "capacity_mbps": capacity_mbps,
                    "packet_loss_rate": loss,
                    "traffic_mode": traffic_mode,
                    "protocol_packet_size_bytes": packet_size,
                    "min_send_interval_ms": min_interval,
                    "max_send_interval_ms": max_interval,
                    "mean_send_interval_ms": mean_interval,
                    "offered_load_mbps": offered_load,
                }

    samples = []
    for key in sorted(grouped):
        throughputs = grouped[key]
        if not all(protocol in throughputs for protocol in LABELS):
            continue
        label = max(LABELS, key=lambda protocol: throughputs[protocol])
        base = metadata[key].copy()
        base.update(
            {
                "tcp_cubic_mbps": throughputs["TCP_CUBIC"],
                "tcp_bbr_mbps": throughputs["TCP_BBR"],
                "quic_bbr_mbps": throughputs["QUIC_BBR"],
                "best_protocol": label,
            }
        )
        samples.append(base)
    return samples


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, base_samples, profiles, expanded_rows):
    label_counts = defaultdict(int)
    for row in expanded_rows:
        label_counts[row["best_protocol"]] += 1

    with path.open("w") as stream:
        stream.write("# Expanded Learning Data Summary\n\n")
        stream.write(f"Base protocol comparison samples: {len(base_samples)}\n\n")
        stream.write(f"Ocean sample profiles: {len(profiles)}\n\n")
        stream.write(f"Expanded learning samples: {len(expanded_rows)}\n\n")
        stream.write("## Label distribution\n\n")
        for label in LABELS:
            stream.write(f"- {label}: {label_counts[label]}\n")
        stream.write("\n## Notes\n\n")
        stream.write(
            "The expanded dataset combines measured protocol throughput results with packet-size "
            "and ocean-data-type features extracted from local sample files. Protocol labels still "
            "come from the ns-3 reliable transport sweep results.\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=DEFAULT_INPUTS)
    parser.add_argument("--sample-dir", default="05_datasets_and_generators/raw-samples")
    parser.add_argument("--packet-sizes", default="256,512,1024,1448,2048,4096")
    parser.add_argument("--max-profile-bytes", type=int, default=5_000_000)
    parser.add_argument("--output-dir", default="05_datasets_and_generators/training-data/expanded-learning-results")
    args = parser.parse_args()

    paths = [ROOT / item.strip() for item in args.inputs.split(",") if item.strip()]
    sample_dir = ROOT / args.sample_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    packet_sizes = parse_int_list(args.packet_sizes)
    base_samples = read_protocol_results(paths)
    if not base_samples:
        raise RuntimeError("No complete TCP_CUBIC/TCP_BBR/QUIC_BBR comparison samples found.")

    profiles = build_sample_profiles(sample_dir, packet_sizes, args.max_profile_bytes)
    if not profiles:
        raise RuntimeError("No usable ocean observation sample files found.")

    expanded_rows = []
    for base in base_samples:
        for profile in profiles:
            row = base.copy()
            row.update(
                {
                    "packet_size_bytes": profile["packet_size_bytes"],
                    "data_type": profile["data_type"],
                    "data_type_id": profile["data_type_id"],
                    "sample_bytes": profile["sample_bytes"],
                    "profile_bytes": profile["profile_bytes"],
                    "chunk_count": profile["chunk_count"],
                    "avg_payload_bytes": f"{profile['avg_payload_bytes']:.6f}",
                    "compression_ratio": f"{profile['compression_ratio']:.6f}",
                }
            )
            expanded_rows.append(row)

    profile_fields = [
        "data_type",
        "data_type_id",
        "packet_size_bytes",
        "sample_bytes",
        "profile_bytes",
        "chunk_count",
        "avg_payload_bytes",
        "compression_ratio",
        "files",
    ]
    training_fields = [
        "scenario",
        "rtt_ms",
        "capacity_mbps",
        "packet_loss_rate",
        "traffic_mode",
        "protocol_packet_size_bytes",
        "min_send_interval_ms",
        "max_send_interval_ms",
        "mean_send_interval_ms",
        "offered_load_mbps",
        "packet_size_bytes",
        "data_type",
        "data_type_id",
        "sample_bytes",
        "profile_bytes",
        "chunk_count",
        "avg_payload_bytes",
        "compression_ratio",
        "tcp_cubic_mbps",
        "tcp_bbr_mbps",
        "quic_bbr_mbps",
        "best_protocol",
    ]
    write_csv(output_dir / "sample-profiles.csv", profile_fields, profiles)
    write_csv(output_dir / "expanded-training-samples.csv", training_fields, expanded_rows)

    label_rows = [{"protocol": label, "count": 0} for label in LABELS]
    counts = defaultdict(int)
    for row in expanded_rows:
        counts[row["best_protocol"]] += 1
    for row in label_rows:
        row["count"] = counts[row["protocol"]]
    write_csv(output_dir / "training-label-distribution.csv", ["protocol", "count"], label_rows)
    write_summary(output_dir / "result-summary.md", base_samples, profiles, expanded_rows)

    print(f"Base samples: {len(base_samples)}")
    print(f"Sample profiles: {len(profiles)}")
    print(f"Expanded samples: {len(expanded_rows)}")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
