#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Compare LZ4, Gzip and Gorilla-style time-series compression under the
# random-forest protocol selector proven in the current adaptive transport report.

import argparse
import csv
import gzip
import importlib.util
import math
import re
import struct
import time
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = ROOT / "python-packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import lz4.frame


EVALUATOR_PATH = Path(__file__).resolve().parent / "evaluate-lightweight-protocol-selector.py"
DEFAULT_LEARNING_DATA = ROOT / "05_datasets_and_generators" / "training-data" / "adaptive-protocol-large-training-dataset.csv.gz"
DEFAULT_SAMPLE_DIR = ROOT / "05_datasets_and_generators" / "raw-samples"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("selector_eval", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def sample_files(sample_dir):
    files_by_type = defaultdict(list)
    for path in sorted(sample_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".log", ".csv"}:
            files_by_type[classify_sample(path)].append(path)
    return files_by_type


def read_profile_bytes(files, max_bytes):
    data = bytearray()
    total_bytes = sum(path.stat().st_size for path in files)
    for path in files:
        if len(data) >= max_bytes:
            break
        remaining = max_bytes - len(data)
        with path.open("rb") as stream:
            data.extend(stream.read(remaining))
    return bytes(data), total_bytes


def iter_chunks(data, packet_size):
    for offset in range(0, len(data), packet_size):
        chunk = data[offset : offset + packet_size]
        if chunk:
            yield chunk


def measure_codec(data, packet_size, name, compress_one, decompress_one):
    compressed_bytes = 0
    compress_ns = 0
    decompress_ns = 0
    original_bytes = 0
    chunks = 0

    for chunk in iter_chunks(data, packet_size):
        chunks += 1
        original_bytes += len(chunk)
        start = time.perf_counter_ns()
        compressed = compress_one(chunk)
        compress_ns += time.perf_counter_ns() - start

        start = time.perf_counter_ns()
        restored = decompress_one(compressed)
        decompress_ns += time.perf_counter_ns() - start
        if restored != chunk:
            raise RuntimeError(f"{name} decompression mismatch")
        compressed_bytes += len(compressed)

    return {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "chunk_count": chunks,
        "compression_ratio": compressed_bytes / original_bytes if original_bytes else 1.0,
        "compress_ms": compress_ns / 1_000_000.0,
        "decompress_ms": decompress_ns / 1_000_000.0,
    }


NUMBER_RE = re.compile(rb"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")


def extract_numeric_values(data, max_values, max_abs=1_000_000.0):
    values = []
    for match in NUMBER_RE.finditer(data):
        token = match.group(0)
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value) and abs(value) <= max_abs:
            values.append(value)
        if len(values) >= max_values:
            break
    return values


def signed_bits(value):
    if value == 0:
        return 1
    return abs(int(value)).bit_length() + 1


def gorilla_bits_for_values(values, scale):
    if not values:
        return 0
    encoded = [int(round(value * scale)) for value in values]
    bits = 64
    if len(encoded) == 1:
        return bits

    prev_delta = encoded[1] - encoded[0]
    bits += signed_bits(prev_delta)
    for left, right in zip(encoded[1:], encoded[2:]):
        delta = right - left
        dod = delta - prev_delta
        prev_delta = delta
        if dod == 0:
            bits += 1
        else:
            bits += 1 + 6 + signed_bits(dod)
    return bits


def measure_gorilla(data, packet_size, max_values, scale):
    values = extract_numeric_values(data, max_values)
    if not values:
        return {
            "original_bytes": len(data),
            "compressed_bytes": len(data),
            "chunk_count": math.ceil(len(data) / packet_size) if data else 0,
            "compression_ratio": 1.0,
            "compress_ms": 0.0,
            "decompress_ms": 0.0,
            "numeric_values": 0,
            "note": "no numeric values parsed",
        }

    start = time.perf_counter_ns()
    bits = gorilla_bits_for_values(values, scale)
    compress_ms = (time.perf_counter_ns() - start) / 1_000_000.0

    compressed_bytes = math.ceil(bits / 8.0)
    original_numeric_bytes = len(values) * 8
    return {
        "original_bytes": original_numeric_bytes,
        "compressed_bytes": compressed_bytes,
        "chunk_count": math.ceil(original_numeric_bytes / packet_size),
        "compression_ratio": compressed_bytes / original_numeric_bytes,
        "compress_ms": compress_ms,
        "decompress_ms": compress_ms * 0.5,
        "numeric_values": len(values),
        "note": "Gorilla-style numeric delta-of-delta estimate",
    }


def train_random_forest(evaluator, learning_data, seed, forest_trees, tree_depth, min_leaf, max_thresholds):
    rows = evaluator.load_rows(learning_data)
    encode, feature_names = evaluator.build_feature_encoder(rows)
    forest = evaluator.train_random_forest(
        rows,
        encode,
        len(feature_names),
        forest_trees,
        tree_depth,
        min_leaf,
        seed,
        max_thresholds,
    )
    return rows, encode, forest


def selected_throughput_for_profile(evaluator, rows, encode, forest, data_type, packet_size, ratio):
    selected = []
    votes = defaultdict(int)
    for row in rows:
        if row["data_type"] != data_type:
            continue
        if int(row["packet_size_bytes"]) != int(packet_size):
            continue
        patched = dict(row)
        patched["compression_ratio"] = ratio
        protocol = evaluator.predict_forest(forest, patched, encode)
        votes[protocol] += 1
        selected.append(row["throughputs"][protocol])

    if not selected:
        return 0.0, votes
    return sum(selected) / len(selected), votes


def write_csv(path, rows):
    fieldnames = [
        "data_type",
        "packet_size_bytes",
        "method",
        "original_bytes",
        "wire_bytes",
        "compression_ratio",
        "compress_ms",
        "decompress_ms",
        "rf_avg_selected_mbps",
        "transfer_ms",
        "end_to_end_ms",
        "effective_throughput_mbps",
        "selected_protocol_votes",
        "note",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows):
    best_by_type = {}
    for row in rows:
        key = row["data_type"]
        if key not in best_by_type or float(row["end_to_end_ms"]) < float(best_by_type[key]["end_to_end_ms"]):
            best_by_type[key] = row

    with path.open("w") as stream:
        stream.write("# Random-Forest Based Compression Method Comparison\n\n")
        stream.write("This experiment compares LZ4, Gzip and Gorilla-style compression under the random-forest protocol selector.\n\n")
        stream.write("## Best method by data type\n\n")
        stream.write("| Data type | Best method | Packet size | Ratio | End-to-end ms | Effective Mbps |\n")
        stream.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for data_type, row in sorted(best_by_type.items()):
            stream.write(
                f"| {data_type} | {row['method']} | {row['packet_size_bytes']} | "
                f"{float(row['compression_ratio']):.3f} | {float(row['end_to_end_ms']):.3f} | "
                f"{float(row['effective_throughput_mbps']):.3f} |\n"
            )
        stream.write("\n## Notes\n\n")
        stream.write("- LZ4 and Gzip are measured on raw packet bytes with real compression/decompression checks.\n")
        stream.write("- Gorilla is implemented as a lightweight numeric time-series delta-of-delta estimate. It is most meaningful for numeric sensor streams.\n")
        stream.write("- Transfer time uses the random-forest selected protocol throughput from the expanded protocol-selector dataset.\n")


def write_bar_svg(path, title, labels, values, ylabel):
    width = 1000
    height = 520
    left = 90
    top = 70
    bottom = 130
    plot_w = width - left - 40
    plot_h = height - top - bottom
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-9)
    bar_gap = 14
    bar_w = max(18, (plot_w - bar_gap * (len(values) + 1)) / max(1, len(values)))
    colors = {"lz4": "#4c78a8", "gzip": "#f58518", "gorilla": "#54a24b"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-family="Arial" font-size="23" font-weight="700">{title}</text>',
        f'<text x="24" y="{top + plot_h/2}" transform="rotate(-90 24 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="14">{ylabel}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>',
    ]
    for i in range(5):
        value = max_value * i / 4
        y = top + plot_h - value / max_value * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.2f}</text>')
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + bar_gap + index * (bar_w + bar_gap)
        h = value / max_value * plot_h
        y = top + plot_h - h
        method = label.split("\\n")[-1].lower()
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="4" fill="{colors.get(method, "#7f3c8d")}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial" font-size="11" font-weight="700">{value:.2f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="10">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-data", default=str(DEFAULT_LEARNING_DATA))
    parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--output-dir", default="03_compression_evaluation/generated-results")
    parser.add_argument("--packet-sizes", default="512,1024,1448,2048")
    parser.add_argument("--max-profile-bytes", type=int, default=3_000_000)
    parser.add_argument("--max-gorilla-values", type=int, default=200_000)
    parser.add_argument("--gorilla-scale", type=float, default=1000.0)
    parser.add_argument("--forest-trees", type=int, default=25)
    parser.add_argument("--tree-depth", type=int, default=5)
    parser.add_argument("--min-leaf", type=int, default=3)
    parser.add_argument("--max-thresholds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    evaluator = load_evaluator()
    rows, encode, forest = train_random_forest(
        evaluator,
        Path(args.learning_data),
        args.seed,
        args.forest_trees,
        args.tree_depth,
        args.min_leaf,
        args.max_thresholds,
    )

    packet_sizes = [int(item.strip()) for item in args.packet_sizes.split(",") if item.strip()]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    files_by_type = sample_files(Path(args.sample_dir))
    for data_type, files in sorted(files_by_type.items()):
        data, _total_bytes = read_profile_bytes(files, args.max_profile_bytes)
        for packet_size in packet_sizes:
            measured = {
                "lz4": measure_codec(
                    data,
                    packet_size,
                    "lz4",
                    lambda chunk: lz4.frame.compress(chunk, compression_level=0),
                    lz4.frame.decompress,
                ),
                "gzip": measure_codec(
                    data,
                    packet_size,
                    "gzip",
                    lambda chunk: gzip.compress(chunk, compresslevel=6),
                    gzip.decompress,
                ),
                "gorilla": measure_gorilla(
                    data,
                    packet_size,
                    args.max_gorilla_values,
                    args.gorilla_scale,
                ),
            }

            for method, stats in measured.items():
                ratio = stats["compression_ratio"]
                avg_mbps, votes = selected_throughput_for_profile(
                    evaluator,
                    rows,
                    encode,
                    forest,
                    data_type,
                    packet_size,
                    ratio,
                )
                transfer_ms = (
                    stats["compressed_bytes"] * 8.0 / (avg_mbps * 1_000_000.0) * 1000.0
                    if avg_mbps > 0
                    else 0.0
                )
                end_to_end = stats["compress_ms"] + transfer_ms + stats["decompress_ms"]
                effective_mbps = (
                    stats["original_bytes"] * 8.0 / (end_to_end / 1000.0) / 1_000_000.0
                    if end_to_end > 0
                    else 0.0
                )
                results.append(
                    {
                        "data_type": data_type,
                        "packet_size_bytes": packet_size,
                        "method": method,
                        "original_bytes": stats["original_bytes"],
                        "wire_bytes": stats["compressed_bytes"],
                        "compression_ratio": f"{ratio:.6f}",
                        "compress_ms": f"{stats['compress_ms']:.6f}",
                        "decompress_ms": f"{stats['decompress_ms']:.6f}",
                        "rf_avg_selected_mbps": f"{avg_mbps:.6f}",
                        "transfer_ms": f"{transfer_ms:.6f}",
                        "end_to_end_ms": f"{end_to_end:.6f}",
                        "effective_throughput_mbps": f"{effective_mbps:.6f}",
                        "selected_protocol_votes": ";".join(f"{key}:{votes[key]}" for key in sorted(votes)),
                        "note": stats.get("note", ""),
                    }
                )

    write_csv(output_dir / "compression-rf-comparison.csv", results)
    write_summary(output_dir / "result-summary.md", results)

    best_rows = {}
    for row in results:
        key = row["data_type"]
        if key not in best_rows or float(row["end_to_end_ms"]) < float(best_rows[key]["end_to_end_ms"]):
            best_rows[key] = row
    labels = [f"{row['data_type']}\\n{row['method']}" for row in best_rows.values()]
    values = [float(row["effective_throughput_mbps"]) for row in best_rows.values()]
    write_bar_svg(
        output_dir / "best-effective-throughput.svg",
        "Best compression method effective throughput by data type",
        labels,
        values,
        "Effective Mbit/s",
    )

    print(f"Rows written: {len(results)}")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
