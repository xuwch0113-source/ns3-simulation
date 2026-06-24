#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Evaluate lightweight protocol-selector models on expanded learning data.

import argparse
import csv
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABELS = ["TCP_CUBIC", "TCP_BBR", "QUIC_BBR"]
NUMERIC_FEATURES = [
    "rtt_ms",
    "capacity_mbps",
    "packet_loss_rate",
    "protocol_packet_size_bytes",
    "packet_size_min",
    "packet_size_max",
    "packet_size_mean",
    "packet_size_std",
    "min_send_interval_ms",
    "max_send_interval_ms",
    "mean_send_interval_ms",
    "interval_min_ms",
    "interval_max_ms",
    "interval_mean_ms",
    "interval_std_ms",
    "phase_progress",
    "loss_trend",
    "capacity_trend",
    "rtt_trend",
    "offered_load_mbps",
    "file_size_mb",
    "effective_file_size_mb",
    "packet_size_bytes",
    "compression_ratio",
    "prev_rtt_ms",
    "prev_capacity_mbps",
    "prev_packet_loss_rate",
]
THROUGHPUT_COLUMNS = {
    "TCP_CUBIC": "tcp_cubic_mbps",
    "TCP_BBR": "tcp_bbr_mbps",
    "QUIC_BBR": "quic_bbr_mbps",
}
FEATURE_FALLBACKS = {
    "protocol_packet_size_bytes": ["packet_size_mean", "packet_size_bytes"],
    "packet_size_bytes": ["packet_size_mean", "protocol_packet_size_bytes"],
    "min_send_interval_ms": ["interval_min_ms"],
    "max_send_interval_ms": ["interval_max_ms"],
    "mean_send_interval_ms": ["interval_mean_ms"],
    "interval_min_ms": ["min_send_interval_ms"],
    "interval_max_ms": ["max_send_interval_ms"],
    "interval_mean_ms": ["mean_send_interval_ms"],
}


def parse_float(row, key, default=0.0):
    value = row.get(key, "")
    if value not in ("", None):
        try:
            return float(value)
        except ValueError:
            return default
    for fallback in FEATURE_FALLBACKS.get(key, []):
        fallback_value = row.get(fallback, "")
        if fallback_value not in ("", None):
            try:
                return float(fallback_value)
            except ValueError:
                continue
    return default


def load_rows(path):
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            label = row.get("best_protocol") or row.get("label")
            if label not in LABELS:
                continue
            parsed = {"label": label, "raw": row}
            for feature in NUMERIC_FEATURES:
                parsed[feature] = parse_float(row, feature)
            parsed["data_type"] = row.get("data_type", "unknown")
            parsed["traffic_mode"] = row.get("traffic_mode", "bulk")
            parsed["state_name"] = row.get("state_name", "unknown")
            parsed["interval_mode"] = row.get("interval_mode", "unknown")
            parsed["prev_best_protocol"] = row.get("prev_best_protocol", "unknown")
            parsed["group_key"] = (
                row.get("source_scenario", row.get("scenario", "")),
                row.get("state_name", ""),
                row.get("rtt_ms", ""),
                row.get("capacity_mbps", ""),
                row.get("packet_loss_rate", ""),
                row.get("packet_size_mean", row.get("protocol_packet_size_bytes", "")),
                row.get("interval_min_ms", row.get("min_send_interval_ms", "")),
                row.get("interval_max_ms", row.get("max_send_interval_ms", "")),
            )
            parsed["throughputs"] = {
                label_name: parse_float(row, column)
                for label_name, column in THROUGHPUT_COLUMNS.items()
            }
            rows.append(parsed)
    return rows


def build_feature_encoder(rows):
    data_types = sorted({row["data_type"] for row in rows})
    traffic_modes = sorted({row["traffic_mode"] for row in rows})
    state_names = sorted({row["state_name"] for row in rows})
    interval_modes = sorted({row["interval_mode"] for row in rows})
    previous_protocols = sorted({row["prev_best_protocol"] for row in rows})
    mins = {feature: min(row[feature] for row in rows) for feature in NUMERIC_FEATURES}
    maxs = {feature: max(row[feature] for row in rows) for feature in NUMERIC_FEATURES}
    names = (
        NUMERIC_FEATURES
        + [f"data_type={name}" for name in data_types]
        + [f"traffic_mode={name}" for name in traffic_modes]
        + [f"state_name={name}" for name in state_names]
        + [f"interval_mode={name}" for name in interval_modes]
        + [f"prev_best_protocol={name}" for name in previous_protocols]
    )

    def encode(row):
        values = []
        for feature in NUMERIC_FEATURES:
            low = mins[feature]
            high = maxs[feature]
            values.append((row[feature] - low) / (high - low) if high > low else 0.0)
        values.extend(1.0 if row["data_type"] == name else 0.0 for name in data_types)
        values.extend(1.0 if row["traffic_mode"] == name else 0.0 for name in traffic_modes)
        values.extend(1.0 if row["state_name"] == name else 0.0 for name in state_names)
        values.extend(1.0 if row["interval_mode"] == name else 0.0 for name in interval_modes)
        values.extend(1.0 if row["prev_best_protocol"] == name else 0.0 for name in previous_protocols)
        return values

    return encode, names


def downsample_rows(rows, max_rows, seed):
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    selected = []
    for label in LABELS:
        label_rows = by_label.get(label, [])[:]
        rng.shuffle(label_rows)
        quota = max(1, round(max_rows * len(label_rows) / len(rows)))
        selected.extend(label_rows[: min(quota, len(label_rows))])
    while len(selected) > max_rows:
        selected.pop(rng.randrange(len(selected)))
    rng.shuffle(selected)
    return selected


def stratified_row_split(rows, test_ratio, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train = []
    test = []
    for label_rows in by_label.values():
        label_rows = label_rows[:]
        rng.shuffle(label_rows)
        test_count = max(1, int(round(len(label_rows) * test_ratio))) if len(label_rows) > 1 else 0
        test.extend(label_rows[:test_count])
        train.extend(label_rows[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def grouped_split(rows, test_ratio, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_key"]].append(row)

    group_items = list(groups.items())
    by_label = defaultdict(list)
    for group_key, group_rows in group_items:
        by_label[group_rows[0]["label"]].append((group_key, group_rows))

    train = []
    test = []
    for label_groups in by_label.values():
        label_groups = label_groups[:]
        rng.shuffle(label_groups)
        test_count = max(1, int(round(len(label_groups) * test_ratio))) if len(label_groups) > 1 else 0
        for _group_key, group_rows in label_groups[:test_count]:
            test.extend(group_rows)
        for _group_key, group_rows in label_groups[test_count:]:
            train.extend(group_rows)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def majority_label(rows):
    counts = Counter(row["label"] for row in rows)
    return sorted(counts.items(), key=lambda item: (-item[1], LABELS.index(item[0])))[0][0]


def gini(rows):
    if not rows:
        return 0.0
    counts = Counter(row["label"] for row in rows)
    return 1.0 - sum((count / len(rows)) ** 2 for count in counts.values())


def build_tree(
    rows,
    encode,
    feature_count,
    max_depth,
    min_leaf,
    rng=None,
    max_features=None,
    max_thresholds=64,
    depth=0,
):
    node = {
        "prediction": majority_label(rows),
        "samples": len(rows),
        "class_counts": dict(Counter(row["label"] for row in rows)),
    }
    if depth >= max_depth or len({row["label"] for row in rows}) == 1:
        node["type"] = "leaf"
        return node

    parent_gini = gini(rows)
    feature_indexes = list(range(feature_count))
    if rng is not None and max_features is not None and max_features < feature_count:
        feature_indexes = sorted(rng.sample(feature_indexes, max_features))

    best = None
    encoded_cache = {id(row): encode(row) for row in rows}
    for feature_index in feature_indexes:
        values = sorted({encoded_cache[id(row)][feature_index] for row in rows})
        if len(values) > max_thresholds:
            values = [
                values[round((len(values) - 1) * index / (max_thresholds - 1))]
                for index in range(max_thresholds)
            ]
            values = sorted(set(values))
        thresholds = [(left + right) / 2.0 for left, right in zip(values, values[1:])]
        for threshold in thresholds:
            left = [row for row in rows if encoded_cache[id(row)][feature_index] <= threshold]
            right = [row for row in rows if encoded_cache[id(row)][feature_index] > threshold]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            weighted = (len(left) * gini(left) + len(right) * gini(right)) / len(rows)
            gain = parent_gini - weighted
            if best is None or gain > best["gain"]:
                best = {
                    "feature_index": feature_index,
                    "threshold": threshold,
                    "gain": gain,
                    "left": left,
                    "right": right,
                }

    if best is None or best["gain"] <= 0:
        node["type"] = "leaf"
        return node

    node.update(
        {
            "type": "split",
            "feature_index": best["feature_index"],
            "threshold": best["threshold"],
            "gain": best["gain"],
            "left": build_tree(
                best["left"],
                encode,
                feature_count,
                max_depth,
                min_leaf,
                rng,
                max_features,
                max_thresholds,
                depth + 1,
            ),
            "right": build_tree(
                best["right"],
                encode,
                feature_count,
                max_depth,
                min_leaf,
                rng,
                max_features,
                max_thresholds,
                depth + 1,
            ),
        }
    )
    return node


def predict_tree(tree, row, encode):
    node = tree
    values = encode(row)
    while node["type"] == "split":
        if values[node["feature_index"]] <= node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
    return node["prediction"]


def train_random_forest(rows, encode, feature_count, trees, max_depth, min_leaf, seed, max_thresholds):
    rng = random.Random(seed)
    max_features = max(1, int(math.sqrt(feature_count)))
    forest = []
    for _ in range(trees):
        bootstrap = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        forest.append(
            build_tree(
                bootstrap,
                encode,
                feature_count,
                max_depth=max_depth,
                min_leaf=min_leaf,
                rng=rng,
                max_features=max_features,
                max_thresholds=max_thresholds,
            )
        )
    return forest


def predict_forest(forest, row, encode):
    votes = Counter(predict_tree(tree, row, encode) for tree in forest)
    return sorted(votes.items(), key=lambda item: (-item[1], LABELS.index(item[0])))[0][0]


def train_knn(rows, encode, k):
    return {"rows": rows, "vectors": [encode(row) for row in rows], "k": k}


def predict_knn(model, row, encode):
    vector = encode(row)
    distances = []
    for train_row, train_vector in zip(model["rows"], model["vectors"]):
        distance = sum((left - right) ** 2 for left, right in zip(vector, train_vector))
        distances.append((distance, train_row["label"]))
    counts = Counter(label for _distance, label in sorted(distances)[: model["k"]])
    return sorted(counts.items(), key=lambda item: (-item[1], LABELS.index(item[0])))[0][0]


def train_gaussian_nb(rows, encode, feature_count):
    model = {"priors": {}, "means": {}, "vars": {}}
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(encode(row))
    for label in LABELS:
        values = by_label.get(label, [])
        if not values:
            continue
        model["priors"][label] = len(values) / len(rows)
        model["means"][label] = [
            sum(vector[i] for vector in values) / len(values) for i in range(feature_count)
        ]
        model["vars"][label] = []
        for i in range(feature_count):
            mean = model["means"][label][i]
            variance = sum((vector[i] - mean) ** 2 for vector in values) / len(values)
            model["vars"][label].append(max(variance, 1e-6))
    return model


def predict_gaussian_nb(model, row, encode):
    vector = encode(row)
    best_label = None
    best_score = None
    for label in model["priors"]:
        score = math.log(model["priors"][label])
        for value, mean, variance in zip(vector, model["means"][label], model["vars"][label]):
            score += -0.5 * math.log(2.0 * math.pi * variance)
            score += -((value - mean) ** 2) / (2.0 * variance)
        if best_score is None or score > best_score:
            best_label = label
            best_score = score
    return best_label or LABELS[0]


def softmax(values):
    high = max(values)
    exps = [math.exp(value - high) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def train_logistic(rows, encode, feature_count, epochs, learning_rate, seed):
    rng = random.Random(seed)
    weights = [[rng.uniform(-0.05, 0.05) for _ in range(feature_count)] for _ in LABELS]
    biases = [0.0 for _ in LABELS]
    vectors = [encode(row) for row in rows]
    targets = [LABELS.index(row["label"]) for row in rows]
    for _ in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for index in order:
            vector = vectors[index]
            target = targets[index]
            logits = [
                biases[o] + sum(weights[o][i] * vector[i] for i in range(feature_count))
                for o in range(len(LABELS))
            ]
            probs = softmax(logits)
            probs[target] -= 1.0
            for o in range(len(LABELS)):
                for i in range(feature_count):
                    weights[o][i] -= learning_rate * probs[o] * vector[i]
                biases[o] -= learning_rate * probs[o]
    return {"weights": weights, "biases": biases}


def predict_logistic(model, row, encode):
    vector = encode(row)
    logits = [
        model["biases"][o] + sum(model["weights"][o][i] * vector[i] for i in range(len(vector)))
        for o in range(len(LABELS))
    ]
    return LABELS[max(range(len(LABELS)), key=lambda index: logits[index])]


def train_mlp(rows, encode, feature_count, hidden_size, epochs, learning_rate, seed):
    rng = random.Random(seed)
    w1 = [[rng.uniform(-0.4, 0.4) for _ in range(feature_count)] for _ in range(hidden_size)]
    b1 = [0.0 for _ in range(hidden_size)]
    w2 = [[rng.uniform(-0.4, 0.4) for _ in range(hidden_size)] for _ in LABELS]
    b2 = [0.0 for _ in LABELS]
    vectors = [encode(row) for row in rows]
    targets = [LABELS.index(row["label"]) for row in rows]

    for _ in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for index in order:
            vector = vectors[index]
            target = targets[index]
            hidden_raw = [
                b1[h] + sum(w1[h][i] * vector[i] for i in range(feature_count))
                for h in range(hidden_size)
            ]
            hidden = [math.tanh(value) for value in hidden_raw]
            logits = [
                b2[o] + sum(w2[o][h] * hidden[h] for h in range(hidden_size))
                for o in range(len(LABELS))
            ]
            probs = softmax(logits)
            probs[target] -= 1.0
            old_w2 = [row[:] for row in w2]
            for o in range(len(LABELS)):
                for h in range(hidden_size):
                    w2[o][h] -= learning_rate * probs[o] * hidden[h]
                b2[o] -= learning_rate * probs[o]
            for h in range(hidden_size):
                grad = sum(probs[o] * old_w2[o][h] for o in range(len(LABELS)))
                grad *= 1.0 - hidden[h] ** 2
                for i in range(feature_count):
                    w1[h][i] -= learning_rate * grad * vector[i]
                b1[h] -= learning_rate * grad
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "hidden_size": hidden_size}


def predict_mlp(model, row, encode):
    vector = encode(row)
    hidden = [
        math.tanh(model["b1"][h] + sum(model["w1"][h][i] * vector[i] for i in range(len(vector))))
        for h in range(model["hidden_size"])
    ]
    logits = [
        model["b2"][o] + sum(model["w2"][o][h] * hidden[h] for h in range(model["hidden_size"]))
        for o in range(len(LABELS))
    ]
    return LABELS[max(range(len(LABELS)), key=lambda index: logits[index])]


def evaluate(rows, predictor):
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    selected = 0.0
    oracle = 0.0
    correct = 0
    for row in rows:
        actual = row["label"]
        predicted = predictor(row)
        matrix[actual][predicted] += 1
        correct += int(actual == predicted)
        selected += row["throughputs"][predicted]
        oracle += row["throughputs"][actual]

    per_label = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in LABELS if actual != label)
        fn = sum(matrix[label][predicted] for predicted in LABELS if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = sum(values["f1"] for values in per_label.values()) / len(LABELS)
    avg_selected = selected / len(rows) if rows else 0.0
    avg_oracle = oracle / len(rows) if rows else 0.0
    return {
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1": macro_f1,
        "avg_selected_mbps": avg_selected,
        "avg_oracle_mbps": avg_oracle,
        "wrong_selection_loss_mbps": avg_oracle - avg_selected,
        "per_label": per_label,
        "matrix": matrix,
    }


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_bar_svg(path, title, labels, values, ylabel):
    width = 980
    height = 520
    margin_left = 80
    margin_bottom = 110
    plot_width = width - margin_left - 40
    plot_height = height - 90 - margin_bottom
    top = 70
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-9)
    bar_gap = 18
    bar_width = max(18, (plot_width - bar_gap * (len(values) + 1)) / max(1, len(values)))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#7f3c8d", "#9c6ade", "#1696a7", "#e45756"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{title}</text>',
        f'<text x="24" y="{top + plot_height / 2}" transform="rotate(-90 24 {top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="15">{ylabel}</text>',
        f'<line x1="{margin_left}" y1="{top + plot_height}" x2="{margin_left + plot_width}" y2="{top + plot_height}" stroke="#1f2937" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{top}" x2="{margin_left}" y2="{top + plot_height}" stroke="#1f2937" stroke-width="2"/>',
    ]
    for i in range(5):
        value = max_value * i / 4
        y = top + plot_height - (value / max_value) * plot_height
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.2f}</text>')
    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + bar_gap + index * (bar_width + bar_gap)
        bar_height = (value / max_value) * plot_height
        y = top + plot_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="5" fill="{colors[index % len(colors)]}"/>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">{value:.3f}</text>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{top + plot_height + 26}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_outputs(output_dir, train_rows, test_rows, feature_names, metrics, matrices):
    metric_rows = []
    for model_name, values in metrics.items():
        metric_rows.append(
            {
                "model": model_name,
                "accuracy": f"{values['accuracy']:.6f}",
                "macro_f1": f"{values['macro_f1']:.6f}",
                "avg_selected_mbps": f"{values['avg_selected_mbps']:.6f}",
                "avg_oracle_mbps": f"{values['avg_oracle_mbps']:.6f}",
                "wrong_selection_loss_mbps": f"{values['wrong_selection_loss_mbps']:.6f}",
            }
        )
    write_csv(
        output_dir / "model-metrics.csv",
        ["model", "accuracy", "macro_f1", "avg_selected_mbps", "avg_oracle_mbps", "wrong_selection_loss_mbps"],
        metric_rows,
    )

    for model_name, matrix in matrices.items():
        rows = []
        for actual in LABELS:
            row = {"actual": actual}
            row.update(matrix[actual])
            rows.append(row)
        write_csv(output_dir / f"confusion-matrix-{model_name}.csv", ["actual"] + LABELS, rows)

    write_bar_svg(
        output_dir / "macro-f1-comparison.svg",
        "Lightweight selector macro F1 comparison",
        [row["model"] for row in metric_rows],
        [float(row["macro_f1"]) for row in metric_rows],
        "Macro F1",
    )
    write_bar_svg(
        output_dir / "model-throughput-comparison.svg",
        "Average selected throughput by model",
        [row["model"] for row in metric_rows],
        [float(row["avg_selected_mbps"]) for row in metric_rows],
        "Mbit/s",
    )

    with (output_dir / "result-summary.md").open("w") as stream:
        stream.write("# Lightweight Selector Evaluation Summary\n\n")
        stream.write(f"Training samples: {len(train_rows)}\n\n")
        stream.write(f"Test samples: {len(test_rows)}\n\n")
        stream.write("Features:\n\n")
        for feature in feature_names:
            stream.write(f"- {feature}\n")
        stream.write("\n## Metrics\n\n")
        stream.write("| Model | Accuracy | Macro F1 | Avg throughput | Wrong-selection loss |\n")
        stream.write("| --- | ---: | ---: | ---: | ---: |\n")
        for row in metric_rows:
            stream.write(
                f"| {row['model']} | {float(row['accuracy']):.3f} | "
                f"{float(row['macro_f1']):.3f} | {float(row['avg_selected_mbps']):.3f} | "
                f"{float(row['wrong_selection_loss_mbps']):.3f} |\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="05_datasets_and_generators/training-data/adaptive-protocol-large-training-dataset.csv.gz")
    parser.add_argument("--output-dir", default="03_compression_evaluation/generated-model-evaluation")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--split-mode", choices=["group", "row"], default="group")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tree-depth", type=int, default=5)
    parser.add_argument("--min-leaf", type=int, default=3)
    parser.add_argument("--max-thresholds", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--forest-trees", type=int, default=25)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--hidden-size", type=int, default=10)
    args = parser.parse_args()

    rows = load_rows(ROOT / args.input)
    rows = downsample_rows(rows, args.max_rows, args.seed)
    if len(rows) < 6:
        raise RuntimeError("Need at least 6 learning samples for model evaluation.")

    encode, feature_names = build_feature_encoder(rows)
    if args.split_mode == "group":
        train_rows, test_rows = grouped_split(rows, args.test_ratio, args.seed)
    else:
        train_rows, test_rows = stratified_row_split(rows, args.test_ratio, args.seed)
    feature_count = len(feature_names)

    majority = majority_label(train_rows)
    tree = build_tree(
        train_rows,
        encode,
        feature_count,
        args.tree_depth,
        args.min_leaf,
        max_thresholds=args.max_thresholds,
    )
    forest = train_random_forest(
        train_rows,
        encode,
        feature_count,
        args.forest_trees,
        args.tree_depth,
        args.min_leaf,
        args.seed,
        args.max_thresholds,
    )
    knn = train_knn(train_rows, encode, args.knn_k)
    nb = train_gaussian_nb(train_rows, encode, feature_count)
    logistic = train_logistic(train_rows, encode, feature_count, args.epochs, args.learning_rate, args.seed)
    mlp = train_mlp(
        train_rows,
        encode,
        feature_count,
        args.hidden_size,
        args.epochs,
        args.learning_rate,
        args.seed,
    )

    predictors = {
        "majority": lambda row: majority,
        "decision_tree": lambda row: predict_tree(tree, row, encode),
        "logistic_regression": lambda row: predict_logistic(logistic, row, encode),
        "knn": lambda row: predict_knn(knn, row, encode),
        "gaussian_nb": lambda row: predict_gaussian_nb(nb, row, encode),
        "random_forest": lambda row: predict_forest(forest, row, encode),
        "mlp": lambda row: predict_mlp(mlp, row, encode),
    }

    metrics = {}
    matrices = {}
    for model_name, predictor in predictors.items():
        result = evaluate(test_rows, predictor)
        metrics[model_name] = result
        matrices[model_name] = result["matrix"]

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(output_dir, train_rows, test_rows, feature_names, metrics, matrices)

    with (output_dir / "decision-tree-model.json").open("w") as stream:
        json.dump({"features": feature_names, "labels": LABELS, "tree": tree}, stream, indent=2)

    best = max(metrics, key=lambda name: (metrics[name]["macro_f1"], metrics[name]["avg_selected_mbps"]))
    print(f"Training samples: {len(train_rows)}")
    print(f"Test samples: {len(test_rows)}")
    print(f"Best model: {best}")
    print(f"Best macro F1: {metrics[best]['macro_f1']:.3f}")
    print(f"Best avg throughput: {metrics[best]['avg_selected_mbps']:.3f} Mbit/s")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
