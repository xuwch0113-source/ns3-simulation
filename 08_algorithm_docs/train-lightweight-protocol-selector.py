#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Train lightweight adaptive protocol selectors on the large tabular dataset.

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / "python-packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ns3-final-submission-matplotlib")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


LABELS = ["TCP_CUBIC", "TCP_BBR", "QUIC_BBR"]
AUC_LABELS = sorted(LABELS)
NUMERIC_FEATURES = [
    "rtt_ms",
    "capacity_mbps",
    "packet_loss_rate",
    "phase_progress",
    "loss_trend",
    "capacity_trend",
    "rtt_trend",
    "packet_size_min",
    "packet_size_max",
    "packet_size_mean",
    "packet_size_std",
    "interval_min_ms",
    "interval_max_ms",
    "interval_mean_ms",
    "interval_std_ms",
    "offered_load_mbps",
    "file_size_mb",
    "effective_file_size_mb",
    "compression_ratio",
    "prev_rtt_ms",
    "prev_capacity_mbps",
    "prev_packet_loss_rate",
]
CATEGORICAL_FEATURES = [
    "source_scenario",
    "state_name",
    "interval_mode",
    "traffic_mode",
    "data_type",
    "prev_best_protocol",
]
THROUGHPUT_COLUMNS = {
    "TCP_CUBIC": "tcp_cubic_mbps",
    "TCP_BBR": "tcp_bbr_mbps",
    "QUIC_BBR": "quic_bbr_mbps",
}
BANDIT_REWARD_COLUMNS = {
    "TCP_CUBIC": "bandit_action_tcp_cubic_reward",
    "TCP_BBR": "bandit_action_tcp_bbr_reward",
    "QUIC_BBR": "bandit_action_quic_bbr_reward",
}


def load_dataset(path, max_rows, seed):
    df = pd.read_csv(path)
    df = df[df["best_protocol"].isin(LABELS)].copy()
    if max_rows and len(df) > max_rows:
        parts = []
        for label in LABELS:
            group = df[df["best_protocol"] == label]
            quota = max(1, round(max_rows * len(group) / len(df)))
            parts.append(group.sample(n=min(quota, len(group)), random_state=seed))
        df = pd.concat(parts, ignore_index=True)
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    for column in NUMERIC_FEATURES:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    for column in CATEGORICAL_FEATURES:
        if column not in df.columns:
            df[column] = "unknown"
        df[column] = df[column].fillna("unknown").astype(str)
    return df


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def selected_throughput(df, predictions):
    values = []
    for index, protocol in enumerate(predictions):
        values.append(float(df.iloc[index][THROUGHPUT_COLUMNS[protocol]]))
    return float(np.mean(values)) if values else 0.0


def oracle_throughput(df):
    values = []
    for _index, row in df.iterrows():
        protocol = row["best_protocol"]
        values.append(float(row[THROUGHPUT_COLUMNS[protocol]]))
    return float(np.mean(values)) if values else 0.0


def static_protocol_throughput(df, protocol):
    return float(pd.to_numeric(df[THROUGHPUT_COLUMNS[protocol]], errors="coerce").fillna(0.0).mean())


def normalize_score_matrix(scores):
    if scores is None:
        return None
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(LABELS):
        return None
    if np.all(matrix >= 0):
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def auc_metrics(y_true, scores):
    probabilities = normalize_score_matrix(scores)
    if probabilities is None:
        return float("nan"), float("nan")
    auc_indexes = [LABELS.index(label) for label in AUC_LABELS]
    auc_probabilities = probabilities[:, auc_indexes]
    try:
        macro_auc = roc_auc_score(
            y_true,
            auc_probabilities,
            labels=AUC_LABELS,
            multi_class="ovr",
            average="macro",
        )
        weighted_auc = roc_auc_score(
            y_true,
            auc_probabilities,
            labels=AUC_LABELS,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        return float("nan"), float("nan")
    return macro_auc, weighted_auc


def scores_from_model(model, feature_frame, label_encoder):
    if hasattr(model, "predict_proba"):
        raw_scores = model.predict_proba(feature_frame)
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(feature_frame)
    else:
        return None
    raw_scores = np.asarray(raw_scores)
    if raw_scores.ndim != 2:
        return None
    ordered = np.zeros((raw_scores.shape[0], len(LABELS)))
    for target_index, label in enumerate(LABELS):
        encoded_index = int(label_encoder.transform([label])[0])
        ordered[:, target_index] = raw_scores[:, encoded_index]
    return ordered


def scores_from_static(protocol, row_count):
    scores = np.zeros((row_count, len(LABELS)))
    scores[:, LABELS.index(protocol)] = 1.0
    return scores


def evaluate_predictions(name, y_true, predictions, test_df, scores=None):
    avg_selected = selected_throughput(test_df, predictions)
    avg_oracle = oracle_throughput(test_df)
    macro_auc, weighted_auc = auc_metrics(y_true, scores)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, predictions, labels=LABELS, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0),
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "avg_selected_mbps": avg_selected,
        "avg_oracle_mbps": avg_oracle,
        "oracle_gap_mbps": avg_oracle - avg_selected,
    }


def build_models(seed, fast):
    tree_count = 80 if fast else 220
    xgb_estimators = 80 if fast else 220
    lgb_estimators = 80 if fast else 220
    cat_iterations = 80 if fast else 220
    mlp_iter = 120 if fast else 280
    return {
        "LightGBM": LGBMClassifier(
            n_estimators=lgb_estimators,
            learning_rate=0.06,
            num_leaves=31,
            objective="multiclass",
            random_state=seed,
            verbosity=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=xgb_estimators,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=tree_count,
            max_depth=16,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "TinyMLP": MLPClassifier(
            hidden_layer_sizes=(24, 12),
            activation="relu",
            alpha=0.0008,
            learning_rate_init=0.002,
            max_iter=mlp_iter,
            early_stopping=True,
            random_state=seed,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=cat_iterations,
            depth=6,
            learning_rate=0.06,
            loss_function="MultiClass",
            verbose=False,
            random_seed=seed,
        ),
        "LinearSVM": LinearSVC(
            C=0.8,
            class_weight="balanced",
            random_state=seed,
            max_iter=6000,
        ),
    }


def train_contextual_bandit(train_df, test_df):
    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    x_test = preprocessor.transform(test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    reward_predictions = []
    for protocol in LABELS:
        reward = pd.to_numeric(train_df[BANDIT_REWARD_COLUMNS[protocol]], errors="coerce").fillna(0.0)
        regressor = Ridge(alpha=1.0)
        regressor.fit(x_train, reward)
        reward_predictions.append(regressor.predict(x_test))
    stacked = np.vstack(reward_predictions).T
    predictions = [LABELS[index] for index in np.argmax(stacked, axis=1)]
    return predictions, stacked, {"preprocessor": preprocessor}


def write_confusion_matrix(path, matrix):
    matrix_df = pd.DataFrame(matrix, index=LABELS, columns=LABELS)
    matrix_df.index.name = "actual"
    matrix_df.to_csv(path)


def draw_metric_bars(path, metrics_df, metric, title, ylabel):
    plt.figure(figsize=(10.5, 5.8))
    bars = plt.bar(metrics_df["model"], metrics_df[metric], color="#4c78a8")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=28, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_report(path, metrics_df, train_size, test_size, max_rows):
    best_f1 = metrics_df.sort_values(["macro_f1", "avg_selected_mbps"], ascending=False).iloc[0]
    best_throughput = metrics_df.sort_values(["avg_selected_mbps", "macro_f1"], ascending=False).iloc[0]
    with path.open("w") as stream:
        stream.write("# Lightweight Protocol Selector Training Results\n\n")
        stream.write(f"Training samples: {train_size}\n\n")
        stream.write(f"Test samples: {test_size}\n\n")
        stream.write(f"Max rows setting: {max_rows if max_rows else 'all'}\n\n")
        stream.write("## Best models\n\n")
        stream.write(f"- Best Macro F1: {best_f1['model']} ({best_f1['macro_f1']:.3f})\n")
        stream.write(f"- Best average selected throughput: {best_throughput['model']} ({best_throughput['avg_selected_mbps']:.3f} Mbit/s)\n\n")
        stream.write("## Metrics\n\n")
        stream.write("| Model | Accuracy | Macro F1 | Macro AUC | Weighted AUC | Macro Precision | Macro Recall | Avg selected Mbit/s | Oracle gap |\n")
        stream.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for _index, row in metrics_df.iterrows():
            stream.write(
                f"| {row['model']} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} | "
                f"{row['macro_auc']:.3f} | {row['weighted_auc']:.3f} | "
                f"{row['macro_precision']:.3f} | {row['macro_recall']:.3f} | "
                f"{row['avg_selected_mbps']:.3f} | {row['oracle_gap_mbps']:.3f} |\n"
            )
        stream.write("\nNote: AUC is computed with multiclass one-vs-rest. Oracle is the theoretical upper bound that selects the best protocol for every test sample. A smaller Oracle gap means the selector is closer to the optimal choice.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="05_datasets_and_generators/training-data/adaptive-protocol-large-training-dataset.csv.gz")
    parser.add_argument("--output-dir", default="08_algorithm_docs/model-results")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=30000)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(ROOT / args.input, args.max_rows, args.seed)
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=df["best_protocol"],
    )

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    y_train = train_df["best_protocol"]
    y_test = test_df["best_protocol"]
    label_encoder = LabelEncoder()
    label_encoder.fit(LABELS)
    y_train_encoded = label_encoder.transform(y_train)

    metrics = []
    trained_models = {}
    for name, estimator in build_models(args.seed, args.fast).items():
        print(f"Training {name}...")
        model = Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("model", estimator),
            ]
        )
        model.fit(train_df[feature_columns], y_train_encoded)
        predictions = label_encoder.inverse_transform(model.predict(test_df[feature_columns]))
        model_scores = scores_from_model(model, test_df[feature_columns], label_encoder)
        metrics.append(evaluate_predictions(name, y_test, predictions, test_df, model_scores))
        write_confusion_matrix(
            output_dir / f"confusion-matrix-{name}.csv",
            confusion_matrix(y_test, predictions, labels=LABELS),
        )
        (output_dir / f"classification-report-{name}.txt").write_text(
            classification_report(y_test, predictions, labels=LABELS, zero_division=0)
        )
        trained_models[name] = model

    print("Training ContextualBandit...")
    bandit_predictions, bandit_scores, bandit_model = train_contextual_bandit(train_df, test_df)
    metrics.append(
        evaluate_predictions(
            "ContextualBandit-Ridge",
            y_test,
            bandit_predictions,
            test_df,
            bandit_scores,
        )
    )
    write_confusion_matrix(
        output_dir / "confusion-matrix-ContextualBandit-Ridge.csv",
        confusion_matrix(y_test, bandit_predictions, labels=LABELS),
    )
    (output_dir / "classification-report-ContextualBandit-Ridge.txt").write_text(
        classification_report(y_test, bandit_predictions, labels=LABELS, zero_division=0)
    )
    trained_models["ContextualBandit-Ridge"] = bandit_model

    for protocol in LABELS:
        static_predictions = [protocol] * len(test_df)
        metrics.append(
            evaluate_predictions(
                f"Static-{protocol}",
                y_test,
                static_predictions,
                test_df,
                scores_from_static(protocol, len(test_df)),
            )
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.sort_values(["macro_f1", "avg_selected_mbps"], ascending=False)
    metrics_df.to_csv(output_dir / "model-metrics.csv", index=False)
    draw_metric_bars(output_dir / "macro-f1-comparison.png", metrics_df, "macro_f1", "Macro F1 comparison", "Macro F1")
    draw_metric_bars(output_dir / "auc-comparison.png", metrics_df, "macro_auc", "Macro AUC comparison", "Macro AUC")
    draw_metric_bars(
        output_dir / "average-selected-throughput-comparison.png",
        metrics_df,
        "avg_selected_mbps",
        "Average selected throughput comparison",
        "Mbit/s",
    )
    draw_metric_bars(
        output_dir / "oracle-gap-comparison.png",
        metrics_df.sort_values("oracle_gap_mbps", ascending=True),
        "oracle_gap_mbps",
        "Gap to oracle comparison",
        "Mbit/s",
    )
    write_report(output_dir / "training-summary.md", metrics_df, len(train_df), len(test_df), args.max_rows)
    joblib.dump(trained_models, output_dir / "trained-models.joblib")

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
