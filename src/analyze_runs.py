#!/usr/bin/env python3
"""Aggregate retained experiment artifacts without inventing missing measurements.

This script is intentionally conservative: it only consumes JSON artifacts whose
measurement_type is ``actual_run`` and whose status is ``completed``. Speedup and
scaling efficiency are calculated only when a matching one-worker baseline exists
for the same training volume. Missing or incomplete conditions remain missing.

A completed JSON file is not accepted as evidence merely because it contains
plausible-looking metric fields. Its retained model artifact, SHA-256 integrity
hash, run identity, partition accounting, row counts, and metric ranges are
validated first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def sample_std(values):
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def metric_summary(items, key):
    values = []
    for _, obj in items:
        value = obj.get(key)
        if value is not None:
            values.append(float(value))
    return mean(values), sample_std(values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_path(json_path: Path, recorded_path: str) -> Path:
    candidate = Path(recorded_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate
    return json_path.parent / candidate


def validate_completed_artifact(json_path: Path, obj: dict) -> list[str]:
    errors = []
    required = [
        "artifact_schema_version",
        "workers",
        "train_rows",
        "validation_rows",
        "test_rows",
        "partition_count",
        "partition_sizes",
        "partition_rows",
        "global_batch_size",
        "local_batch_size",
        "epochs",
        "learning_rate",
        "seed",
        "distributed_training_wall_clock_seconds",
        "throughput_examples_per_second",
        "training_loss",
        "validation_loss",
        "validation_roc_auc",
        "validation_accuracy",
        "test_loss",
        "test_roc_auc",
        "test_accuracy",
        "environment",
        "git_sha",
        "model_artifact_path",
        "model_artifact_sha256",
    ]
    for key in required:
        if key not in obj:
            errors.append(f"missing field: {key}")

    if errors:
        return errors

    workers = int(obj["workers"])
    train_rows = int(obj["train_rows"])
    partition_sizes = [int(x) for x in obj["partition_sizes"]]
    expected_partition_rows = train_rows // workers if workers > 0 else -1
    if workers < 1:
        errors.append("workers must be >= 1")
    if train_rows < 1:
        errors.append("train_rows must be > 0")
    if int(obj["partition_count"]) != workers:
        errors.append("partition_count does not equal workers")
    if len(partition_sizes) != workers:
        errors.append("partition_sizes length does not equal workers")
    if workers > 0 and any(size != expected_partition_rows for size in partition_sizes):
        errors.append("worker partition sizes are not exactly equal")
    if int(obj["partition_rows"]) != expected_partition_rows:
        errors.append("partition_rows does not match train_rows / workers")
    if int(obj["global_batch_size"]) % workers != 0:
        errors.append("global_batch_size is not divisible by workers")
    if int(obj["local_batch_size"]) != int(obj["global_batch_size"]) // workers:
        errors.append("local_batch_size is inconsistent with global_batch_size / workers")

    if str(obj["git_sha"]).strip() in {"", "unknown", "None"}:
        errors.append("git_sha is missing or unknown")

    environment = obj["environment"]
    for key in ("pytorch", "pyspark", "hadoop_version", "java_version", "spark_master", "spark_app_id"):
        if not environment.get(key):
            errors.append(f"environment metadata missing: {key}")

    positive_fields = (
        "distributed_training_wall_clock_seconds",
        "throughput_examples_per_second",
        "learning_rate",
    )
    for key in positive_fields:
        try:
            if float(obj[key]) <= 0:
                errors.append(f"{key} must be > 0")
        except (TypeError, ValueError):
            errors.append(f"{key} is not numeric")

    for key in ("validation_accuracy", "test_accuracy", "validation_roc_auc", "test_roc_auc"):
        try:
            value = float(obj[key])
            if not 0.0 <= value <= 1.0:
                errors.append(f"{key} must be within [0, 1]")
        except (TypeError, ValueError):
            errors.append(f"{key} is not numeric")

    model_path = resolve_model_path(json_path, str(obj["model_artifact_path"]))
    if not model_path.is_file():
        errors.append(f"model artifact not found: {model_path}")
    else:
        try:
            actual_hash = sha256_file(model_path)
            if actual_hash != str(obj["model_artifact_sha256"]):
                errors.append("model artifact SHA-256 does not match retained metadata")
        except OSError as exc:
            errors.append(f"model artifact could not be hashed: {exc}")

    return errors


def load_artifacts(root: Path):
    rows = []
    rejected = []
    for path in sorted(root.rglob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if obj.get("measurement_type") != "actual_run" or obj.get("status") != "completed":
            continue
        errors = validate_completed_artifact(path, obj)
        if errors:
            rejected.append({"path": str(path), "reasons": errors})
            continue
        rows.append((path, obj))
    return rows, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing retained JSON run artifacts")
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    args = parser.parse_args()

    artifacts, rejected = load_artifacts(Path(args.input))
    groups = defaultdict(list)
    for path, obj in artifacts:
        groups[(int(obj["train_rows"]), int(obj["workers"]))].append((path, obj))

    summary = []
    for (train_rows, workers), items in sorted(groups.items()):
        times = [float(obj["distributed_training_wall_clock_seconds"]) for _, obj in items]
        throughput = [float(obj["throughput_examples_per_second"]) for _, obj in items]
        row = {
            "train_rows": train_rows,
            "workers": workers,
            "retained_runs": len(items),
            "training_seconds_mean": mean(times),
            "training_seconds_std": sample_std(times),
            "throughput_mean": mean(throughput),
            "throughput_std": sample_std(throughput),
            "artifact_paths": [str(path) for path, _ in items],
        }
        for metric in (
            "training_loss",
            "validation_loss",
            "validation_roc_auc",
            "validation_accuracy",
            "test_loss",
            "test_roc_auc",
            "test_accuracy",
        ):
            metric_mean, metric_std = metric_summary(items, metric)
            row[f"{metric}_mean"] = metric_mean
            row[f"{metric}_std"] = metric_std
        summary.append(row)

    baseline_by_rows = {}
    for row in summary:
        if row["workers"] == 1 and row["retained_runs"] > 0:
            baseline_by_rows[row["train_rows"]] = row["training_seconds_mean"]

    for row in summary:
        baseline = baseline_by_rows.get(row["train_rows"])
        if baseline is None or row["training_seconds_mean"] is None:
            row["speedup_vs_1_worker"] = None
            row["scaling_efficiency"] = None
            continue
        speedup = baseline / row["training_seconds_mean"]
        row["speedup_vs_1_worker"] = speedup
        row["scaling_efficiency"] = speedup / row["workers"]

    output = {
        "measurement_type": "derived_from_actual_runs",
        "artifact_count": len(artifacts),
        "rejected_artifact_count": len(rejected),
        "rejected_artifacts": rejected,
        "conditions": summary,
        "note": "Only completed actual-run artifacts passing integrity validation are analyzed. No values are imputed. Speedup and scaling efficiency remain null when a matching one-worker baseline is unavailable.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
