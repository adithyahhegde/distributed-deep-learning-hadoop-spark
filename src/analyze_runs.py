#!/usr/bin/env python3
"""Aggregate retained experiment artifacts without inventing missing measurements.

This script is intentionally conservative: it only consumes JSON artifacts whose
measurement_type is ``actual_run`` and whose status is ``completed``. Speedup and
scaling efficiency are calculated only when a matching one-worker baseline exists
for the same training volume. Missing or incomplete conditions remain missing.
"""
from __future__ import annotations

import argparse
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


def load_artifacts(root: Path):
    rows = []
    for path in sorted(root.rglob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if obj.get("measurement_type") != "actual_run" or obj.get("status") != "completed":
            continue
        required = [
            "workers",
            "train_rows",
            "distributed_training_wall_clock_seconds",
            "throughput_examples_per_second",
        ]
        if any(key not in obj for key in required):
            continue
        rows.append((path, obj))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing retained JSON run artifacts")
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    args = parser.parse_args()

    artifacts = load_artifacts(Path(args.input))
    groups = defaultdict(list)
    for path, obj in artifacts:
        groups[(int(obj["train_rows"]), int(obj["workers"]))].append((path, obj))

    summary = []
    for (train_rows, workers), items in sorted(groups.items()):
        times = [float(obj["distributed_training_wall_clock_seconds"]) for _, obj in items]
        throughput = [float(obj["throughput_examples_per_second"]) for _, obj in items]
        summary.append(
            {
                "train_rows": train_rows,
                "workers": workers,
                "retained_runs": len(items),
                "training_seconds_mean": mean(times),
                "training_seconds_std": sample_std(times),
                "throughput_mean": mean(throughput),
                "throughput_std": sample_std(throughput),
                "artifact_paths": [str(path) for path, _ in items],
            }
        )

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
        "conditions": summary,
        "note": "No values are imputed. Speedup and scaling efficiency remain null when a matching one-worker baseline is unavailable.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
