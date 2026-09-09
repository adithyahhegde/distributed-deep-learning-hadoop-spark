#!/usr/bin/env python3
"""Minimal TorchDistributor pilot.

Purpose: validate Spark -> TorchDistributor -> PyTorch process orchestration.
This deliberately uses synthetic data. It is NOT a benchmark and produces no
paper results. HDFS/data-path validation is a separate gate.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def train(steps: int, batch_size: int, seed: int):
    import torch
    import torch.distributed as dist

    torch.manual_seed(seed + int(os.environ.get("RANK", "0")))
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))

    initialized_here = False
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        initialized_here = True

    model = torch.nn.Sequential(
        torch.nn.Linear(28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    start = time.perf_counter()
    for _ in range(steps):
        x = torch.randn(batch_size, 28, device=device)
        y = torch.randint(0, 2, (batch_size, 1), device=device).float()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start

    result = {
        "rank": rank,
        "world_size": world_size,
        "device": str(device),
        "steps": steps,
        "batch_size": batch_size,
        "elapsed_training_seconds": elapsed,
        "status": "pilot_ok",
    }

    if initialized_here:
        dist.destroy_process_group()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="pilot_result.json")
    args = parser.parse_args()

    if args.num_processes < 1:
        raise ValueError("num-processes must be >= 1")

    from pyspark.sql import SparkSession
    from pyspark.ml.torch.distributor import TorchDistributor

    spark = SparkSession.builder.appName("distributed-dl-torch-pilot").getOrCreate()
    try:
        distributor = TorchDistributor(
            num_processes=args.num_processes,
            local_mode=False,
            use_gpu=False,
        )
        start = time.perf_counter()
        result = distributor.run(
            train,
            args.steps,
            args.batch_size,
            args.seed,
        )
        elapsed = time.perf_counter() - start

        if not isinstance(result, dict):
            raise RuntimeError("TorchDistributor pilot did not return a rank-0 result dictionary")
        if int(result.get("rank", -1)) != 0:
            raise RuntimeError(f"Expected rank 0 result, observed {result.get('rank')}")
        if int(result.get("world_size", -1)) != args.num_processes:
            raise RuntimeError(
                f"Requested {args.num_processes} processes but observed world_size={result.get('world_size')}"
            )
        if result.get("status") != "pilot_ok":
            raise RuntimeError(f"Pilot returned unexpected status: {result.get('status')}")

        evidence = {
            "measurement_type": "environment_pilot",
            "status": "completed",
            "requested_processes": args.num_processes,
            "observed_rank0": int(result["rank"]),
            "observed_world_size": int(result["world_size"]),
            "device": result["device"],
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "pilot_training_seconds_rank0": float(result["elapsed_training_seconds"]),
            "pilot_job_wall_clock_seconds_driver": elapsed,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
