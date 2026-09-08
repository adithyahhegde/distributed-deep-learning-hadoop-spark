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


def train(steps: int, batch_size: int, seed: int, output_path: str):
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

    if rank == 0:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

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

    from pyspark.sql import SparkSession
    from pyspark.ml.torch.distributor import TorchDistributor

    spark = SparkSession.builder.appName("distributed-dl-torch-pilot").getOrCreate()
    try:
        distributor = TorchDistributor(
            num_processes=args.num_processes,
            local_mode=False,
            use_gpu=False,
        )
        distributor.run(
            train,
            args.steps,
            args.batch_size,
            args.seed,
            args.output,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
