#!/usr/bin/env python3
"""Compatibility entry point for the controlled Spark 3.5.9 benchmark.

Spark 3.5.9 exposes the Spark-partition DataLoader helper under the private
``_get_spark_partition_data_loader`` name. The benchmark's worker function is
pickled by TorchDistributor and executed in a fresh torchrun process, so a
module-level alias on the driver is not sufficient. This adapter replaces the
worker function with an equivalent implementation that resolves the Spark 3.5.9
helper directly inside the worker process.
"""
from __future__ import annotations

import math
import os

import numpy as np

import train_higgs_distributed as benchmark


def train_partition_compat(
    num_samples: int,
    global_batch_size: int,
    workers: int,
    epochs: int,
    learning_rate: float,
    seed: int,
):
    """Run one DDP worker using Spark 3.5.9's actual private DataLoader helper."""
    import time
    import torch
    import torch.distributed as dist

    try:
        from pyspark.ml.torch.distributor import get_spark_partition_data_loader
    except ImportError:
        from pyspark.ml.torch.distributor import (
            _get_spark_partition_data_loader as get_spark_partition_data_loader
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != workers:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} does not match requested workers={workers}"
        )
    if global_batch_size % workers != 0:
        raise ValueError("global_batch_size must be divisible by workers")

    local_batch_size = global_batch_size // workers
    if local_batch_size <= 0:
        raise ValueError("local_batch_size must be positive")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
        owns_process_group = True
    else:
        owns_process_group = False

    device = torch.device("cpu")
    model = benchmark.build_model(torch).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    loader = get_spark_partition_data_loader(
        num_samples=num_samples,
        batch_size=local_batch_size,
        num_workers=0,
    )

    start = time.perf_counter()
    batches_per_epoch = math.ceil(num_samples / local_batch_size)
    final_epoch_loss_sum = 0.0
    final_epoch_count = 0

    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        epoch_count = 0
        epoch_batches = 0
        for batch in loader:
            values = batch.to(device=device, dtype=torch.float32)
            y = values[:, 0].view(-1, 1)
            x = values[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            batch_count = int(values.shape[0])
            epoch_loss_sum += float(loss.detach()) * batch_count
            epoch_count += batch_count
            epoch_batches += 1
            if epoch_batches >= batches_per_epoch:
                break

        if epoch == epochs - 1:
            final_epoch_loss_sum = epoch_loss_sum
            final_epoch_count = epoch_count

    elapsed = time.perf_counter() - start
    elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    synchronized_elapsed = float(elapsed_tensor.item())

    loss_tensor = torch.tensor(
        [final_epoch_loss_sum, float(final_epoch_count)], dtype=torch.float64
    )
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    final_epoch_training_loss = float(loss_tensor[0].item() / loss_tensor[1].item())

    state = {k: v.detach().cpu() for k, v in model.module.state_dict().items()}
    result = {
        "rank": rank,
        "world_size": world_size,
        "device": "cpu",
        "num_samples_per_worker": num_samples,
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "training_seconds_rank": elapsed,
        "training_seconds_synchronized_max": synchronized_elapsed,
        "batches_per_epoch": batches_per_epoch,
        "training_loss": final_epoch_training_loss,
        "state_dict": state if rank == 0 else None,
    }

    if owns_process_group:
        dist.destroy_process_group()
    return result


benchmark.train_partition = train_partition_compat


if __name__ == "__main__":
    benchmark.main()
