#!/usr/bin/env python3
"""Run the controlled HIGGS distributed-training experiment.

Primary path: HDFS Parquet -> Spark DataFrame -> TorchDistributor.train_on_dataframe -> PyTorch DDP.
The benchmark consumes provenance-preserving pre-split HDFS paths so the UCI final 500,000 test
observations remain untouched. This script records measurements and never fabricates results.

TorchDistributor.train_on_dataframe requires evenly divided input partitions. The preparation
script therefore retains source_row_id; this runner uses source_row_id modulo the worker count
to construct exactly equal worker buckets and verifies the partition sizes before training.

The benchmark keeps the *global* batch size constant across worker counts. Each worker therefore
uses local_batch_size = global_batch_size / workers, preventing worker-count changes from silently
changing the optimization regime.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

FEATURES = [f"f{i}" for i in range(28)]
COLUMNS = ["label", *FEATURES]


def build_model(torch):
    return torch.nn.Sequential(
        torch.nn.Linear(28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )


def train_partition(num_samples: int, global_batch_size: int, workers: int, epochs: int, learning_rate: float, seed: int):
    import os
    import time
    import torch
    import torch.distributed as dist
    from pyspark.ml.torch.distributor import get_spark_partition_data_loader

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != workers:
        raise RuntimeError(f"WORLD_SIZE={world_size} does not match requested workers={workers}")
    if global_batch_size % workers != 0:
        raise ValueError("global_batch_size must be divisible by workers")
    local_batch_size = global_batch_size // workers
    if local_batch_size <= 0:
        raise ValueError("local_batch_size must be positive")

    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        owns_process_group = True
    else:
        owns_process_group = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(torch).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    loader = get_spark_partition_data_loader(
        num_samples=num_samples,
        batch_size=local_batch_size,
        num_workers=0,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    last_loss = None
    batches_per_epoch = math.ceil(num_samples / local_batch_size)
    for _ in range(epochs):
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
            last_loss = float(loss.detach().cpu())
            epoch_batches += 1
            if epoch_batches >= batches_per_epoch:
                break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    synchronized_elapsed = float(elapsed_tensor.item())

    state = {k: v.detach().cpu() for k, v in model.module.state_dict().items()}
    result = {
        "rank": rank,
        "world_size": world_size,
        "device": str(device),
        "num_samples_per_worker": num_samples,
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "training_seconds_rank": elapsed,
        "training_seconds_synchronized_max": synchronized_elapsed,
        "batches_per_epoch": batches_per_epoch,
        "last_batch_loss": last_loss,
        "state_dict": state if rank == 0 else None,
    }

    if owns_process_group:
        dist.destroy_process_group()
    return result


def collect_environment(spark):
    import pandas
    import pyarrow
    import pyspark
    import sklearn
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pytorch": torch.__version__,
        "pyspark": pyspark.__version__,
        "hadoop_version": spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion(),
        "java_version": spark.sparkContext._jvm.java.lang.System.getProperty("java.version"),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
        "spark_master": spark.sparkContext.master,
        "spark_app_id": spark.sparkContext.applicationId,
        "spark_executor_memory": spark.conf.get("spark.executor.memory", None),
        "spark_executor_cores": spark.conf.get("spark.executor.cores", None),
        "spark_default_parallelism": spark.sparkContext.defaultParallelism,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }


def evaluate_streaming(spark_df, state_dict, batch_size: int):
    import torch
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    model = build_model(torch)
    model.load_state_dict(state_dict)
    model.eval()

    ys, ps = [], []
    batch_x, batch_y = [], []
    for row in spark_df.select(*COLUMNS).toLocalIterator():
        batch_y.append(float(row[0]))
        batch_x.append([float(row[i]) for i in range(1, 29)])
        if len(batch_y) >= batch_size:
            with torch.no_grad():
                p = torch.sigmoid(model(torch.tensor(batch_x, dtype=torch.float32))).view(-1).numpy()
            ys.extend(batch_y)
            ps.extend(p.tolist())
            batch_x, batch_y = [], []
    if batch_y:
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(batch_x, dtype=torch.float32))).view(-1).numpy()
        ys.extend(batch_y)
        ps.extend(p.tolist())

    labels = np.asarray(ys, dtype=np.int64)
    probs = np.asarray(ps, dtype=np.float64)
    return {
        "rows_evaluated": int(len(labels)),
        "accuracy": float(accuracy_score(labels, probs >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "log_loss": float(log_loss(labels, probs, labels=[0, 1])),
    }


def read_higgs(spark, path: str, include_source_row_id: bool = False):
    from pyspark.sql import types as T

    schema_fields = [T.StructField("label", T.DoubleType(), False)]
    schema_fields += [T.StructField(f"f{i}", T.DoubleType(), False) for i in range(28)]
    if include_source_row_id:
        schema_fields.append(T.StructField("source_row_id", T.LongType(), False))
    return spark.read.schema(T.StructType(schema_fields)).parquet(path)


def exact_worker_partition(df, workers: int, train_rows: int):
    from pyspark.sql import functions as F

    if train_rows % workers != 0:
        raise ValueError("train_rows must be divisible by workers for exact per-worker samples")
    expected = train_rows // workers
    bucketed = (
        df.withColumn("_worker_bucket", F.pmod(F.col("source_row_id"), F.lit(workers)))
        .repartition(workers, "_worker_bucket")
        .drop("_worker_bucket")
    )

    partition_sizes = bucketed.rdd.mapPartitions(lambda rows: [sum(1 for _ in rows)]).collect()
    if len(partition_sizes) != workers or any(size != expected for size in partition_sizes):
        raise RuntimeError(
            "TorchDistributor requires evenly divided partitions; observed "
            f"{partition_sizes}, expected {[expected] * workers}"
        )
    return bucketed.select(*COLUMNS), partition_sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--validation-rows", type=int, required=True)
    parser.add_argument("--test-rows", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.train_rows % args.workers != 0:
        raise ValueError("train_rows must be divisible by workers for exact partition sampling")
    if args.global_batch_size % args.workers != 0:
        raise ValueError("global_batch_size must be divisible by workers")

    from pyspark.sql import SparkSession
    from pyspark.ml.torch.distributor import TorchDistributor

    spark = (
        SparkSession.builder
        .appName(f"higgs-ddl-workers-{args.workers}-rows-{args.train_rows}")
        .getOrCreate()
    )
    try:
        start_job = time.perf_counter()
        train = read_higgs(spark, args.train_path, include_source_row_id=True)
        validation = read_higgs(spark, args.validation_path)
        test = read_higgs(spark, args.test_path)

        actual_train = train.count()
        actual_validation = validation.count()
        actual_test = test.count()
        if (actual_train, actual_validation, actual_test) != (
            args.train_rows,
            args.validation_rows,
            args.test_rows,
        ):
            raise ValueError(
                "Declared split counts do not match HDFS data: "
                f"train={actual_train}, validation={actual_validation}, test={actual_test}"
            )

        train_partition_df, partition_sizes = exact_worker_partition(
            train, args.workers, args.train_rows
        )
        partition_rows = args.train_rows // args.workers
        distributor = TorchDistributor(
            num_processes=args.workers,
            local_mode=False,
            use_gpu=False,
        )

        before_train = time.perf_counter()
        result = distributor.train_on_dataframe(
            train_partition,
            train_partition_df,
            partition_rows,
            args.global_batch_size,
            args.workers,
            args.epochs,
            args.learning_rate,
            args.seed,
        )
        end_train = time.perf_counter()
        distributed_training_seconds = end_train - before_train

        validation_metrics = evaluate_streaming(validation, result["state_dict"], args.global_batch_size)
        test_metrics = evaluate_streaming(test, result["state_dict"], args.global_batch_size)
        environment = collect_environment(spark)
        output = {
            "status": "completed",
            "measurement_type": "actual_run",
            "workers": args.workers,
            "train_rows": args.train_rows,
            "validation_rows": args.validation_rows,
            "test_rows": args.test_rows,
            "partition_count": len(partition_sizes),
            "partition_sizes": partition_sizes,
            "partition_rows": partition_rows,
            "global_batch_size": args.global_batch_size,
            "local_batch_size": args.global_batch_size // args.workers,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "job_wall_clock_seconds": time.perf_counter() - start_job,
            "distributed_training_wall_clock_seconds": distributed_training_seconds,
            "worker_training_seconds": result["training_seconds_synchronized_max"],
            "throughput_examples_per_second": args.train_rows / distributed_training_seconds,
            "worker_compute_throughput_examples_per_second": args.train_rows / result["training_seconds_synchronized_max"],
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "environment": environment,
            "git_sha": os.environ.get("GIT_COMMIT_SHA", "unknown"),
            "train_path": args.train_path,
            "validation_path": args.validation_path,
            "test_path": args.test_path,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(json.dumps(output, indent=2, default=str))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
