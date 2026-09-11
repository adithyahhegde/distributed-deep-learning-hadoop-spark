#!/usr/bin/env python3
"""Compatibility entry point for Spark 3.5.x TorchDistributor DataFrame training.

Spark 3.5.9 exposes the partition DataLoader helper as the private
``_get_spark_partition_data_loader`` symbol, while the benchmark training function imports
``get_spark_partition_data_loader``.  Install the exact Spark-provided implementation under the
public spelling before loading the benchmark module.  No benchmark logic or measurements are
changed by this adapter.
"""
from __future__ import annotations

import pyspark.ml.torch.distributor as distributor

if not hasattr(distributor, "get_spark_partition_data_loader"):
    distributor.get_spark_partition_data_loader = distributor._get_spark_partition_data_loader

from train_higgs_distributed import main


if __name__ == "__main__":
    main()
