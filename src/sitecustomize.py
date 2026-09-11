"""Spark/PySpark compatibility shim loaded automatically for Python workers.

PySpark 3.5.x exposes the TorchDistributor partition loader internally as
``_get_spark_partition_data_loader``. The benchmark training function historically
imports the public spelling. Alias it before TorchDistributor launches worker Python
processes so the same implementation is available in both driver and executor processes.
"""

try:
    import pyspark.ml.torch.distributor as distributor

    if not hasattr(distributor, "get_spark_partition_data_loader"):
        distributor.get_spark_partition_data_loader = distributor._get_spark_partition_data_loader
except Exception:
    # Do not interfere with unrelated Python startup; the benchmark will surface
    # a normal import error if PySpark is unavailable.
    pass
