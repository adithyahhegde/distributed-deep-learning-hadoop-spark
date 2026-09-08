#!/usr/bin/env python3
"""Create reproducible HIGGS splits and training-volume datasets on HDFS.

The UCI dataset defines the final 500,000 observations as the test set. Because
that boundary is positional, this script refuses to proceed unless the source
is read as one Spark partition, then uses zipWithIndex to preserve source order.
The resulting datasets are written once and reused by all benchmark conditions.

Training datasets retain a deterministic source_row_id so the benchmark runner
can construct exactly equal worker partitions for TorchDistributor rather than
assuming Spark's generic repartition operation is perfectly balanced.
"""
from __future__ import annotations

import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql import types as T

TOTAL_ROWS = 11_000_000
TRAIN_ROWS = 10_000_000
VALIDATION_ROWS = 500_000
TEST_ROWS = 500_000
TRAINING_VOLUMES = (1_000_000, 2_500_000, 5_000_000, 10_000_000)


def with_source_row_id(rdd):
    return rdd.zipWithIndex().map(lambda item: (*item[0], int(item[1])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original UCI HIGGS CSV/CSV.GZ on HDFS")
    parser.add_argument("--output-root", required=True, help="HDFS directory for prepared datasets")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("prepare-higgs-splits").getOrCreate()
    try:
        schema = T.StructType(
            [T.StructField("label", T.DoubleType(), False)]
            + [T.StructField(f"f{i}", T.DoubleType(), False) for i in range(28)]
        )
        indexed_schema = T.StructType(
            schema.fields + [T.StructField("source_row_id", T.LongType(), False)]
        )
        raw = spark.read.schema(schema).option("header", "false").csv(args.input)
        partitions = raw.rdd.getNumPartitions()
        if partitions != 1:
            raise RuntimeError(
                f"Refusing positional split: source was read as {partitions} Spark partitions; expected 1. "
                "Use the original single HIGGS.csv.gz source or explicitly validate a replacement ingestion path."
            )

        count = raw.count()
        if count != TOTAL_ROWS:
            raise RuntimeError(f"Expected {TOTAL_ROWS} rows from UCI HIGGS, found {count}")

        indexed = raw.rdd.zipWithIndex()
        train_rdd = indexed.filter(lambda item: item[1] < TRAIN_ROWS).map(
            lambda item: (*item[0], int(item[1]))
        )
        validation_rdd = indexed.filter(
            lambda item: TRAIN_ROWS <= item[1] < TRAIN_ROWS + VALIDATION_ROWS
        ).map(lambda item: item[0])
        test_rdd = indexed.filter(
            lambda item: item[1] >= TRAIN_ROWS + VALIDATION_ROWS
        ).map(lambda item: item[0])

        train_df = spark.createDataFrame(train_rdd, schema=indexed_schema).cache()
        validation_df = spark.createDataFrame(validation_rdd, schema=schema)
        test_df = spark.createDataFrame(test_rdd, schema=schema)

        counts = {
            "train": train_df.count(),
            "validation": validation_df.count(),
            "test": test_df.count(),
        }
        expected_counts = {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS, "test": TEST_ROWS}
        if counts != expected_counts:
            raise RuntimeError(f"Split counts mismatch: expected {expected_counts}, found {counts}")

        train_df.write.mode("overwrite").parquet(f"{args.output_root}/train_10000000")
        validation_df.write.mode("overwrite").parquet(f"{args.output_root}/validation_500000")
        test_df.write.mode("overwrite").parquet(f"{args.output_root}/test_500000")

        manifest = {
            "source_rows": TOTAL_ROWS,
            "training_rows": TRAIN_ROWS,
            "validation_rows": VALIDATION_ROWS,
            "official_test_rows": TEST_ROWS,
            "training_volume_rows": list(TRAINING_VOLUMES),
            "source_order_preserved": True,
            "source_partition_count": partitions,
            "training_row_id_column": "source_row_id",
            "training_row_id_range": [0, TRAIN_ROWS - 1],
        }

        for volume in TRAINING_VOLUMES:
            subset = train_df.filter(train_df.source_row_id < volume)
            actual = subset.count()
            if actual != volume:
                raise RuntimeError(f"training volume {volume}: expected {volume}, found {actual}")
            path = f"{args.output_root}/train_{volume}"
            subset.write.mode("overwrite").parquet(path)
            print(f"wrote training volume {volume}: {path}")

        print("SPLIT_MANIFEST=" + json.dumps(manifest, sort_keys=True))
        print("HIGGS split preparation completed without changing row membership.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
