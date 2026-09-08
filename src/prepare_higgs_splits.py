#!/usr/bin/env python3
"""Create reproducible HIGGS development/test splits on HDFS.

The UCI dataset defines the final 500,000 observations as the test set. Because
that boundary is positional, this script refuses to proceed unless the source
is read as one Spark partition, then uses zipWithIndex to preserve source order.
The resulting train/validation/test datasets are written once and reused by all
benchmark conditions.
"""
from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import types as T

TOTAL_ROWS = 11_000_000
TRAIN_ROWS = 10_000_000
VALIDATION_ROWS = 500_000
TEST_ROWS = 500_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original UCI HIGGS CSV/CSV.GZ on HDFS")
    parser.add_argument("--output-root", required=True, help="HDFS directory for prepared splits")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("prepare-higgs-splits").getOrCreate()
    try:
        schema = T.StructType(
            [T.StructField("label", T.DoubleType(), False)]
            + [T.StructField(f"f{i}", T.DoubleType(), False) for i in range(28)]
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
        train_rdd = indexed.filter(lambda item: item[1] < TRAIN_ROWS).map(lambda item: item[0])
        validation_rdd = indexed.filter(
            lambda item: TRAIN_ROWS <= item[1] < TRAIN_ROWS + VALIDATION_ROWS
        ).map(lambda item: item[0])
        test_rdd = indexed.filter(lambda item: item[1] >= TRAIN_ROWS + VALIDATION_ROWS).map(lambda item: item[0])

        for name, rdd, expected in (
            ("train", train_rdd, TRAIN_ROWS),
            ("validation", validation_rdd, VALIDATION_ROWS),
            ("test", test_rdd, TEST_ROWS),
        ):
            df = spark.createDataFrame(rdd, schema=schema)
            actual = df.count()
            if actual != expected:
                raise RuntimeError(f"{name}: expected {expected} rows, found {actual}")
            df.write.mode("overwrite").parquet(f"{args.output_root}/{name}")
            print(f"wrote {name}: {actual} rows -> {args.output_root}/{name}")

        print("HIGGS split preparation completed without changing row membership.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
