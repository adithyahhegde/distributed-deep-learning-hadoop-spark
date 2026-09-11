from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pmod
from pyspark.sql.types import StructType, StructField, LongType, DoubleType
import sys

inp, out = sys.argv[1], sys.argv[2]
spark = SparkSession.builder.appName("prepare-higgs-ddp").getOrCreate()
fields = [StructField("source_row_id", LongType(), False), StructField("label", DoubleType(), False)] + [StructField(f"f{i}", DoubleType(), False) for i in range(28)]
df = spark.read.option("header", "false").schema(StructType(fields)).csv(inp)
train = df.filter(col("source_row_id") < 10_000_000).withColumn("bucket4", pmod(col("source_row_id"), 4))
train.write.mode("overwrite").partitionBy("bucket4").parquet(out + "/train")
val = df.filter((col("source_row_id") >= 10_000_000) & (col("source_row_id") < 10_500_000)).drop("source_row_id")
val.write.mode("overwrite").parquet(out + "/validation.parquet")
test = df.filter(col("source_row_id") >= 10_500_000).drop("source_row_id")
test.write.mode("overwrite").parquet(out + "/test.parquet")
spark.stop()
