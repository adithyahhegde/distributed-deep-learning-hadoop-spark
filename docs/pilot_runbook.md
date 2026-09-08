# Environment-validation pilot runbook

This runbook is for validating the execution path before any empirical result is used in the paper.

## 1. Capture environment

```bash
python src/environment_check.py | tee environment_report.json
```

Retain the output with the run artifacts. Do not edit version fields manually.

## 2. Validate HDFS separately

After placing the official UCI HIGGS file or a controlled prepared representation in HDFS, verify:

```bash
hdfs dfs -ls <HDFS_DATA_PATH>
spark-submit --version
```

Then use a Spark job to read the path and report schema, row count, and a small deterministic sample. The official final 500,000 test rows must remain untouched.

## 3. Validate TorchDistributor orchestration

Run the synthetic pilot first. Example:

```bash
spark-submit \
  --master <validated-cluster-master> \
  --conf spark.executor.instances=2 \
  --conf spark.executor.cores=1 \
  src/pilot_torch_distributor.py \
  --num-processes 2 \
  --steps 5 \
  --batch-size 64 \
  --seed 42 \
  --output pilot_result.json
```

The exact Spark deployment and executor settings are environment-dependent and must be recorded rather than assumed.

## 4. Acceptance criteria

The pilot passes only if:

- Spark starts successfully in the intended non-local deployment mode.
- `TorchDistributor` is importable in that exact environment.
- The requested number of processes is actually observed.
- All processes initialize and terminate cleanly.
- The process group uses an available backend (`gloo` for CPU or an appropriate GPU backend when GPU execution is explicitly validated).
- A retained JSON result and environment report are produced.
- The git commit SHA used for the pilot is recorded.

A successful synthetic pilot proves orchestration only. It does **not** prove HDFS data loading, benchmark scalability, or predictive performance.

## 5. Freeze before benchmark

Before running HIGGS benchmarks, freeze:

- Python, PyTorch, PySpark/Spark, Hadoop, Java and supporting-library versions.
- Hardware and Spark executor resources.
- HDFS configuration and data representation.
- Model architecture, optimizer, learning rate, batch size, epochs and seed policy.
- Baseline definition and timing boundary.
- Worker-count and training-volume matrix.

No benchmark result should enter the paper until these fields are retained with the run artifact.
