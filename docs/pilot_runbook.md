# Environment-validation and benchmark runbook

This runbook separates **environment validation**, **data preparation**, and **empirical benchmarking**. No benchmark number is accepted until an actual run produces a retained artifact.

## 1. Capture environment

```bash
python src/environment_check.py | tee environment_report.json
```

Retain the output with the run artifacts. Do not edit version fields manually.

## 2. Validate HDFS and Spark

After placing the official UCI HIGGS file in HDFS:

```bash
hdfs dfs -ls <HDFS_DATA_PATH>
spark-submit --version
```

The source should be the official 11,000,000-row HIGGS file. The repository preparation script refuses a positional split unless Spark reads that source as exactly one partition.

## 3. Prepare provenance-preserving HDFS splits

```bash
spark-submit \
  --master <validated-cluster-master> \
  src/prepare_higgs_splits.py \
  --input <HDFS_HIGGS_CSV_GZ> \
  --output-root <HDFS_PREPARED_ROOT>
```

The script verifies 11,000,000 source rows, preserves the final 500,000 observations as the official test set, creates 500,000 validation observations from the preceding development pool, and creates exact training-volume datasets of 1M, 2.5M, 5M, and 10M rows.

The preparation step is not a benchmark and its runtime must not be mixed with training-time results.

## 4. Validate TorchDistributor orchestration

Run the synthetic pilot first:

```bash
spark-submit \
  --master <validated-cluster-master> \
  --conf spark.executor.instances=2 \
  --conf spark.executor.cores=1 \
  --conf spark.dynamicAllocation.enabled=false \
  src/pilot_torch_distributor.py \
  --num-processes 2 \
  --steps 5 \
  --batch-size 64 \
  --seed 42 \
  --output pilot_result.json
```

The exact Spark deployment and executor settings are environment-dependent and must be recorded rather than assumed. The pilot's two processes should have two schedulable task slots available concurrently.

## 5. Acceptance criteria

The pilot passes only if:

- Spark starts successfully in the intended non-local deployment mode.
- `TorchDistributor` is importable in that exact environment.
- The requested number of processes is actually observed.
- All processes initialize and terminate cleanly.
- The process group uses an available backend (`gloo` for CPU or an explicitly validated GPU backend).
- The prepared HDFS paths can be read by Spark.
- The environment provides at least one schedulable CPU core per requested TorchDistributor process without intentional oversubscription.
- Dynamic allocation is disabled for the controlled pilot/benchmark unless a separately documented design requires it.
- A retained JSON result and environment report are produced.
- The git commit SHA used for the pilot is recorded.

A successful synthetic pilot proves orchestration only. It does **not** prove HIGGS data loading, benchmark scalability, or predictive performance.

## 6. Run one benchmark pilot on real HIGGS

Start with the smallest locked condition: **1 worker, 1M training rows, 5 epochs**.

```bash
spark-submit \
  --master <validated-cluster-master> \
  --conf spark.executor.instances=1 \
  --conf spark.executor.cores=1 \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.executor.memory=<fixed-per-executor-memory> \
  src/train_higgs_distributed.py \
  --train-path <HDFS_PREPARED_ROOT>/train_1000000 \
  --validation-path <HDFS_PREPARED_ROOT>/validation_500000 \
  --test-path <HDFS_PREPARED_ROOT>/test_500000 \
  --workers 1 \
  --train-rows 1000000 \
  --validation-rows 500000 \
  --test-rows 500000 \
  --global-batch-size 1024 \
  --epochs 5 \
  --learning-rate 0.001 \
  --seed 42 \
  --git-sha <exact-repository-commit-sha> \
  --output results/worker1_rows1m_run1.json
```

The runner also writes `results/worker1_rows1m_run1.model.pt` containing the exact rank-0 model state and records its SHA-256 hash in the JSON artifact. Both files must be retained together.

The fixed per-executor memory value chosen during environment validation must remain the same for all worker counts. Only the executor count scales with worker count in the primary strong-scaling design: 1, 2, 4, and 8 executors respectively, each with one CPU core. If the target cluster cannot support this allocation, the unsupported worker level is excluded rather than oversubscribed or silently reconfigured.

Only after this completes cleanly should the larger worker/data-volume matrix be attempted.

## 7. Full benchmark matrix

For every feasible pair of:

- workers: 1, 2, 4, 8;
- training rows: 1M, 2.5M, 5M, 10M;
- repetition: target 3 runs;

execute the same script with only the declared worker count, training-volume path, and output path changed. For the primary strong-scaling comparison, use:

- `spark.dynamicAllocation.enabled=false`;
- `spark.executor.cores=1`;
- `spark.executor.instances=<workers>`;
- one fixed executor-memory value across all worker counts;
- one fixed driver-memory value across all runs;
- the same cluster/deployment mode and HDFS storage configuration.

This makes worker count the intended scaling factor while keeping the problem size fixed within each training-volume condition. The experiment must be described as **strong scaling**, not weak scaling.

The one-worker condition is the primary speedup denominator. A native single-process PyTorch run is a separate comparison and must never replace that denominator.

## 8. Retained run artifact requirements

Each JSON artifact must contain:

- actual status and measurement type;
- worker count and training row count;
- timing values;
- throughput;
- training loss;
- validation loss, ROC-AUC, and accuracy;
- test loss, ROC-AUC, and accuracy;
- software versions;
- hardware/cluster metadata;
- Spark application ID;
- HDFS paths;
- model/training configuration;
- exact git SHA;
- model artifact path and SHA-256 hash.

Each completed run must also retain the referenced `.model.pt` file. Do not manually edit empirical values after the run. Derived tables and figures should be generated from these retained artifacts.

## 9. Freeze before paper Results

Before writing Results, verify:

- all reported values map to an artifact;
- the artifact's model file exists and its SHA-256 hash matches;
- speedup uses the declared one-worker baseline for the same training volume;
- scaling efficiency uses the reported worker count;
- repeated runs are summarized with variability where available;
- test metrics are not used to select hyperparameters;
- unsupported worker counts are reported as unavailable rather than simulated;
- executor count, executor cores, memory, dynamic-allocation state, Spark master, and other material resource settings are identical to the declared design;
- the final software/environment metadata are retained.

No benchmark result should enter the paper until these checks pass.
