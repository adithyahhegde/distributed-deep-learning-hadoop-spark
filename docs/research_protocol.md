# Research Protocol

## Study title

**Performance and Scalability Evaluation of Distributed Deep Learning Using Hadoop and Apache Spark**

## Research question

How does increasing distributed worker count affect training time, speedup, scaling efficiency, and predictive performance of a deep neural network when Spark orchestrates distributed PyTorch training over data stored in a Hadoop/HDFS environment?

## System boundary

- **Hadoop/HDFS:** storage and cluster/data ecosystem.
- **Apache Spark 3.5.9:** distributed orchestration and execution layer.
- **PyTorch:** model training and optimization.
- **Spark TorchDistributor:** distributed bridge used for PyTorch training.

Hadoop MapReduce and Spark are not treated as competing deep-learning frameworks. The experiment evaluates a Spark-orchestrated PyTorch training path with HDFS as the distributed storage layer.

The DataFrame-integrated TorchDistributor path is version-sensitive. The benchmark therefore pins PySpark 3.5.9 and uses Spark 3.5.9's `_train_on_dataframe` implementation explicitly rather than assuming a stable cross-version public API.

## Primary dataset and provenance

The primary dataset is the **UCI HIGGS** dataset. UCI reports 11,000,000 instances, a binary class label followed by 28 real-valued features, no missing values, and a final 500,000-example test partition.

The repository preparation script creates HDFS-backed Parquet datasets from the original UCI file. Because the test boundary is positional, the preparation step refuses to proceed unless the original source is ingested as one Spark partition, then uses `zipWithIndex` to preserve source order. It verifies the exact counts of 10,000,000 training rows, 500,000 validation rows, and 500,000 official test rows before writing the prepared datasets.

Training datasets retain a deterministic `source_row_id` so that worker partitioning can be checked rather than inferred from generic Spark balancing.

This is a preprocessing/provenance step, not a benchmark result.

## Data leakage controls

1. Preserve the official final 500,000 HIGGS observations as the test set.
2. Treat the preceding 10,500,000 observations as the development pool.
3. Use 10,000,000 observations for training and 500,000 for validation in the locked protocol.
4. Training-volume conditions select only from the fixed 10,000,000-row training partition.
5. Validation data never enter training.
6. The test set is evaluated only after the model and training protocol are locked.
7. No preprocessing statistics may be calculated using validation or test observations.
8. The same prepared HDFS paths are used across worker-count comparisons.

## Locked model protocol

The benchmark uses a small feed-forward binary classifier so the systems question remains understandable and computationally practical:

- Input: 28 features.
- Hidden layers: 128 and 64 units.
- Activation: ReLU.
- Output: one logit.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: Adam.
- Learning rate: **0.001**.
- Global batch size: **1,024**.
- Local batch size: **1,024 / workers**; therefore the effective global batch size is held constant across worker counts.
- Epoch budget: **5**.
- Seed: **42**, with rank-specific deterministic offsets for worker initialization.
- Preprocessing: none in the locked benchmark protocol.

These settings are fixed across worker counts and training-volume conditions. No condition-specific hyperparameter tuning is permitted.

## Experimental factors

### Worker scaling

Candidate distributed worker/process levels are **1, 2, 4, and 8**, subject to the actual environment being able to execute each level reliably. Unsupported or unstable levels are reported as unavailable rather than simulated.

### Training-data volume

Candidate training-volume levels are **1,000,000; 2,500,000; 5,000,000; and 10,000,000 rows**. All four values are divisible by 8, allowing exact equal per-worker sample counts at every candidate worker level.

For a training-volume condition below 10,000,000, the subset must be selected deterministically from the fixed training partition and the exact retained row count must be recorded in the run artifact.

### Controlled variables

For worker-scaling comparisons, keep constant:

- model architecture;
- optimizer and learning rate;
- **global batch size** and resulting fixed effective optimization regime;
- epoch budget;
- random-seed policy;
- train/validation/test definitions;
- preprocessing;
- evaluation code;
- software environment;
- prepared HDFS storage format and paths;
- executor/worker resource configuration except for the intended worker-count factor.

## TorchDistributor partitioning gate

The Spark DataFrame-integrated TorchDistributor path requires the input DataFrame to have evenly divided partitions, because each Spark task processes one partition. Divisibility of the total row count alone does not prove that generic Spark repartitioning produced equal partitions.

The repository therefore uses the retained `source_row_id` to assign each training row to `source_row_id mod workers`, repartitions by that deterministic worker bucket, counts the resulting Spark partitions, and refuses to start training unless every partition contains exactly `train_rows / workers` observations. This verification is retained as run metadata.

This gate is necessary for interpreting worker-count comparisons as controlled changes in parallelism rather than accidental differences in per-worker sample count.

## Primary baseline definition

The primary speedup baseline is **one worker using the same Spark 3.5.9 TorchDistributor DataFrame-integrated execution path** and the same HDFS data, model, optimizer, global batch size, epoch budget, timing boundary, and software environment.

This baseline is intentionally different from a native single-process PyTorch baseline. A native PyTorch run can be reported as a secondary comparison, but it must not be silently substituted for the one-worker Spark/TorchDistributor denominator because that would confound distributed scaling with framework/storage differences.

## Timing and measurements

Every retained run captures:

- distributed training wall-clock time;
- synchronized maximum worker-side training time;
- total job wall-clock time;
- examples/second;
- validation and test metrics;
- worker/process count;
- dataset row count;
- partition count and verified partition sizes;
- software versions;
- hardware/cluster configuration;
- Spark application ID;
- git commit SHA;
- seed and training configuration;
- exact trained model `state_dict` as a separate `.model.pt` artifact;
- SHA-256 hash of that model artifact.

The primary timing boundary is the elapsed time around the Spark 3.5.9 DataFrame-integrated TorchDistributor call. It therefore includes Spark barrier/data-transfer/orchestration overhead associated with the distributed training job. Worker-side model-computation time is retained separately and uses the synchronized maximum across workers rather than rank-0 time alone.

For worker count `p`, with one-worker baseline time `T1` and distributed time `Tp`:

- **Speedup:** `S(p) = T1 / Tp`
- **Scaling efficiency:** `E(p) = S(p) / p`
- **Throughput:** `Q(p) = N / Tp`, where `N` is the number of training examples under the declared training condition.

Startup and data-preparation measurements must be reported separately if they are measured outside the primary timing boundary. They must not be silently omitted when they materially affect the end-to-end interpretation.

## Predictive evaluation

The trained rank-0 model state is retained as a serialized `.model.pt` artifact and referenced by path and SHA-256 hash in the JSON run artifact. Validation and test predictions are evaluated using the same fixed model architecture and metric implementation.

Reported predictive metrics are:

- binary cross-entropy/log loss;
- ROC-AUC;
- accuracy.

Predictive performance is a secondary systems-control metric: the study is not claiming that worker scaling improves model quality. If worker count changes predictive performance materially, the result must be investigated before a scaling conclusion is made.

## Repetition and uncertainty

Use at least three retained runs per benchmark condition where feasible. Report mean, standard deviation, and/or confidence intervals when repeated observations support them. If only one run is feasible, label the estimate as a single-run measurement and explicitly limit the strength of the inference.

## Evidence rule

No empirical result may enter the paper unless it maps to a retained run artifact and configuration. No benchmark number, graph, speedup, accuracy, scalability claim, or resource-utilization value may be invented or presented as illustrative in the Results section.

## Pilot gate

Before the full experiment:

1. validate the HDFS data path;
2. validate Spark reading of the prepared Parquet splits;
3. validate Spark-to-PyTorch distributed execution;
4. verify the Spark 3.5.9 TorchDistributor DataFrame-integrated path and partition data loader in the target environment;
5. run a small end-to-end pilot;
6. confirm worker/process counts;
7. verify process-group initialization and clean shutdown;
8. verify metric consistency;
9. verify retained metadata and run artifacts, including the serialized model artifact;
10. freeze the environment and benchmark configuration.

## Interpretation guardrails

- Faster training is not automatically better if predictive performance changes materially.
- More workers do not imply linear speedup.
- Poor scaling must be investigated in terms of computation, communication, synchronization, data loading, partitioning, scheduling, and resource contention before assigning a cause.
- HDFS/storage effects and distributed-compute effects must not be conflated.
- Results from one cluster configuration must not be generalized to arbitrary Spark/Hadoop clusters.
- The study measures the tested configuration and workload, not universal Spark/Hadoop performance.

## Current status

**Research design:** locked for pilot execution.

**Partition-balance correctness gate:** implemented and statically validated; target execution still required.

**Global batch-size control:** implemented; target execution still required.

**Spark target version:** pinned to 3.5.9; exact target cluster execution still required.

**Model artifact retention:** implemented with SHA-256 integrity metadata; target execution still required.

**Environment validation:** pending actual target Spark/HDFS/PyTorch execution.

**Empirical benchmark:** not yet run.

**Paper Results section:** blocked until retained experimental runs exist.
