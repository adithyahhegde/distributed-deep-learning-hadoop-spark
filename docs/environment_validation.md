# Environment Validation Gate

This gate must be passed before any full benchmark result is accepted.

## Required software metadata

Record exact versions for:

- Python
- PyTorch
- PySpark / Apache Spark
- Hadoop / HDFS
- Java
- NumPy
- pandas or the chosen data reader
- scikit-learn (if used for metrics)

Also record the repository commit SHA used for every retained run.

## Required hardware/cluster metadata

Record:

- driver host type;
- worker/executor host type;
- CPU model and core count;
- RAM per host;
- GPU model/count, if GPUs are used;
- network type/speed where known;
- Spark master/deployment mode;
- executor count;
- executor cores;
- executor memory;
- task CPU/GPU allocation;
- HDFS replication factor and relevant block configuration;
- Spark version and Hadoop version.

## Functional checks

1. HDFS can store and read the HIGGS data path.
2. Spark can read the data from HDFS.
3. The PyTorch environment is available to distributed workers.
4. `TorchDistributor` imports successfully.
5. A two-process or otherwise smallest supported non-local distributed training pilot completes successfully.
6. The distributed workers can initialize the PyTorch process group and terminate cleanly.
7. The same input and model configuration produces valid metrics across repeated pilot runs.
8. Run metadata can be written to a retained artifact location.

Apache Spark documents `TorchDistributor` as a PySpark class for distributed PyTorch/PyTorch Lightning training. Its current API exposes `num_processes`, `local_mode`, and `use_gpu`; the non-local execution mode is the relevant path for executor-based distributed training and must be validated in the target environment.

## Baseline checks

Before measuring speedup, establish:

- the exact single-machine PyTorch baseline;
- the exact single-worker distributed baseline, if used;
- whether HDFS is used in both paths;
- whether startup/data-loading time is included in each timing boundary;
- whether preprocessing is performed once or repeated inside each run.

Do not calculate speedup until the baseline is explicitly frozen.

## Pilot acceptance criteria

A pilot is accepted only if:

- all processes start and finish without hidden retries;
- the run records the expected worker/process count;
- data volume is known exactly;
- training configuration is captured;
- loss/metric outputs are valid;
- wall-clock timing is captured;
- environment metadata are captured;
- the run can be tied to a repository commit.

A failed pilot is a research finding about the execution environment, not a result to be hidden. Fix the underlying issue or document the limitation before continuing.
