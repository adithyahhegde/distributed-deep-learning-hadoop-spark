# Performance and Scalability Evaluation of Distributed Deep Learning Using Hadoop and Apache Spark

Research repository for evaluating distributed PyTorch deep-learning training orchestrated by Apache Spark over an HDFS-backed data environment.

## Research question

How does increasing distributed worker count affect training time, speedup, scaling efficiency, and predictive performance of a deep neural network when Spark orchestrates distributed PyTorch training over data stored in Hadoop/HDFS?

## Technology boundary

- **HDFS/Hadoop:** storage and cluster/data ecosystem
- **Apache Spark:** distributed orchestration
- **PyTorch:** model training
- **TorchDistributor:** Spark-to-PyTorch distributed execution path, subject to validation in the target environment

Spark's current documentation exposes TorchDistributor for distributed PyTorch/PyTorch Lightning training, including non-local execution. The exact target Spark version remains an empirical environment-validation item.

## Dataset

Primary candidate: **UCI HIGGS**. UCI documents 11,000,000 instances, 28 input features plus a binary target, no missing values, and a final 500,000-example test partition. The official test partition is preserved for final evaluation.

## Repository controls

- `docs/research_protocol.md` — research logic, leakage controls, metrics, baseline rules, and interpretation guardrails.
- `docs/literature_matrix.md` — literature synthesis and research-gap tracking.
- `configs/experiment_matrix.yaml` — candidate worker/data-volume matrix and integrity controls.
- `src/environment_check.py` — captures the actual Python/library/cluster runtime metadata.
- `src/pilot_torch_distributor.py` — minimal synthetic TorchDistributor orchestration pilot; not a benchmark.
- `docs/pilot_runbook.md` — acceptance criteria and execution procedure for the environment gate.

## Experimental integrity

**No fabricated results.** Benchmark values enter the paper only after an actual run produces a retained artifact containing the configuration and environment metadata needed to reproduce or audit the result.

The repository does not treat Hadoop MapReduce and Spark as competing deep-learning frameworks. It evaluates the defined Spark + PyTorch execution path while treating Hadoop/HDFS primarily as the storage/cluster layer.

## Current status

- Research question: defined
- Dataset candidate: defined
- Literature matrix: established
- Experiment protocol: established
- Pilot implementation: committed
- Distributed execution environment: **validation pending**
- HDFS data-path validation: **pending**
- Pilot run: **not yet run in target environment**
- Baseline: **blocked until environment/timing boundary is frozen**
- Full benchmark: **blocked until pilot validation**
- Paper Results section: **blocked until retained empirical runs exist**
