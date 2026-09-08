# Performance and Scalability Evaluation of Distributed Deep Learning Using Hadoop and Apache Spark

Research repository for evaluating distributed PyTorch deep-learning training orchestrated by Apache Spark over an HDFS-backed data environment.

## Research question

How does increasing distributed worker count affect training time, speedup, scaling efficiency, and predictive performance of a deep neural network when Spark orchestrates distributed PyTorch training over data stored in Hadoop/HDFS?

## Technology boundary

- **HDFS/Hadoop:** storage and cluster/data ecosystem.
- **Apache Spark:** distributed orchestration and execution layer.
- **PyTorch:** model training and optimization.
- **TorchDistributor:** Spark-to-PyTorch distributed execution path.

The study does not treat Hadoop MapReduce and Spark as competing deep-learning frameworks. It evaluates one explicitly defined Spark + PyTorch path while treating HDFS as the storage/cluster layer.

## Dataset

Primary dataset: **UCI HIGGS**. UCI documents 11,000,000 instances, 28 input features plus a binary target, no missing values, and a final 500,000-example test partition. The official test partition is preserved for final evaluation.

The preparation pipeline creates exact HDFS-backed Parquet datasets for:

- 1,000,000 training rows
- 2,500,000 training rows
- 5,000,000 training rows
- 10,000,000 training rows
- 500,000 validation rows
- 500,000 official test rows

## Locked benchmark configuration

- Workers: 1, 2, 4, 8, subject to environment feasibility.
- Model: 28 -> 128 -> 64 -> 1 feed-forward classifier.
- Activation: ReLU.
- Loss: BCEWithLogitsLoss.
- Optimizer: Adam.
- Learning rate: 0.001.
- Batch size: 1,024.
- Epochs: 5.
- Seed: 42.
- Primary baseline: 1-worker TorchDistributor using the same HDFS/Parquet path and timing boundary.

## Repository controls

- `docs/research_protocol.md` — locked research logic, leakage controls, metrics, baseline rules, and interpretation guardrails.
- `docs/literature_matrix.md` — literature synthesis and research-gap tracking.
- `docs/references_verified.md` — checked scholarly, dataset, and official technology source register.
- `docs/paper_format_spec.md` — verified visual/layout benchmark from the prior Operations/Capacity Planning paper.
- `configs/experiment_matrix.yaml` — locked worker/data-volume matrix and integrity controls.
- `src/environment_check.py` — captures actual Python/library/cluster runtime metadata.
- `src/prepare_higgs_splits.py` — creates provenance-preserving HDFS splits and exact training-volume datasets.
- `src/pilot_torch_distributor.py` — minimal synthetic TorchDistributor orchestration pilot; not a benchmark.
- `src/train_higgs_distributed.py` — real HIGGS Spark/TorchDistributor training and evaluation runner.
- `docs/pilot_runbook.md` — environment gate and benchmark execution procedure.

## Experimental integrity

**No fabricated results.** Benchmark values enter the paper only after an actual run produces a retained artifact containing the configuration and environment metadata needed to reproduce or audit the result.

No result, graph, timing, speedup, predictive metric, or resource measurement is considered evidence until it maps to a retained run artifact.

## Current status

- Research question: **locked**
- Dataset/provenance controls: **locked**
- Literature matrix: **established; source register verified**
- Experiment protocol: **locked for pilot**
- Formatting benchmark: **verified from rendered prior paper**
- Static validation workflow: **committed; execution status must be checked in GitHub Actions**
- Target distributed environment: **not yet validated**
- HDFS prepared dataset: **not yet generated in target environment**
- Synthetic distributed pilot: **not yet run in target environment**
- Real HIGGS benchmark: **not yet run**
- Paper Results section: **blocked until retained empirical runs exist**
- Final DOCX/PDF: **blocked until empirical analysis is complete**
