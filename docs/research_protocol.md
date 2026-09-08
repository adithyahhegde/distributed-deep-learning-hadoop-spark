# Research Protocol

## Study title

**Performance and Scalability Evaluation of Distributed Deep Learning Using Hadoop and Apache Spark**

## Research question

How does increasing distributed worker count affect training time, speedup, scaling efficiency, and predictive performance of a deep neural network when Spark orchestrates distributed PyTorch training over data stored in a Hadoop/HDFS environment?

## System boundary

- **Hadoop/HDFS:** storage and cluster/data ecosystem.
- **Apache Spark:** distributed orchestration and execution layer.
- **PyTorch:** model training and optimization.
- **Spark TorchDistributor:** distributed bridge used for PyTorch training where supported by the target Spark environment.

Hadoop MapReduce and Spark are not treated as competing deep-learning frameworks. The experiment evaluates a Spark-orchestrated PyTorch training path with HDFS as the distributed storage layer.

## Primary dataset

UCI HIGGS is the primary candidate. UCI reports 11,000,000 instances, 28 real-valued input features plus a binary target, no missing values, and a final 500,000-example test partition. The official test partition must remain untouched during model selection and training-data-volume experiments.

## Data split and leakage controls

1. Preserve the official final 500,000 HIGGS examples as the test set.
2. Use the preceding 10,500,000 examples as the development pool.
3. Split the development pool into training and validation partitions using a documented, reproducible rule.
4. Any training-volume experiment must select subsets only from the training partition.
5. Validation data must not enter training.
6. The final test set is evaluated only after the model/training protocol has been locked.

## Model protocol

The default candidate is a small feed-forward binary classifier so that the study remains technically defensible and computationally practical:

- Input: 28 features.
- Hidden layers: 128 and 64 units.
- Activation: ReLU.
- Output: one logit.
- Loss: binary cross-entropy with logits.
- Optimizer: Adam.

The exact learning rate, batch size, epoch budget, seed policy, validation procedure, and any regularization must be frozen before the benchmark phase. They must not be tuned separately for different worker counts.

## Experimental factors

### Worker scaling

Candidate distributed worker/process levels are **1, 2, 4, and 8**, subject to the actual environment being able to execute each level reliably. Unsupported or unstable levels must be reported as unavailable rather than simulated.

### Training-data volume

Candidate training-volume levels should be selected from the fixed training partition after a pilot establishes feasible runtime. Suggested candidates are 1M, 2.5M, 5M, and 10.5M rows. The final matrix must record the exact row counts actually used.

### Controlled variables

For a scaling comparison, keep constant:

- model architecture;
- optimizer and learning rate;
- batch size;
- epoch budget;
- random-seed policy;
- train/validation/test definitions;
- feature preprocessing;
- evaluation code;
- software environment;
- data format and storage path for the compared runs;
- Spark executor/worker configuration except for the intended worker-count factor.

## Baseline definition

The baseline must be explicitly defined before results are accepted. A single-machine PyTorch baseline and a single-worker distributed configuration must not be treated as equivalent unless their execution paths and overheads are documented. Storage differences must also be separated from compute differences; otherwise a local-vs-HDFS comparison cannot be interpreted as a pure scaling result.

## Measurements

Every retained run should capture:

- wall-clock training time;
- data-loading/startup overhead separately where measurable;
- examples/second;
- speedup relative to the declared baseline;
- scaling efficiency;
- training/validation loss;
- ROC-AUC and/or accuracy;
- resource utilization where measurement is reliable;
- worker/process count;
- dataset row count;
- partition count;
- software versions;
- hardware/cluster configuration;
- git commit SHA;
- seed and training configuration.

For worker count `p`, with baseline time `T1` and distributed time `Tp`:

- **Speedup:** `S(p) = T1 / Tp`
- **Scaling efficiency:** `E(p) = S(p) / p`
- **Throughput:** `Q(p) = N / Tp`, where `N` is the number of training examples processed under the declared timing boundary.

The timing boundary must be identical across compared runs. If startup, data loading, or framework initialization is excluded from training time, it must be reported separately rather than silently discarded.

## Repetition and uncertainty

Use at least three retained runs per benchmark condition where feasible. If resource constraints prevent this, record the limitation and do not present a single run as evidence of stable performance. Report variability alongside central values when repeated runs exist.

## Evidence rule

No empirical result may enter the paper unless it maps to a retained run artifact and configuration. No benchmark number, graph, speedup, accuracy, scalability claim, or resource-utilization value may be invented or presented as illustrative in the Results section.

## Pilot gate

Before the full experiment:

1. validate the HDFS data path;
2. validate Spark-to-PyTorch distributed execution;
3. verify TorchDistributor behavior in the target Spark version;
4. run a small end-to-end pilot;
5. confirm that all worker counts intended for the benchmark are actually supported;
6. verify metric consistency;
7. verify that run metadata are retained;
8. revise only the pre-declared feasibility parameters if the pilot exposes a genuine infrastructure constraint.

## Interpretation guardrails

- Faster training is not automatically better if predictive performance changes materially.
- More workers do not imply linear speedup.
- Poor scaling must be investigated in terms of computation, communication, synchronization, data loading, partitioning, and resource contention before assigning a cause.
- HDFS/storage effects and distributed-compute effects must not be conflated.
- Results from a single machine must not be generalized to arbitrary clusters.
- The study measures the tested configuration and workload, not universal Spark/Hadoop performance.

## Current status

**Research design:** sufficiently specified to begin environment validation.

**Empirical benchmark:** not yet run.

**Paper Results section:** blocked until retained experimental runs exist.
