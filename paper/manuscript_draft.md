# Performance and Scalability Evaluation of Distributed Deep Learning Using Hadoop and Apache Spark

**ADITHYA H HEGDE**  
**PES1PG25MB432**  
**MBA Trimester III**  
**PES University**  
**Faculty: Nitesh Rajamani**

> **WORKING MANUSCRIPT — EMPIRICAL RESULTS PENDING**
>
> This document is a research manuscript draft, not the final submission. No benchmark result is included unless it comes from a retained, validated actual-run artifact.

## Abstract

Distributed deep learning performance depends on the interaction of model computation, data movement, scheduling, storage, synchronization, and available compute resources. This study develops a controlled evaluation of a PyTorch feed-forward neural network trained through Apache Spark's TorchDistributor path over data stored in a Hadoop/HDFS environment. The UCI HIGGS dataset is used as the empirical workload. The study fixes the model architecture, optimizer, learning rate, global batch size, epoch budget, dataset definitions, software target, and timing rules while varying worker count and training-data volume. The primary worker-scaling experiment is designed as strong scaling, with one-worker TorchDistributor execution serving as the speedup baseline. Training time, throughput, speedup, scaling efficiency, predictive metrics, and cluster metadata are retained as run artifacts. The methodology introduces explicit controls for the official HIGGS test boundary, deterministic worker partitioning, equal partition sizes, model-artifact integrity, and reproducibility metadata. At this manuscript stage, the target Spark/HDFS/PyTorch environment has not yet produced admissible benchmark runs; therefore, empirical results, figures, and performance conclusions remain pending.

**Keywords:** distributed deep learning; Apache Spark; Hadoop; HDFS; PyTorch; TorchDistributor; scalability; strong scaling; HIGGS

## 1. INTRODUCTION

### 1.1 Background

Deep learning workloads increasingly operate on datasets large enough for storage, data movement, and distributed execution to become systems-level concerns. Distributed training divides computation across workers, but additional workers do not guarantee proportional reductions in elapsed training time. Synchronization, communication, partitioning, scheduling, and storage can limit the benefit obtained from additional parallelism.

Apache Spark is treated primarily as the distributed data-processing and orchestration layer, while PyTorch performs model training. Hadoop/HDFS is treated as the distributed storage and cluster ecosystem rather than as a competing deep-learning framework. The research question therefore concerns a particular Spark-orchestrated PyTorch path over HDFS-backed data, not a universal comparison of Hadoop, Spark, and PyTorch.

### 1.2 Problem Statement

Earlier Spark-oriented distributed deep-learning studies provide evidence about distributed training, storage, networking, and resource effects, but many use older software stacks or different training frameworks. The practical problem is to establish a reproducible contemporary baseline for Spark-orchestrated PyTorch training and quantify how worker count affects performance under controlled conditions.

### 1.3 Research Question

How does increasing distributed worker count affect training time, speedup, scaling efficiency, and predictive performance of a deep neural network when Spark orchestrates distributed PyTorch training over data stored in a Hadoop/HDFS environment?

### 1.4 Objectives

1. Establish a reproducible Spark 3.5.9 + TorchDistributor + PyTorch distributed-training pipeline over HDFS-backed Parquet data.
2. Measure training time and throughput across 1, 2, 4, and 8 workers where the target environment satisfies the resource gate.
3. Calculate speedup and scaling efficiency against a matched one-worker TorchDistributor baseline.
4. Measure validation and test loss, ROC-AUC, and accuracy.
5. Retain software, hardware, configuration, partition, and model-artifact metadata for accepted runs.

### 1.5 Scope

The study evaluates one fixed feed-forward binary classifier, one public dataset, one pinned Spark version, and defined worker/data-volume conditions. Conclusions are restricted to the tested cluster configuration and workload.

## 2. LITERATURE REVIEW

BigDL established a Spark-native distributed deep-learning approach and demonstrated the feasibility of running deep-learning applications on Hadoop/Spark clusters. Cruz, Tous, and Otero evaluated distributed deep-learning deployment with Spark on MareNostrum and emphasized parallelism, storage, networking, and configuration. Langer, Hall, He, and Rahayu proposed MPCA-SGD for distributed deep learning on Spark and examined practical scalability constraints. DLoBD characterized multiple deep-learning-over-big-data stacks in terms of performance, scalability, accuracy, and resource utilization. Ahn, Kim, and You examined Spark on YARN and TensorFlowOnSpark. Phan and Do later discussed obstacles encountered when using Spark for distributed DNN training and proposed an alternative framework independent of Spark.

### 2.1 Research Gap

The literature provides strong historical evidence for distributed deep learning over Spark/Hadoop stacks, but it does not directly answer the controlled question posed here: how a current Spark-to-PyTorch integration behaves as worker count and training-data volume vary under a fixed model, global batch size, dataset split, software version, and reproducibility protocol. The proposed contribution is therefore a controlled empirical evaluation of this execution path rather than a claim of universal superiority.

## 3. RESEARCH FRAMEWORK AND ARCHITECTURE

**HDFS/Parquet → Spark DataFrame → deterministic equal worker partitioning → Spark TorchDistributor → PyTorch distributed training → retained rank-0 model artifact → validation/test evaluation.**

HDFS supplies distributed storage. Spark provides orchestration and partitioned data delivery. TorchDistributor bridges Spark execution to PyTorch distributed training. PyTorch implements the model, optimization, synchronization, and predictive evaluation.

## 4. METHODOLOGY

### 4.1 Dataset and Provenance

The study uses the UCI HIGGS dataset (Whiteson, 2014), containing 11,000,000 observations with a binary target and 28 real-valued features. UCI identifies the final 500,000 observations as the test partition. The preparation pipeline preserves this positional boundary and constructs a 10,000,000-row training partition and 500,000-row validation partition from the preceding development pool.

### 4.2 Leakage and Partition Controls

- Preserve the official final 500,000 HIGGS observations as the test set.
- Select training-volume conditions only from the fixed training partition.
- Keep validation observations outside model training.
- Do not compute preprocessing statistics from validation or test data.
- Retain deterministic source-row identifiers.
- Assign worker buckets by `source_row_id mod workers` and explicitly verify equal partition sizes.
- Refuse training when each worker does not receive exactly `train_rows / workers` observations.

### 4.3 Locked Model and Training Protocol

| Parameter | Setting |
|---|---|
| Input features | 28 |
| Architecture | 28 → 128 → 64 → 1 |
| Activation | ReLU |
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Global batch size | 1,024 |
| Local batch size | 1,024 / workers |
| Epochs | 5 |
| Seed | 42 with rank-specific initialization offsets |
| Candidate workers | 1, 2, 4, 8 |

### 4.4 Experimental Design

The primary scaling experiment uses strong scaling: for a fixed training volume, worker count increases while the model, global batch size, epoch budget, dataset representation, software environment, and intended resource configuration remain fixed. Training-volume conditions are 1,000,000; 2,500,000; 5,000,000; and 10,000,000 rows.

The primary speedup baseline is one worker using the same Spark 3.5.9 TorchDistributor DataFrame-integrated execution path. A native single-process PyTorch baseline, if later collected, is secondary and will not replace the matched one-worker denominator.

### 4.5 Measurements and Statistical Treatment

Each accepted run records distributed training wall-clock time, synchronized maximum worker-side training time, total job wall time, throughput, predictive metrics, partition metadata, software versions, cluster resources, Spark application ID, Git commit SHA, seed, configuration, and a SHA-256-verified model `state_dict` artifact.

For worker count `p`:

- Speedup: `S(p) = T1 / Tp`
- Scaling efficiency: `E(p) = S(p) / p`
- Throughput: `Q(p) = N / Tp`

Repeated runs are summarized using mean and sample standard deviation when sufficient observations exist. No values are imputed when a required baseline or repeated measurement is missing.

## 5. RESULTS

**EMPIRICAL RESULTS PENDING.**

No accepted target-cluster benchmark artifact exists at manuscript generation time. Training times, throughput, speedup, scaling efficiency, predictive metrics, resource measurements, figures, and statistical comparisons will be populated only from retained actual-run artifacts passing the repository evidence-integrity checks.

## 6. DISCUSSION

The discussion will interpret measured scaling behavior in terms of computation, communication, synchronization, data loading, partitioning, Spark scheduling, and resource contention. Causes will not be inferred from timing alone. A reduction in training time will be evaluated together with predictive performance and documented resource configuration.

## 7. OPERATIONAL AND SCALABILITY INTERPRETATION

Strong scaling measures how quickly a fixed workload completes as resources increase. It does not establish weak scaling, universal cluster scalability, or cost-optimality across arbitrary environments. HDFS/storage effects and distributed-compute effects will be discussed separately where measurements permit.

## 8. LIMITATIONS

1. One workload, one model architecture, and one pinned software configuration are evaluated.
2. The Spark DataFrame-integrated TorchDistributor path is version-sensitive and is pinned to Spark 3.5.9.
3. Worker levels depend on actual cluster resource availability and must not be simulated.
4. One cluster configuration cannot support universal claims about Hadoop/Spark scalability.
5. Uncertainty estimates will be limited if fewer than three valid repetitions are feasible.
6. No empirical performance claim can be made until retained target-cluster runs exist.

## 9. REPRODUCIBILITY

The repository contains dataset preparation, the locked experiment matrix, environment checks, pilot path, distributed training runner, retained-run analyzer, literature-control artifacts, and formatting specification. Accepted empirical runs require an explicit Git commit SHA, software and cluster metadata, exact partition accounting, and a SHA-256-verified model artifact.

## 10. CONCLUSION

This study establishes a controlled research design for evaluating distributed deep learning through a Spark-orchestrated PyTorch execution path over HDFS-backed data. The methodology separates storage, orchestration, model training, worker scaling, and predictive evaluation while controlling dataset provenance, partition balance, global batch size, timing boundaries, and retained-run integrity. A substantive empirical conclusion is deliberately deferred until the target Spark/HDFS/PyTorch environment produces reproducible benchmark artifacts.

## References

Ahn, H.-Y., Kim, H., & You, W. (2018). Performance study of distributed big data analysis in YARN cluster. *2018 International Conference on Information and Communication Technology Convergence (ICTC)*, 1261–1266. https://doi.org/10.1109/ICTC.2018.8539474

Cruz, L., Tous, R., & Otero, B. (2019). Distributed training of deep neural networks with Spark: The MareNostrum experience. *Pattern Recognition Letters, 125*, 174–178. https://doi.org/10.1016/j.patrec.2019.01.020

Dai, J., Wang, Y., Qiu, X., et al. (2019). BigDL: A distributed deep learning framework for big data. *Proceedings of the ACM Symposium on Cloud Computing*, 50–60. https://doi.org/10.1145/3357223.3362707

Langer, M., Hall, A., He, Z., & Rahayu, W. (2018). MPCA SGD—A method for distributed training of deep learning models on Spark. *IEEE Transactions on Parallel and Distributed Systems, 29*(11), 2540–2556. https://doi.org/10.1109/TPDS.2018.2833074

Lu, X., Shi, H., Biswas, R., Javed, M. H., & Panda, D. K. (2018). DLoBD: A comprehensive study of deep learning over big data stacks on HPC clusters. *IEEE Transactions on Multi-Scale Computing Systems, 4*(4), 635–648. https://doi.org/10.1109/TMSCS.2018.2845886

Phan, T., & Do, P. (2023). A novel framework to enhance the performance of training distributed deep neural networks. *Intelligent Data Analysis, 27*(3), 753–768. https://doi.org/10.3233/IDA-226710

Whiteson, D. (2014). *HIGGS* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5V312
