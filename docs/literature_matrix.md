# Literature Matrix

This matrix separates work directly relevant to Spark/Hadoop distributed deep learning from broader distributed-training systems. The broader systems are retained for architectural context; they are not treated as direct substitutes for the proposed Spark + HDFS + PyTorch experiment.

| Reference | Venue / year | Architecture or focus | Relevance to this study | Limitation / gap relative to this study |
|---|---|---|---|---|
| Dai et al., *BigDL: A Distributed Deep Learning Framework for Big Data* | ACM SoCC, 2019 | Distributed deep learning directly integrated with Apache Spark/Hadoop; data-parallel training | Establishes the historical Spark-native distributed DL lineage and motivates studying DL over big-data stacks | Uses a different framework and historical software stack; current study uses PyTorch with Spark TorchDistributor |
| Cruz, Tous, & Otero, *Distributed training of deep neural networks with Spark: The MareNostrum experience* | Pattern Recognition Letters, 2019 | Spark-based DNN deployment on a petascale HPC system; examines parallelism, storage, and networking | Strong evidence that end-to-end performance depends on more than worker count | Different hardware, workload, and software stack; does not answer the proposed controlled worker-scaling question on HIGGS |
| Langer et al., *MPCA SGD—A Method for Distributed Training of Deep Learning Models on Spark* | IEEE TPDS, 2018 | Distributed DL optimization/training on Spark | Directly relevant to distributed training on Spark and performance constraints outside specialized HPC networks | Different optimization/training method and experimental setup |
| Lu et al., *DLoBD: A Comprehensive Study of Deep Learning over Big Data Stacks on HPC Clusters* | IEEE TMSCS, 2018 | Comparative evaluation of CaffeOnSpark, TensorFlowOnSpark, MMLSpark/CNTKOnSpark, and BigDL; performance, scalability, accuracy, resources | Direct precedent for evaluating DL stacks over big-data systems and for separating performance from accuracy/resource effects | Older stacks and HPC configuration; not a current PyTorch/TorchDistributor experiment |
| Ahn, Kim, & You, *Performance Study of Distributed Big Data Analysis in YARN Cluster* | IEEE ICTC, 2018 | Spark on YARN and TensorFlowOnSpark performance/scalability | Supports the Hadoop/YARN/Spark systems context and the importance of distributed execution behavior | Focuses on YARN/TensorFlowOnSpark rather than the current Spark/PyTorch path |
| Phan & Do, *A novel framework to enhance the performance of training distributed deep neural networks* | Intelligent Data Analysis, 2023 | Discusses obstacles encountered when using Spark for distributed DNN training and proposes a non-Spark framework | Useful counterpoint: Spark-based approaches can introduce framework-level performance constraints | Proposed framework is independent of Spark and therefore not the implementation used here |
| Dean et al., *Large Scale Distributed Deep Networks* | NeurIPS, 2012 | DistBelief, Downpour SGD, and distributed training at very large scale | Foundational distributed-DL work for understanding data/model replicas, synchronization, and scaling | Predates Spark/HDFS/TorchDistributor and uses a substantially different system |
| Chilimbi et al., *Project Adam: Building an Efficient and Scalable Deep Learning Training System* | USENIX OSDI, 2014 | Distributed DNN training with system-level co-design and asynchronous execution | Reinforces that communication/computation balance is central to scalability | Different system and workload; not Spark/Hadoop based |
| Li et al., *Scaling Distributed Machine Learning with the Parameter Server* | USENIX OSDI, 2014 | Distributed workers plus parameter-server architecture; consistency, communication, fault tolerance | Provides general distributed-ML context for synchronization, communication cost, and scalability | General distributed ML rather than Spark-specific deep learning |
| Abadi et al., *TensorFlow: A System for Large-Scale Machine Learning* | USENIX OSDI, 2016 | Large-scale dataflow execution across heterogeneous distributed devices | Provides background on distributed training system design and graph execution | TensorFlow system rather than Spark/Hadoop/PyTorch |

## Key synthesis

1. The literature consistently shows that distributed deep-learning performance is a systems problem, not only a model problem.
2. Spark/Hadoop-based DL studies demonstrate the relevance of storage, scheduling, networking, partitioning, and framework overhead.
3. Several older Spark-native frameworks are historically important but should not be assumed to represent current Spark capabilities.
4. The proposed study therefore uses a current Spark-to-PyTorch integration path while retaining older Spark-native work as comparative background.
5. The most defensible gap is a controlled, reproducible evaluation of worker scaling and training-data volume for a fixed PyTorch model under a Spark-orchestrated, HDFS-backed execution path, with explicit separation of training time, startup/data-loading overhead, predictive performance, and scaling efficiency.

## Verification status

The bibliographic metadata and relevance claims must be rechecked against publisher/venue records before the final paper bibliography is frozen. Empirical claims from the cited papers must be paraphrased accurately and cited in the paper; this repository matrix is a research-control artifact, not a substitute for the final APA 7 reference list.
