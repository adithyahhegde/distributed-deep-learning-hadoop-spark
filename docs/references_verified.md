# Verified Source Register

This file is a research-control register. It records sources that have been checked against a publisher, venue, institutional repository, DOI record, or official project documentation. It is not the final formatted APA reference list.

## Peer-reviewed / scholarly sources

1. Dai, J. J., Wang, Y., Qiu, X., Ding, D., Zhang, Y., Wang, Y., Jia, X., Zhang, C. L., Wan, Y., Li, Z., Wang, J., Huang, S., Wu, Z., Wang, Y., Yang, Y., She, B., Shi, D., Lu, Q., Huang, K., & Song, G. (2019). *BigDL: A distributed deep learning framework for big data*. Proceedings of the ACM Symposium on Cloud Computing, 50–60. https://doi.org/10.1145/3357223.3362707

2. Cruz, L., Tous, R., & Otero, B. (2019). Distributed training of deep neural networks with Spark: The MareNostrum experience. *Pattern Recognition Letters, 125*, 174–178. https://doi.org/10.1016/j.patrec.2019.01.020

3. Langer, M., Hall, A., He, Z., & Rahayu, W. (2018). MPCA SGD—A method for distributed training of deep learning models on Spark. *IEEE Transactions on Parallel and Distributed Systems, 29*(11), 2540–2556. https://doi.org/10.1109/TPDS.2018.2833074

4. Lu, X., Shi, H., Biswas, R., Javed, M. H., & Panda, D. K. (2018). DLoBD: A comprehensive study of deep learning over big data stacks on HPC clusters. *IEEE Transactions on Multi-Scale Computing Systems, 4*(4), 635–648. https://doi.org/10.1109/TMSCS.2018.2845886

5. Ahn, H. Y., Kim, H.-J., & You, W. (2018). Performance study of distributed big data analysis in YARN cluster. *2018 International Conference on Information and Communication Technology Convergence (ICTC)*, 1261–1266. https://doi.org/10.1109/ICTC.2018.8539474

6. Phan, T., & Do, P. (2023). A novel framework to enhance the performance of training distributed deep neural networks. *Intelligent Data Analysis, 27*(3), 753–768. https://doi.org/10.3233/IDA-226710

7. Dean, J., Corrado, G. S., Monga, R., Chen, K., Devin, M., Mao, M., Senior, A., Tucker, P., Yang, K., Le, Q. V., & Ng, A. Y. (2012). Large scale distributed deep networks. *Advances in Neural Information Processing Systems, 25*.

8. Chilimbi, T., Suzue, Y., Apacible, K., & Kalyanaraman, K. (2014). Project Adam: Building an efficient and scalable deep learning training system. *11th USENIX Symposium on Operating Systems Design and Implementation (OSDI 14)*, 571–582.

9. Li, M., Andersen, D. G., Park, J. W., Smola, A. J., Ahmed, A., Josifovski, V., Long, J., Shekita, E. J., & Su, B.-Y. (2014). Scaling distributed machine learning with the parameter server. *11th USENIX Symposium on Operating Systems Design and Implementation (OSDI 14)*, 583–598.

10. Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., Kudlur, M., Levenberg, J., Monga, R., Moore, S., Murray, D. G., Steiner, B., Tucker, P., Vasudevan, V., Warden, P., Wicke, M., Yu, Y., & Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)*, 265–283.

## Dataset and official technology sources

11. Whiteson, D. (2014). *HIGGS* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5V312

12. Apache Spark. (current documentation). *TorchDistributor — PySpark API reference*. https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.torch.distributor.TorchDistributor.html

13. Apache Spark. (current documentation). *Source code for pyspark.ml.torch.distributor*. https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/torch/distributor.html

## Verification notes

- The BigDL bibliographic record and DOI are confirmed from the project citation record and ACM venue metadata.
- Cruz et al. DOI and bibliographic details are confirmed by the institutional repository record.
- Langer et al. DOI and journal details are confirmed by DBLP/IEEE-linked records.
- Lu et al. DOI and journal details are confirmed by the authors' publication record and DBLP.
- Ahn et al. DOI and conference details are independently corroborated by bibliographic indexes.
- Phan & Do DOI, volume, issue, pages, and publication year are confirmed by the journal publisher page.
- UCI confirms the HIGGS row count, feature structure, missing-value status, official final 500,000 test partition, DOI, and license.
- Apache Spark documentation confirms `TorchDistributor` availability and the `train_on_dataframe` / Spark-partition data-loader path used by this repository.

Any source used in the final paper should be checked once more during final reference formatting so author lists, capitalization, venue names, page ranges, and DOI URLs exactly match the authoritative record.
