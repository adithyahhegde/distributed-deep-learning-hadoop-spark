from types import SimpleNamespace

from src.train_higgs_distributed import benchmark_signature


def make_args(workers):
    return SimpleNamespace(
        git_sha="abc123",
        workers=workers,
        train_rows=1_000_000,
        validation_rows=500_000,
        test_rows=500_000,
        global_batch_size=1024,
        epochs=5,
        learning_rate=0.001,
        seed=42,
    )


def make_environment(executor_instances, default_parallelism, executor_cores="4"):
    return {
        "python": "3.11",
        "platform": "test-platform",
        "pytorch": "2.5.0",
        "pyspark": "3.5.9",
        "hadoop_version": "3.3.6",
        "java_version": "17",
        "numpy": "2.0.0",
        "pandas": "2.2.0",
        "pyarrow": "15.0.0",
        "scikit_learn": "1.5.0",
        "spark_master": "spark://master:7077",
        "spark_executor_memory": "8g",
        "spark_executor_cores": executor_cores,
        "spark_executor_instances": executor_instances,
        "spark_default_parallelism": default_parallelism,
        "torch_cuda_available": False,
        "torch_cuda_version": None,
    }


def test_scaling_signature_ignores_worker_count_and_worker_count_dependent_settings():
    env_1 = make_environment(executor_instances="1", default_parallelism=4)
    env_4 = make_environment(executor_instances="4", default_parallelism=16)

    assert benchmark_signature(make_args(1), env_1) == benchmark_signature(make_args(4), env_4)


def test_scaling_signature_changes_when_invariant_compute_resources_change():
    env_4_cores = make_environment(executor_instances="4", default_parallelism=16, executor_cores="4")
    env_8_cores = make_environment(executor_instances="4", default_parallelism=32, executor_cores="8")

    assert benchmark_signature(make_args(4), env_4_cores) != benchmark_signature(make_args(4), env_8_cores)
