#!/usr/bin/env python3
"""Environment validation for the distributed deep-learning pilot.

This script reports versions and basic runtime facts only. It does not claim that
Spark/HDFS/TorchDistributor are usable until the explicit pilot succeeds.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone


def version_of(module_name: str):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", "installed")
    except Exception as exc:  # pragma: no cover - environment-dependent
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def command_version(command: str, args: list[str] | None = None):
    path = shutil.which(command)
    if not path:
        return {"path": None, "version": "NOT_FOUND"}
    try:
        result = subprocess.run(
            [path, *(args or ["version"])],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        text = (result.stdout or result.stderr).strip().splitlines()
        return {"path": path, "version": text[0] if text else "UNKNOWN"}
    except Exception as exc:  # pragma: no cover
        return {"path": path, "version": f"ERROR: {type(exc).__name__}: {exc}"}


def main() -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "modules": {
            "torch": version_of("torch"),
            "pyspark": version_of("pyspark"),
            "numpy": version_of("numpy"),
            "pandas": version_of("pandas"),
            "sklearn": version_of("sklearn"),
            "pyarrow": version_of("pyarrow"),
        },
        "commands": {
            "java": command_version("java", ["-version"]),
            "hadoop": command_version("hadoop", ["version"]),
            "hdfs": command_version("hdfs", ["version"]),
            "spark-submit": command_version("spark-submit", ["--version"]),
        },
        "environment": {
            key: os.environ.get(key)
            for key in ["JAVA_HOME", "HADOOP_HOME", "HADOOP_CONF_DIR", "SPARK_HOME", "PYSPARK_PYTHON"]
            if os.environ.get(key)
        },
    }

    try:
        import torch
        payload["torch_runtime"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    except Exception as exc:  # pragma: no cover
        payload["torch_runtime"] = {"error": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
