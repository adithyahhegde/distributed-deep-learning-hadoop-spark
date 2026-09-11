#!/usr/bin/env python3
"""CPU single-node PyTorch DDP benchmark for the HIGGS experiment."""
import argparse, hashlib, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--volume", type=int, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--local-rank", "--local_rank", type=int, default=0)
    return p.parse_args()


def set_threads():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load_rank_data(root: Path, volume: int, workers: int, rank: int):
    frames = []
    cols = ["source_row_id", "label"] + [f"f{i}" for i in range(28)]
    for bucket in range(rank, 4, workers):
        frames.append(pd.read_parquet(root / "train" / f"bucket4={bucket}", columns=cols))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["source_row_id"] < volume].sort_values("source_row_id")
    x = df[[f"f{i}" for i in range(28)]].to_numpy(dtype=np.float32, copy=True)
    y = df["label"].to_numpy(dtype=np.float32, copy=True)
    return x, y


def load_eval(root: Path, name: str):
    df = pd.read_parquet(root / name, columns=["label"] + [f"f{i}" for i in range(28)])
    x = df[[f"f{i}" for i in range(28)]].to_numpy(dtype=np.float32, copy=True)
    y = df["label"].to_numpy(dtype=np.float32, copy=True)
    return x, y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def model_sha256(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    args = parse_args()
    set_threads()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != args.workers:
        raise RuntimeError(f"WORLD_SIZE={world} but --workers={args.workers}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + rank)
    dist.init_process_group(backend="gloo")
    root = Path(args.data_root)
    x, y = load_rank_data(root, args.volume, world, rank)
    expected = args.volume // world
    if len(x) != expected:
        raise RuntimeError(f"rank {rank}: expected {expected} rows, got {len(x)}")
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    local_batch = args.batch_size // world
    loader = DataLoader(ds, batch_size=local_batch, shuffle=True, num_workers=0)
    model = DDP(Net())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            optimizer.step()
    train_elapsed = time.perf_counter() - t0
    elapsed = torch.tensor([train_elapsed], dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    dist.barrier()
    result = {"rank": rank, "workers": world, "volume": args.volume, "rows_per_worker": len(x), "epochs": args.epochs, "global_batch_size": args.batch_size, "learning_rate": args.lr, "seed": args.seed, "train_time_max_s": float(elapsed.item())}
    if rank == 0:
        vx, vy = load_eval(root, "validation.parquet")
        tx, ty = load_eval(root, "test.parquet")
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model.module(torch.from_numpy(vx))).numpy()
            tp = torch.sigmoid(model.module(torch.from_numpy(tx))).numpy()
        result.update({"validation_loss": float(log_loss(vy, vp, labels=[0, 1])), "validation_auc": float(roc_auc_score(vy, vp)), "validation_accuracy": float(accuracy_score(vy, vp >= 0.5)), "test_loss": float(log_loss(ty, tp, labels=[0, 1])), "test_auc": float(roc_auc_score(ty, tp)), "test_accuracy": float(accuracy_score(ty, tp >= 0.5)), "model_sha256": model_sha256(model.module)})
        Path(args.output).write_text(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
