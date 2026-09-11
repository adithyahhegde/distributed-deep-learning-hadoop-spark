import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset


def init_ddp():
    dist.init_process_group(backend="gloo")
    return dist.get_rank(), dist.get_world_size()


def model():
    return nn.Sequential(nn.Linear(28, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    rank, world = init_ddp()
    assert world == args.workers
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.data)
    df = df[df.source_row_id % world == rank].sort_values("source_row_id")
    X = torch.tensor(df[[f"f{i}" for i in range(28)]].to_numpy(np.float32))
    y = torch.tensor(df["label"].to_numpy(np.float32)).view(-1, 1)
    local_bs = args.batch_size // world
    assert local_bs >= 1
    loader = DataLoader(TensorDataset(X, y), batch_size=local_bs, shuffle=False, num_workers=0)

    net = DDP(model())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    dist.barrier()
    start = time.perf_counter()
    final_loss = float("nan")
    for _ in range(args.epochs):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().item())
    elapsed = time.perf_counter() - start
    t = torch.tensor([elapsed], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    l = torch.tensor([final_loss], dtype=torch.float64)
    dist.all_reduce(l, op=dist.ReduceOp.SUM)
    l /= world
    if rank == 0:
        result = {
            "workers": world,
            "rows": len(df) * world,
            "epochs": args.epochs,
            "global_batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "train_time_max_s": float(t.item()),
            "throughput_rows_s": float(len(df) * world * args.epochs / t.item()),
            "final_loss": float(l.item()),
            "scope": "Plan B: actual UCI HIGGS subset; local PyTorch DDP; no HDFS/Spark in timed path",
        }
        Path(args.output).write_text(json.dumps(result, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
