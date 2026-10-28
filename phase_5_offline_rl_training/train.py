"""
train.py — Phase 5: Offline Policy Learning (reward-filtered behavioral cloning)
================================================================================
Trains the ContinuousActor (5 -> 128 -> 128 -> 2, Tanh output) on the top-X%
highest-reward transitions of the Phase 4 dataset and exports a TorchScript
policy plus normalization statistics and a provenance record.

CLI knobs (defaults reproduce the original v1 configuration exactly):
    --dataset   path to the Phase 4 parquet (default: rl_transitions_rpi_1.parquet)
    --top-pct   percentage of highest-reward transitions retained (default 30)
    --w-acc     reward weight on proxy accuracy   (default 10.0)
    --w-fps     reward weight on FPS              (default 0.5)
    --w-lat     reward weight on latency penalty  (default 0.1)
    --out-dir   output directory (default: phase_5_offline_rl_training/output)
    --seed      torch/numpy seed (default 0)

With the default weights the reward equals the collected `reward` column
validated by reward_components.recover_components; non-default weights trigger
an offline reward recomputation from the recovered components.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_components import recover_components, recompute_reward, COLLECTION_WEIGHTS


class ContinuousActor(nn.Module):
    """5-D telemetry state -> continuous (imgsz, p_skip) raw action in [-1,1]^2."""

    def __init__(self, state_dim=5, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--top-pct", type=float, default=30.0)
    ap.add_argument("--w-acc", type=float, default=COLLECTION_WEIGHTS["w_acc"])
    ap.add_argument("--w-fps", type=float, default=COLLECTION_WEIGHTS["w_fps"])
    ap.add_argument("--w-lat", type=float, default=COLLECTION_WEIGHTS["w_lat"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    workspace_dir = Path(__file__).resolve().parent.parent
    data_path = (Path(args.dataset) if args.dataset else
                 workspace_dir / "phase_4_rl_data_collection" / "output" / "rl_transitions_rpi_1.parquet")
    output_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading offline dataset from: {data_path}")
    df = pd.read_parquet(data_path)

    # Reward under the selected weight combination.
    if (args.w_acc, args.w_fps, args.w_lat) == (COLLECTION_WEIGHTS["w_acc"],
                                                COLLECTION_WEIGHTS["w_fps"],
                                                COLLECTION_WEIGHTS["w_lat"]):
        df = df[~df["done"].astype(bool)].copy()  # drop terminal rows (reward=0 placeholder)
        df["reward"] = df["reward"].astype(float)
    else:
        comp = recover_components(df)
        df = df.loc[comp.index].copy()
        df["reward"] = recompute_reward(comp, args.w_acc, args.w_fps, args.w_lat)

    # Reward-filtered behavioral cloning: retain the top `--top-pct` % by reward.
    cut = np.percentile(df["reward"].to_numpy(), 100.0 - args.top_pct)
    best_df = df[df["reward"] >= cut]
    print(f"[INFO] Filtered dataset for top {args.top_pct:g}% rewards (>= {cut:.2f}). "
          f"Rows remaining: {len(best_df)}")

    state_cols = ["state_cpu_percent", "state_ram_percent", "state_temperature",
                  "state_fps", "state_latency_ms"]
    action_cols = ["action_imgsz_raw", "action_skip_raw"]
    states = best_df[state_cols].values.astype(np.float32)
    actions = best_df[action_cols].values.astype(np.float32)

    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-8
    states_normalized = (states - state_mean) / state_std

    np.save(output_dir / "state_mean.npy", state_mean)
    np.save(output_dir / "state_std.npy", state_std)

    dataset = TensorDataset(torch.from_numpy(states_normalized), torch.from_numpy(actions))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("[INFO] Initializing PyTorch Continuous Actor Network...")
    actor = ContinuousActor(state_dim=5, action_dim=2)
    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 150
    print(f"[INFO] Commencing offline training for {epochs} epochs...")
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            loss = criterion(actor(batch_states), batch_actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {final_loss:.4f}")

    policy_path = output_dir / "rl_adaptive_policy.pt"
    actor.eval()
    traced = torch.jit.trace(actor, torch.zeros(1, 5))
    traced.save(str(policy_path))
    print(f"[SUCCESS] Exported TorchScript actor to: {policy_path}")

    # Provenance record: pins policy to dataset + hyperparameters.
    provenance = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(data_path),
        "dataset_sha256": sha256_file(data_path),
        "dataset_rows_total": int(len(pd.read_parquet(data_path))),
        "rows_used": int(len(best_df)),
        "top_pct": args.top_pct,
        "reward_weights": {"w_acc": args.w_acc, "w_fps": args.w_fps, "w_lat": args.w_lat},
        "architecture": "5 -> 128 -> 128 -> 2 (ReLU, Tanh)",
        "optimizer": "Adam lr=1e-3, MSE, batch=64, epochs=150",
        "seed": args.seed,
        "final_train_mse": float(final_loss),
        "policy_sha256": sha256_file(policy_path),
    }
    with open(output_dir / "policy_provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)
    print(f"[SUCCESS] Wrote provenance to {output_dir / 'policy_provenance.json'}")


if __name__ == "__main__":
    main()
