import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import json
import argparse

import sim_core as sim
from model_def import DeepONet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="beam_dataset_v2.npz", help="Dataset filename under data/")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale_u", type=float, default=10000.0, help="Scale factor for displacement targets (meters -> scaled)")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--out", type=str, default="best_model_v2.pth", help="Checkpoint filename under checkpoints/")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load dataset
    data_path = os.path.join(sim.DATA_DIR, args.dataset)
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}\nRun generate_data.py first.")
        return

    raw = np.load(data_path)

    # Expected:
    #  t: (S,1) in mm offset
    #  x: (S,P,3) in canonical normalized coords [-1,1]
    #  u: (S,P,3) in canonical displacement components (meters)
    t_raw = raw["t"].astype(np.float32)
    x_raw = raw["x"].astype(np.float32)
    u_raw = raw["u"].astype(np.float32)

    t_scaled = sim.t_to_branch_input(t_raw).astype(np.float32)  # [-1,1]
    x_scaled = x_raw  # already [-1,1]
    u_scaled = (u_raw * args.scale_u).astype(np.float32)

    # Torch tensors
    t_data = torch.tensor(t_scaled, dtype=torch.float32, device=device)
    x_data = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    u_data = torch.tensor(u_scaled, dtype=torch.float32, device=device)

    n_samples = t_data.shape[0]
    idx = np.random.permutation(n_samples)
    n_val = max(1, int(n_samples * args.val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = TensorDataset(t_data[train_idx], x_data[train_idx], u_data[train_idx])
    val_ds = TensorDataset(t_data[val_idx], x_data[val_idx], u_data[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2) Model
    model = DeepONet(hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Robust loss (on scaled targets)
    criterion = nn.HuberLoss(delta=0.1)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100
    )

    print(f"[System] Start Training on {device} (v2, canonical coords/components)...")
    print(f" - dataset: {data_path}")
    print(f" - t scaling: sim.t_to_branch_input (mm -> [-1,1])")
    print(f" - x scaling: already [-1,1] in dataset")
    print(f" - u scaling: x{args.scale_u}")
    print(f" - samples: {n_samples} (train {len(train_idx)}, val {len(val_idx)})")

    # 3) Train loop
    best_loss = float("inf")
    out_path = os.path.join(sim.CHECKPOINT_DIR, args.out)
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for t_batch, x_batch, u_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(t_batch, x_batch)
            loss = criterion(pred, u_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t_batch, x_batch, u_batch in val_loader:
                pred = model(t_batch, x_batch)
                loss = criterion(pred, u_batch)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        scheduler.step(val_loss)

        if epoch % 100 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:04d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.1e}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), out_path)

            meta = {
                "dataset": args.dataset,
                "scale_u": float(args.scale_u),
                "t_min_mm": float(sim.T_MIN),
                "t_max_mm": float(sim.T_MAX),
                "x_range": "[-1,1] canonical (L,W,T)",
                "u_components": "(uL,uW,uT) canonical",
                "model_hidden_dim": int(args.hidden_dim),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    print("[System] Training Finished.")
    print(f"Best Val Loss (scaled): {best_loss:.6f}")
    print(f"Checkpoint: {out_path}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
