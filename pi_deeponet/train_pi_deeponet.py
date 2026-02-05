from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from pi_config import BeamConfig
from model_def import DeepONet
from pi_utils import (
    ScaleConfig,
    make_branch_features,
    apply_hard_bc_gate,
    spans_from_t_fallback,
    sample_interior_points,
    sample_fixed_face_points,
    sample_load_patch_points,
    traction_loss_from_model,
    energy_balance_loss_from_model,
    dirichlet_monitor_loss_from_model,
    l2_reg_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=os.path.join("data", "beam_dataset_pi.npz"))
    p.add_argument("--out_dir", type=str, default="checkpoints_pi")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--stage1_epochs", type=int, default=400, help="data-only warmup epochs")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)

    # model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--act", type=str, default="silu")

    # scaling (must match data generation)
    p.add_argument("--scale_u", type=float, default=10000.0)
    p.add_argument("--no_thickness_scaling", action="store_true")

    # physics weights
    p.add_argument("--w_trac", type=float, default=0.10)
    p.add_argument("--w_energy", type=float, default=0.10)
    p.add_argument("--w_reg", type=float, default=1e-6)

    # ramp schedule for physics terms (applied after stage1)
    p.add_argument("--ramp_len", type=int, default=600, help="epochs to ramp physics weights from 0 to 1 (after stage1)")
    p.add_argument("--clip_grad", type=float, default=5.0)

    # collocation sizes
    p.add_argument("--n_int", type=int, default=1024)
    p.add_argument("--n_bc", type=int, default=256)
    p.add_argument("--n_trac", type=int, default=512)

    # interior bias near fixed end
    p.add_argument("--bias_fixed", type=float, default=0.50)

    # out-of-range physics-only
    p.add_argument("--use_out_of_range", action="store_true")
    p.add_argument("--t_phys_min", type=float, default=-6.0)
    p.add_argument("--t_phys_max", type=float, default=5.0)
    p.add_argument("--alpha_out", type=float, default=0.05)

    p.add_argument("--print_every", type=int, default=50)
    return p.parse_args()


def ramp_value(epoch: int, stage1_epochs: int, ramp_len: int) -> float:
    if epoch < stage1_epochs:
        return 0.0
    if ramp_len <= 0:
        return 1.0
    e = epoch - stage1_epochs
    return float(np.clip(e / ramp_len, 0.0, 1.0))


def sample_t_out_of_range(
    B: int, t_min: float, t_max: float, t_phys_min: float, t_phys_max: float, device: torch.device
) -> torch.Tensor:
    """Uniform sample from [t_phys_min, t_min] ∪ [t_max, t_phys_max]."""
    # choose side randomly
    side = torch.randint(low=0, high=2, size=(B, 1), device=device)
    u = torch.rand((B, 1), device=device)
    t_left = t_phys_min + (t_min - t_phys_min) * u
    t_right = t_max + (t_phys_max - t_max) * u
    t = torch.where(side == 0, t_left, t_right)
    return t


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg = BeamConfig()
    scfg = ScaleConfig(scale_u=args.scale_u, use_thickness_scaling=not args.no_thickness_scaling)

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        return

    raw = np.load(args.dataset)
    t_np = raw["t"].astype(np.float32)  # (N,1)
    x_np = raw["x"].astype(np.float32)  # (N,P,3) normalized [-1,1]
    u_np = raw["u"].astype(np.float32)  # (N,P,3) model-space target
    spans_np = raw["spans"].astype(np.float32) if "spans" in raw.files else None

    N = t_np.shape[0]
    # shuffle indices
    idx = np.arange(N)
    np.random.shuffle(idx)
    n_train = int(N * 0.90)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    # tensors (keep on CPU to reduce GPU memory, move per batch)
    t_train = torch.from_numpy(t_np[train_idx])
    x_train = torch.from_numpy(x_np[train_idx])
    u_train = torch.from_numpy(u_np[train_idx])
    if spans_np is not None:
        s_train = torch.from_numpy(spans_np[train_idx])
    else:
        s_train = None

    t_val = torch.from_numpy(t_np[val_idx])
    x_val = torch.from_numpy(x_np[val_idx])
    u_val = torch.from_numpy(u_np[val_idx])
    if spans_np is not None:
        s_val = torch.from_numpy(spans_np[val_idx])
    else:
        s_val = None

    train_ds = TensorDataset(t_train, x_train, u_train) if s_train is None else TensorDataset(t_train, x_train, u_train, s_train)
    val_ds = TensorDataset(t_val, x_val, u_val) if s_val is None else TensorDataset(t_val, x_val, u_val, s_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = DeepONet(branch_dim=4, trunk_dim=3, hidden_dim=args.hidden_dim, depth=args.depth, output_dim=3, act=args.act).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=200)

    criterion = nn.HuberLoss(delta=0.1)

    # infer t_range from dataset (for features)
    t_min = float(t_np.min())
    t_max = float(t_np.max())
    t_range = (t_min, t_max)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best_pi_deeponet.pth")
    last_path = os.path.join(args.out_dir, "last_pi_deeponet.pth")

    print(f"[PI-DeepONet Stable] device={device}")
    print(f" dataset: {args.dataset}")
    print(f" out_dir: {args.out_dir}")
    print(f" branch_dim=4  n_train={len(train_idx)}  n_val={len(val_idx)}")
    print(f" physics-only(out-of-range)={'ON' if args.use_out_of_range else 'OFF'}")
    print(f" scale_u={scfg.scale_u} thickness_scaling={scfg.use_thickness_scaling}")
    print(f" stage1_epochs={args.stage1_epochs} ramp_len={args.ramp_len}")
    print(f" weights: w_trac={args.w_trac} w_energy={args.w_energy} w_reg={args.w_reg}")

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ramp = ramp_value(epoch, args.stage1_epochs, args.ramp_len)

        totals = []
        data_ls = []
        bc_ls = []
        trac_ls = []
        en_ls = []
        reg_ls = []
        U_ls = []
        W_ls = []
        out_ls = []

        for batch in train_loader:
            if s_train is None:
                t_b, x_b, u_b = batch
                t_b = t_b.to(device)
                x_b = x_b.to(device)
                u_b = u_b.to(device)
                spans_b = spans_from_t_fallback(t_b, cfg).to(device)
            else:
                t_b, x_b, u_b, spans_b = batch
                t_b = t_b.to(device)
                x_b = x_b.to(device)
                u_b = u_b.to(device)
                spans_b = spans_b.to(device)

            opt.zero_grad()

            bfeat = make_branch_features(t_b, cfg, t_range)
            # supervised prediction
            pred = model(bfeat, x_b)
            pred = apply_hard_bc_gate(pred, x_b)
            loss_data = criterion(pred, u_b)

            # stage1: data only
            loss = loss_data

            loss_bc = torch.tensor(0.0, device=device)
            loss_tr = torch.tensor(0.0, device=device)
            loss_en = torch.tensor(0.0, device=device)
            loss_reg = torch.tensor(0.0, device=device)
            out_loss = torch.tensor(0.0, device=device)
            U_mean = 0.0
            W_mean = 0.0

            if ramp > 0.0:
                # collocation points
                B = t_b.shape[0]
                x_int = sample_interior_points(B, args.n_int, device=device, bias_fixed=args.bias_fixed)
                x_bc = sample_fixed_face_points(B, args.n_bc, device=device)
                x_trac = sample_load_patch_points(B, args.n_trac, device=device, spans_m=spans_b, cfg=cfg)

                # monitor bc (hard BC -> ~0)
                loss_bc = dirichlet_monitor_loss_from_model(model, bfeat, t_b, x_bc, cfg, scfg)

                # traction + energy balance
                loss_tr, _ = traction_loss_from_model(model, bfeat, t_b, x_trac, spans_b, cfg, scfg)
                loss_en, en_stats = energy_balance_loss_from_model(model, bfeat, t_b, x_int, x_trac, spans_b, cfg, scfg)
                U_mean = en_stats["U_J"]
                W_mean = en_stats["W_J"]

                # small reg to suppress crazy outputs early
                loss_reg = l2_reg_loss(pred)

                phys = args.w_trac * loss_tr + args.w_energy * loss_en + args.w_reg * loss_reg
                loss = loss_data + ramp * phys

                # out-of-range physics-only (optional)
                if args.use_out_of_range and args.alpha_out > 0.0:
                    t_out = sample_t_out_of_range(
                        B, t_min=t_min, t_max=t_max,
                        t_phys_min=args.t_phys_min, t_phys_max=args.t_phys_max,
                        device=device
                    )
                    spans_out = spans_from_t_fallback(t_out, cfg).to(device)
                    bfeat_out = make_branch_features(t_out, cfg, t_range)

                    x_int_o = sample_interior_points(B, args.n_int, device=device, bias_fixed=args.bias_fixed)
                    x_tr_o = sample_load_patch_points(B, args.n_trac, device=device, spans_m=spans_out, cfg=cfg)

                    tr_o, _ = traction_loss_from_model(model, bfeat_out, t_out, x_tr_o, spans_out, cfg, scfg)
                    en_o, _ = energy_balance_loss_from_model(model, bfeat_out, t_out, x_int_o, x_tr_o, spans_out, cfg, scfg)

                    out_loss = args.w_trac * tr_o + args.w_energy * en_o
                    loss = loss + ramp * args.alpha_out * out_loss

            loss.backward()
            if args.clip_grad and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            opt.step()

            totals.append(float(loss.detach().cpu()))
            data_ls.append(float(loss_data.detach().cpu()))
            bc_ls.append(float(loss_bc.detach().cpu()))
            trac_ls.append(float(loss_tr.detach().cpu()))
            en_ls.append(float(loss_en.detach().cpu()))
            reg_ls.append(float(loss_reg.detach().cpu()))
            U_ls.append(float(U_mean))
            W_ls.append(float(W_mean))
            out_ls.append(float(out_loss.detach().cpu()))

        # train averages
        tr_total = float(np.mean(totals))
        tr_data = float(np.mean(data_ls))
        tr_bc = float(np.mean(bc_ls))
        tr_trac = float(np.mean(trac_ls))
        tr_en = float(np.mean(en_ls))
        tr_reg = float(np.mean(reg_ls))
        tr_U = float(np.mean(U_ls))
        tr_W = float(np.mean(W_ls))
        tr_out = float(np.mean(out_ls))

        # validation (data only)
        model.eval()
        vls = []
        with torch.no_grad():
            for batch in val_loader:
                if s_val is None:
                    t_b, x_b, u_b = batch
                    t_b = t_b.to(device)
                    x_b = x_b.to(device)
                    u_b = u_b.to(device)
                else:
                    t_b, x_b, u_b, _ = batch
                    t_b = t_b.to(device)
                    x_b = x_b.to(device)
                    u_b = u_b.to(device)
                bfeat = make_branch_features(t_b, cfg, t_range)
                pred = apply_hard_bc_gate(model(bfeat, x_b), x_b)
                vls.append(float(criterion(pred, u_b).detach().cpu()))
        val_data = float(np.mean(vls))

        scheduler.step(val_data)
        lr_now = opt.param_groups[0]["lr"]

        if epoch % args.print_every == 0 or epoch == args.epochs - 1:
            dt = time.time() - t0
            print(
                f"Epoch {epoch:04d} | "
                f"Train total={tr_total:.5f} data={tr_data:.5f} bc={tr_bc:.3e} "
                f"trac={tr_trac:.5f} en={tr_en:.5f} reg={tr_reg:.3e} "
                f"U={tr_U:.3e} W={tr_W:.3e} out={tr_out:.5f} | "
                f"Val(data)={val_data:.5f} | LR={lr_now:.2e} | ramp={ramp:.2f} | t={dt/60:.1f}m"
            )

        # save last
        torch.save({"model": model.state_dict(), "args": vars(args)}, last_path)

        # save best by val(data)
        if val_data < best_val:
            best_val = val_data
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    print("[Done]")
    print(f"Best Val(data): {best_val:.6f}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
