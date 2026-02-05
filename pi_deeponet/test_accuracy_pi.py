from __future__ import annotations

import argparse
import os
import numpy as np
import torch

from pi_config import BeamConfig
from model_def import DeepONet
from pi_utils import (
    ScaleConfig,
    make_branch_features,
    apply_hard_bc_gate,
    model_to_physical_u,
)
import sim_core_pi as sim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t", type=float, default=-1.5)
    p.add_argument("--ckpt", type=str, default=os.path.join("checkpoints_pi", "best_pi_deeponet.pth"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use_onshape", action="store_true")
    p.add_argument("--t_min", type=float, default=-4.0)
    p.add_argument("--t_max", type=float, default=3.0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = BeamConfig()

    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        return

    ck = torch.load(args.ckpt, map_location=device)
    ck_args = ck.get("args", {})
    scale_u = float(ck_args.get("scale_u", 10000.0))
    use_th = not bool(ck_args.get("no_thickness_scaling", False))
    scfg = ScaleConfig(scale_u=scale_u, use_thickness_scaling=use_th)

    model = DeepONet(
        branch_dim=4,
        trunk_dim=3,
        hidden_dim=int(ck_args.get("hidden_dim", 128)),
        depth=int(ck_args.get("depth", 4)),
        output_dim=3,
        act=str(ck_args.get("act", "silu")),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    print(f"[Test PI] FEM vs PI-DeepONet at t={args.t} mm")
    print(f" ckpt: {args.ckpt}")
    print(f" scale_u={scfg.scale_u} thickness_scaling={scfg.use_thickness_scaling}")

    # FEM solve
    gmsh_inst = sim.generate_mesh(args.t, cfg, use_onshape=args.use_onshape)
    if gmsh_inst is None:
        return
    mesh, basis, u_fem = sim.run_simulation(gmsh_inst, cfg)
    coords, disp = sim.extract_vertex_displacement(mesh, basis, u_fem)
    x_hat, disp_c, spans, center, perm = sim.canonicalize_points_and_vectors(coords, disp)

    # predict
    t_tensor = torch.tensor([[args.t]], dtype=torch.float32, device=device)
    x_tensor = torch.tensor(x_hat[None, :, :], dtype=torch.float32, device=device)
    bfeat = make_branch_features(t_tensor, cfg, (args.t_min, args.t_max))

    with torch.no_grad():
        u_model = apply_hard_bc_gate(model(bfeat, x_tensor), x_tensor)
        u_phys = model_to_physical_u(u_model, t_tensor, cfg, scfg).squeeze(0).cpu().numpy()

    # error
    err = np.linalg.norm(disp_c - u_phys, axis=1)
    mag = np.linalg.norm(disp_c, axis=1)

    max_def = float(np.max(mag))
    mean_err = float(np.mean(err))
    max_err = float(np.max(err))
    rel = (mean_err / max_def * 100.0) if max_def > 1e-12 else float("nan")

    print("=" * 56)
    print(f" Max deflection (GT) : {max_def*1000:.4f} mm")
    print(f" Mean error          : {mean_err*1000:.4f} mm")
    print(f" Max error           : {max_err*1000:.4f} mm")
    print(f" Relative(mean/max)  : {rel:.2f} %")
    print("=" * 56)


if __name__ == "__main__":
    main()
