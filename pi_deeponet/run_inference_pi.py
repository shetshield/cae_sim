from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import pyvista as pv
from skfem import Mesh

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
    p.add_argument("--t", type=float, default=-3.0)
    p.add_argument("--ckpt", type=str, default=os.path.join("checkpoints_pi", "best_pi_deeponet.pth"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--warp", type=float, default=50.0)
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

    gmsh_inst = sim.generate_mesh(args.t, cfg, use_onshape=args.use_onshape)
    if gmsh_inst is None:
        return

    # load mesh vertices (no FEM solve)
    temp_msh = os.path.join(sim.MESH_DIR, "temp_infer.msh")
    gmsh_inst.write(temp_msh)
    mesh = Mesh.load(temp_msh)
    coords = mesh.p.T  # (N,3)

    # canonicalize coords (no displacement available here)
    x_hat, _, spans, center, perm = sim.canonicalize_points_and_vectors(coords, np.zeros_like(coords))

    t_tensor = torch.tensor([[args.t]], dtype=torch.float32, device=device)
    x_tensor = torch.tensor(x_hat[None, :, :], dtype=torch.float32, device=device)
    bfeat = make_branch_features(t_tensor, cfg, (args.t_min, args.t_max))

    with torch.no_grad():
        u_model = apply_hard_bc_gate(model(bfeat, x_tensor), x_tensor)
        u_canon = model_to_physical_u(u_model, t_tensor, cfg, scfg).squeeze(0).cpu().numpy()  # (N,3) in canonical axes

    # map back to global axes
    u_global = np.zeros_like(u_canon)
    for i in range(3):
        u_global[:, perm[i]] = u_canon[:, i]

    mag = np.linalg.norm(u_global, axis=1)

    cloud = pv.PolyData(coords)
    cloud["Displacement"] = u_global
    cloud["Magnitude"] = mag

    warped = cloud.warp_by_vector("Displacement", factor=args.warp)

    pl = pv.Plotter(shape=(1, 2))

    # isometric
    pl.subplot(0, 0)
    pl.add_text("Isometric", font_size=10)
    pl.add_mesh(warped, scalars="Magnitude", cmap="jet", show_edges=False)
    pl.add_axes()

    # side view (fixed -> tip)
    pl.subplot(0, 1)
    pl.add_text("Side View", font_size=10)
    pl.add_mesh(warped, scalars="Magnitude", cmap="jet", show_edges=False)

    # draw fixed and load planes in global coordinates
    axis_len = int(perm[0])
    axis_thk = int(perm[2])
    mins = coords.min(axis=0); maxs = coords.max(axis=0)
    x_min = mins[axis_len]; x_max = maxs[axis_len]
    L = x_max - x_min
    x_fix = x_min
    x_load = x_min + cfg.load_ratio * L

    # line along thickness
    y_center = 0.5 * (mins.sum() + maxs.sum())  # not used; keep simple below
    # center in other axes
    other_axes = [a for a in [0,1,2] if a != axis_len]
    c0 = 0.5 * (mins[other_axes[0]] + maxs[other_axes[0]])
    c1 = 0.5 * (mins[other_axes[1]] + maxs[other_axes[1]])

    # build start/end points for lines (vary thickness axis)
    zmin = mins[axis_thk]; zmax = maxs[axis_thk]
    # fixed plane line
    p0 = np.zeros(3); p1 = np.zeros(3)
    p0[axis_len] = x_fix; p1[axis_len] = x_fix
    p0[other_axes[0]] = c0; p1[other_axes[0]] = c0
    p0[other_axes[1]] = c1; p1[other_axes[1]] = c1
    p0[axis_thk] = zmin - 0.01; p1[axis_thk] = zmax + 0.02
    pl.add_mesh(pv.Line(p0, p1), color="white", line_width=3, label="Fixed")

    # load plane line
    q0 = p0.copy(); q1 = p1.copy()
    q0[axis_len] = x_load; q1[axis_len] = x_load
    pl.add_mesh(pv.Line(q0, q1), color="red", line_width=3, label="Load")

    pl.view_xz()
    pl.show_grid()
    pl.add_legend()
    pl.link_views()
    pl.show()


if __name__ == "__main__":
    main()
