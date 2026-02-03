import torch
import numpy as np
import argparse
import os

import sim_core as sim
from model_def import DeepONet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=-1.5, help="Thickness offset (mm)")
    parser.add_argument("--ckpt", type=str, default="best_model_v2.pth", help="Checkpoint filename under checkpoints/")
    parser.add_argument("--scale_u", type=float, default=10000.0, help="Scale factor used during training (meters -> scaled)")
    args = parser.parse_args()

    TEST_T = float(args.t)
    device = "cpu"

    print(f"[Test] Comparing FEM vs DeepONet (v2) for t={TEST_T} mm...")
    print(" - Evaluating ONLY on valid mesh vertices")
    print(" - Using canonical coords/components internally")

    # 1) Ground truth FEM
    print("  1) Running Ground Truth FEM...")
    gmsh_inst = sim.generate_mesh(TEST_T, use_onshape=True)
    if gmsh_inst is None:
        return

    mesh, basis, u_fem, meta = sim.run_simulation(gmsh_inst)
    axes = tuple(meta["axes"])
    aL, aW, aT = axes

    coords_xyz = mesh.p.T  # (N,3)
    coords_lwt = sim.canonicalize_points(coords_xyz, axes)
    mins_lwt = coords_lwt.min(axis=0)
    spans_lwt = coords_lwt.max(axis=0) - mins_lwt
    x_in = 2.0 * sim.normalize_points(coords_lwt, mins_lwt, spans_lwt) - 1.0  # [-1,1]

    # FEM displacement at vertices (xyz)
    M = basis.probes(mesh.p)
    u_flat = M @ u_fem
    n = coords_xyz.shape[0]
    u_xyz = np.vstack([u_flat[:n], u_flat[n:2*n], u_flat[2*n:]]).T  # (N,3)

    # Canonical components (uL,uW,uT)
    u_gt = u_xyz[:, [aL, aW, aT]]

    # 2) DeepONet prediction
    print("  2) Running DeepONet Prediction...")
    model_path = os.path.join(sim.CHECKPOINT_DIR, args.ckpt)
    if not os.path.exists(model_path):
        print(f"Checkpoint not found: {model_path}")
        return

    model = DeepONet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    t_in = torch.tensor([[sim.t_to_branch_input(TEST_T)]], dtype=torch.float32, device=device)  # (1,1)
    x_in_t = torch.tensor(x_in[np.newaxis, :, :], dtype=torch.float32, device=device)           # (1,N,3)

    with torch.no_grad():
        u_pred_scaled = model(t_in, x_in_t).squeeze(0).cpu().numpy()  # (N,3) canonical, scaled

    u_pred = u_pred_scaled / float(args.scale_u)  # meters

    # 3) Error metrics
    err = np.linalg.norm(u_gt - u_pred, axis=1)
    defl = np.linalg.norm(u_gt, axis=1)

    max_defl = float(np.max(defl))
    mean_err = float(np.mean(err))
    max_err = float(np.max(err))

    # where is the maximum deflection along length?
    i_max = int(np.argmax(defl))
    L_coord = float(coords_lwt[i_max, 0])
    L_min, L_max = float(coords_lwt[:, 0].min()), float(coords_lwt[:, 0].max())
    L_ratio = (L_coord - L_min) / max(1e-12, (L_max - L_min))

    print("\n" + "=" * 60)
    print(f"[Accuracy Report | t={TEST_T} mm]")
    print("=" * 60)
    print(f" Max Deflection (FEM)     : {max_defl * 1000:.6f} mm")
    print(f" Mean Prediction Error    : {mean_err * 1000:.6f} mm")
    print(f" Max  Prediction Error    : {max_err * 1000:.6f} mm")
    if max_defl > 1e-12:
        rel = (mean_err / max_defl) * 100.0
        print(f" Relative Error (Mean)    : {rel:.2f} %")
    print("-" * 60)
    print(f" FEM max deflection @ L~{L_ratio*100:.1f}% of length (canonical)")
    print("=" * 60)


if __name__ == "__main__":
    main()
