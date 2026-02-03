import torch
import numpy as np
import pyvista as pv
import argparse
import os

import sim_core as sim
from model_def import DeepONet


def _set_side_view(plotter: pv.Plotter, width_axis: int):
    # Show (length vs thickness) by looking along width axis.
    if width_axis == 0:
        plotter.view_yz()  # look along +x
    elif width_axis == 1:
        plotter.view_xz()  # look along +y
    else:
        plotter.view_xy()  # look along +z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=-3.0, help="Thickness offset (mm)")
    parser.add_argument("--ckpt", type=str, default="best_model_v2.pth", help="Checkpoint filename under checkpoints/")
    parser.add_argument("--scale_u", type=float, default=10000.0, help="Scale factor used during training")
    parser.add_argument("--no_onshape", action="store_true", help="Disable Onshape STEP download (use cached STEP only)")
    parser.add_argument("--warp", type=float, default=200.0, help="Visual warp factor")
    args = parser.parse_args()

    use_onshape = (not args.no_onshape)
    device = "cpu"

    # 1) Load model
    model_path = os.path.join(sim.CHECKPOINT_DIR, args.ckpt)
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        return

    model = DeepONet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2) Build geometry points from REAL mesh (no arbitrary box grid)
    gmsh_inst = sim.generate_mesh(args.t, use_onshape=use_onshape)
    if gmsh_inst is None:
        return

    mesh = sim.load_mesh_from_gmsh(gmsh_inst, msh_filename="temp_infer.msh")
    coords_xyz = mesh.p.T  # (N,3)

    axes, mins_xyz, maxs_xyz, spans_xyz = sim.infer_axes_from_points(coords_xyz)
    aL, aW, aT = axes

    # Canonicalize coords (L,W,T) and normalize -> [-1,1]
    coords_lwt = sim.canonicalize_points(coords_xyz, axes)
    mins_lwt = coords_lwt.min(axis=0)
    spans_lwt = coords_lwt.max(axis=0) - mins_lwt
    x_in = 2.0 * sim.normalize_points(coords_lwt, mins_lwt, spans_lwt) - 1.0  # [-1,1]

    # 3) Predict (canonical components)
    t_in = torch.tensor([[sim.t_to_branch_input(args.t)]], dtype=torch.float32, device=device)
    x_in_t = torch.tensor(x_in[np.newaxis, :, :], dtype=torch.float32, device=device)

    print(f"[Inference v2] Predicting deformation for t={args.t} mm...")
    print(f" - inferred axes (L,W,T) = {axes} (global xyz indices)")
    print(f" - bbox spans (xyz, m): {spans_xyz}")

    with torch.no_grad():
        u_pred_scaled = model(t_in, x_in_t).squeeze(0).cpu().numpy()  # (N,3) in (uL,uW,uT), scaled

    u_pred_lwt = u_pred_scaled / float(args.scale_u)  # meters
    u_pred_xyz = sim.uncanonicalize_points(u_pred_lwt, axes)         # back to xyz for visualization

    mag = np.linalg.norm(u_pred_xyz, axis=1)
    i_max = int(np.argmax(mag))

    L_min = float(coords_xyz[:, aL].min())
    L_max = float(coords_xyz[:, aL].max())
    L = max(1e-12, (L_max - L_min))
    load_pos_L = L_min + sim.LOAD_RATIO * L

    # location of max predicted displacement along length
    L_at_max = float(coords_xyz[i_max, aL])
    L_ratio = (L_at_max - L_min) / L

    print(f"[Check] Load location (along length): {sim.LOAD_RATIO*100:.1f}%  -> coord={load_pos_L:.6f} m")
    print(f"[Check] Max |u_pred| @ {L_ratio*100:.1f}% of length  (coord={L_at_max:.6f} m)")
    print(f"[Check] Max |u_pred| = {mag[i_max]*1000:.6f} mm")

    # 4) Visualization
    cloud = pv.PolyData(coords_xyz)
    cloud["Displacement"] = u_pred_xyz
    cloud["Magnitude"] = mag

    warped = cloud.warp_by_vector("Displacement", factor=args.warp)

    pl = pv.Plotter(shape=(1, 2))

    # View 1: Isometric
    pl.subplot(0, 0)
    pl.add_text("Isometric View (Mesh Vertices)", font_size=10)
    pl.add_mesh(warped, scalars="Magnitude", cmap="jet", render_points_as_spheres=True, point_size=3)
    pl.add_axes()

    # Mark max point
    pl.add_mesh(pv.Sphere(radius=0.001 * spans_xyz.max(), center=coords_xyz[i_max]), color="white")

    # View 2: Side View (length-thickness)
    pl.subplot(0, 1)
    pl.add_text("Side View (Length vs Thickness)", font_size=10)
    pl.add_mesh(warped, scalars="Magnitude", cmap="jet", render_points_as_spheres=True, point_size=3)

    # Add markers: Fixed / Load / Tip
    W_center = 0.5 * (coords_xyz[:, aW].min() + coords_xyz[:, aW].max())
    T_min = float(coords_xyz[:, aT].min())
    T_max = float(coords_xyz[:, aT].max())
    margin = 0.2 * (T_max - T_min + 1e-12)

    def _line_at(pos_L, color, label):
        p1 = np.zeros(3, dtype=float)
        p2 = np.zeros(3, dtype=float)
        p1[aL] = pos_L; p2[aL] = pos_L
        p1[aW] = W_center; p2[aW] = W_center
        p1[aT] = T_min - margin
        p2[aT] = T_max + margin
        pl.add_mesh(pv.Line(p1, p2), color=color, line_width=3, label=label)

    _line_at(L_min, "green", f"Fixed ({L_min:.3f} m)")
    _line_at(load_pos_L, "red", f"Load ({load_pos_L:.3f} m)")
    _line_at(L_max, "blue", f"Tip ({L_max:.3f} m)")

    _set_side_view(pl, aW)
    pl.show_grid()
    pl.add_legend()
    pl.link_views()
    pl.show()


if __name__ == "__main__":
    main()
