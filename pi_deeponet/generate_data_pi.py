from __future__ import annotations

import argparse
import os
import numpy as np
from tqdm import tqdm

from pi_config import BeamConfig
from pi_utils import ScaleConfig, physical_to_model_u
import sim_core_pi as sim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=80)
    p.add_argument("--t_min", type=float, default=-4.0)
    p.add_argument("--t_max", type=float, default=3.0)
    p.add_argument("--p_points", type=int, default=6000)
    p.add_argument("--out", type=str, default=os.path.join("data", "beam_dataset_pi.npz"))
    p.add_argument("--use_onshape", action="store_true", help="Download STEP from Onshape (requires env vars)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale_u", type=float, default=10000.0)
    p.add_argument("--no_thickness_scaling", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    cfg = BeamConfig()
    scfg = ScaleConfig(scale_u=args.scale_u, use_thickness_scaling=not args.no_thickness_scaling)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    t_values = np.linspace(args.t_min, args.t_max, args.num_samples).astype(np.float32)

    dataset_t = []
    dataset_x = []
    dataset_u = []
    dataset_spans = []
    dataset_center = []
    dataset_perm = []

    print(f"[DataGen PI] samples={args.num_samples}  t_range=[{args.t_min},{args.t_max}]  P={args.p_points}")
    print(f" output: {args.out}")

    for t in tqdm(t_values, desc="Generating"):
        gmsh_inst = sim.generate_mesh(float(t), cfg, use_onshape=args.use_onshape)
        if gmsh_inst is None:
            tqdm.write(f"[Warn] Mesh generation failed at t={t:.2f}")
            continue

        mesh, basis, u_fem = sim.run_simulation(gmsh_inst, cfg)
        coords, disp = sim.extract_vertex_displacement(mesh, basis, u_fem)
        x_hat, disp_c, spans, center, perm = sim.canonicalize_points_and_vectors(coords, disp)

        n = x_hat.shape[0]
        P = args.p_points
        replace = n < P
        idx = np.random.choice(n, P, replace=replace)

        x_s = x_hat[idx].astype(np.float32)
        u_phys = disp_c[idx].astype(np.float32)

        u_model = physical_to_model_u(u_phys, float(t), cfg, scfg).astype(np.float32)

        dataset_t.append([float(t)])
        dataset_x.append(x_s)
        dataset_u.append(u_model)
        dataset_spans.append(spans)
        dataset_center.append(center)
        dataset_perm.append(perm.astype(np.int16))

    if len(dataset_t) == 0:
        print("[CRITICAL] No samples generated.")
        return

    t_arr = np.asarray(dataset_t, dtype=np.float32)           # (N,1)
    x_arr = np.stack(dataset_x, axis=0).astype(np.float32)    # (N,P,3)
    u_arr = np.stack(dataset_u, axis=0).astype(np.float32)    # (N,P,3)
    spans_arr = np.asarray(dataset_spans, dtype=np.float32)   # (N,3)
    center_arr = np.asarray(dataset_center, dtype=np.float32) # (N,3)
    perm_arr = np.asarray(dataset_perm, dtype=np.int16)       # (N,3)

    np.savez(args.out, t=t_arr, x=x_arr, u=u_arr, spans=spans_arr, center=center_arr, perm=perm_arr)

    print(f"[DataGen PI] saved: {args.out}")
    print(f" - t: {t_arr.shape}  x: {x_arr.shape}  u: {u_arr.shape}")
    print(f" - spans: {spans_arr.shape}  center: {center_arr.shape}  perm: {perm_arr.shape}")
    print(f" - scale_u={scfg.scale_u}  thickness_scaling={scfg.use_thickness_scaling}")


if __name__ == "__main__":
    main()
