import numpy as np
import sim_core as sim
import os
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=80, help="Number of FEM samples to generate")
    parser.add_argument("--p_points", type=int, default=5000, help="Number of points per sample (fixed for DeepONet)")
    parser.add_argument("--t_min", type=float, default=sim.T_MIN, help="Min thickness offset (mm)")
    parser.add_argument("--t_max", type=float, default=sim.T_MAX, help="Max thickness offset (mm)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="beam_dataset_v2.npz", help="Output dataset filename under data/")
    parser.add_argument("--no_onshape", action="store_true", help="Disable Onshape STEP download (use cached STEP only)")
    args = parser.parse_args()

    use_onshape = (not args.no_onshape)
    np.random.seed(args.seed)

    print(f"[System] Starting Data Generation v2 ({args.num_samples} samples)...")
    print(" - Sampling ONLY valid mesh vertices (no 'Air' padding).")
    print(" - Coordinates are canonicalized (L,W,T) and normalized to [-1,1].")
    print(" - Displacement vectors are ALSO canonicalized to (uL,uW,uT).")
    print(f" - t range: [{args.t_min}, {args.t_max}] mm")
    print(f" - points per sample: {args.p_points}")

    t_values = np.linspace(args.t_min, args.t_max, args.num_samples)

    dataset_t = []
    dataset_x = []
    dataset_u = []
    dataset_axes = []
    dataset_mins = []
    dataset_spans = []

    success_count = 0
    fail_count = 0

    for t in tqdm(t_values):
        try:
            # STEP 파일 무결성 체크 (0바이트 파일이면 삭제)
            step_path = os.path.join(sim.MODEL_DIR, f"beam_t{t:.2f}.step")
            if os.path.exists(step_path) and os.path.getsize(step_path) == 0:
                tqdm.write(f" [Warn] Corrupt STEP file found, deleting: {step_path}")
                os.remove(step_path)

            gmsh_inst = sim.generate_mesh(t, use_onshape=use_onshape)
            if gmsh_inst is None:
                raise RuntimeError("Mesh generation failed (Onshape/Gmsh)")

            mesh, basis, u_fem, meta = sim.run_simulation(gmsh_inst)

            # Vertex coordinates (N,3) in xyz meters
            coords_xyz = mesh.p.T
            axes = tuple(meta["axes"])
            aL, aW, aT = axes

            # Canonicalize coords -> (L,W,T) and normalize
            coords_lwt = sim.canonicalize_points(coords_xyz, axes)
            mins_lwt = coords_lwt.min(axis=0)
            spans_lwt = coords_lwt.max(axis=0) - mins_lwt

            coords_01 = sim.normalize_points(coords_lwt, mins_lwt, spans_lwt)
            coords_in = (2.0 * coords_01 - 1.0).astype(np.float32)  # [-1,1]

            # Displacement at vertices (N,3) in xyz (meters)
            M = basis.probes(mesh.p)
            u_flat = M @ u_fem
            n_verts = coords_xyz.shape[0]
            u_xyz = np.vstack([
                u_flat[0:n_verts],
                u_flat[n_verts:2*n_verts],
                u_flat[2*n_verts:],
            ]).T.astype(np.float32)

            # Canonicalize displacement components to (uL,uW,uT)
            u_lwt = u_xyz[:, [aL, aW, aT]]

            # Fixed number of points per sample
            N = n_verts
            P = args.p_points
            replace = N < P
            idx = np.random.choice(N, size=P, replace=replace)

            dataset_t.append(np.float32(t))
            dataset_x.append(coords_in[idx])   # (P,3) canonical normalized coords
            dataset_u.append(u_lwt[idx])       # (P,3) canonical displacement components (meters)

            dataset_axes.append(np.array(axes, dtype=np.int32))
            dataset_mins.append(mins_lwt.astype(np.float32))
            dataset_spans.append(spans_lwt.astype(np.float32))

            success_count += 1

        except Exception as e:
            fail_count += 1
            tqdm.write(f"\n[Error at t={t:.2f}] {e}")
            continue

    print(f"\n[Result] Success: {success_count}, Failed: {fail_count}")
    if success_count == 0:
        print("[CRITICAL] No valid data generated. Check error messages.")
        return

    # Stack arrays
    dataset_t = np.array(dataset_t, dtype=np.float32).reshape(-1, 1)         # (S,1)
    dataset_x = np.array(dataset_x, dtype=np.float32)                        # (S,P,3)
    dataset_u = np.array(dataset_u, dtype=np.float32)                        # (S,P,3)
    dataset_axes = np.array(dataset_axes, dtype=np.int32)                    # (S,3)
    dataset_mins = np.array(dataset_mins, dtype=np.float32)                  # (S,3) (L,W,T mins in meters)
    dataset_spans = np.array(dataset_spans, dtype=np.float32)                # (S,3) (L,W,T spans in meters)

    save_path = os.path.join(sim.DATA_DIR, args.out)
    np.savez(save_path,
             t=dataset_t,
             x=dataset_x,
             u=dataset_u,
             axes=dataset_axes,
             mins_lwt=dataset_mins,
             spans_lwt=dataset_spans)

    print(f"[System] Data saved to {save_path}")
    print(f" - Shapes: t={dataset_t.shape}, x={dataset_x.shape}, u={dataset_u.shape}")
    print(f" - Extra: axes={dataset_axes.shape}, mins_lwt={dataset_mins.shape}, spans_lwt={dataset_spans.shape}")


if __name__ == "__main__":
    main()
