import numpy as np
import os
import argparse

import sim_core as sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="beam_dataset_v2.npz", help="Dataset filename under data/")
    args = parser.parse_args()

    data_path = os.path.join(sim.DATA_DIR, args.dataset)
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    data = np.load(data_path)
    print(f"File found: {data_path}")
    print(f"Keys: {data.files}")

    t = data["t"]
    x = data["x"]
    u = data["u"]

    print(f"t: shape={t.shape}, dtype={t.dtype}, min={t.min():.3f}, max={t.max():.3f} (mm offset)")
    print(f"x: shape={x.shape}, dtype={x.dtype} (canonical coords in [-1,1])")
    print(f"u: shape={u.shape}, dtype={u.dtype} (canonical disp in meters)")

    # quick stats
    umag = np.linalg.norm(u, axis=2)
    print(f"|u|: min={umag.min():.3e} m, mean={umag.mean():.3e} m, max={umag.max():.3e} m")

    if "axes" in data.files:
        axes = data["axes"]
        print(f"axes: shape={axes.shape}, unique rows={np.unique(axes, axis=0).shape[0]}")
        print(f"axes unique: {np.unique(axes, axis=0)}")

    if "mins_lwt" in data.files and "spans_lwt" in data.files:
        spans = data["spans_lwt"]
        print(f"spans_lwt: mean={spans.mean(axis=0)}, min={spans.min(axis=0)}, max={spans.max(axis=0)} (meters)")


if __name__ == "__main__":
    main()
