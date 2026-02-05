from __future__ import annotations

import argparse
import os
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=os.path.join("data", "beam_dataset_pi.npz"))
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.dataset):
        print(f"File not found: {args.dataset}")
        return

    data = np.load(args.dataset)
    print(f"File: {args.dataset}")
    print(f"Keys: {data.files}")
    for k in data.files:
        arr = data[k]
        if hasattr(arr, "shape"):
            print(f" - {k}: shape={arr.shape} dtype={arr.dtype}")
        else:
            print(f" - {k}: {type(arr)}")
    if "t" in data:
        print(f"t range: [{data['t'].min():.3f}, {data['t'].max():.3f}]")
    if "spans" in data:
        spans = data["spans"]
        print(f"mean spans (m): {spans.mean(axis=0)}")


if __name__ == "__main__":
    main()
