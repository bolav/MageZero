"""
Merge multiple FeatureMap JSON files into one canonical map and remap
the HDF5 index arrays to use the merged indices.

Usage:
  python scripts/merge_feature_maps.py \
    --maps data/features/argentum-v1-a.json data/features/argentum-v1-b.json \
    --hdf5 data/uwtempo/ver1/testing/session_a.hdf5 \
           data/uwtempo/ver1/testing/session_b.hdf5 \
    --out-map data/features/argentum-v1.json

The HDF5 files are remapped IN PLACE.
"""
import argparse
import json
import numpy as np
import h5py
from pathlib import Path


def merge_maps(map_paths: list[str]) -> dict[str, int]:
    """
    Merge feature maps from multiple files into a single canonical map.
    Existing indices from the first map that appears are preserved where possible.
    All string→int assignments are stable: the same string always gets the same
    final index regardless of which worker discovered it first.
    """
    # Collect all feature strings across all maps
    all_features: set[str] = set()
    maps = []
    for path in map_paths:
        with open(path) as f:
            m = json.load(f)
        maps.append(m)
        all_features.update(m.keys())

    # Build merged map: sort deterministically so all future runs agree
    sorted_features = sorted(all_features)
    merged = {feat: idx for idx, feat in enumerate(sorted_features)}
    return merged, maps


def remap_hdf5(hdf5_path: str, old_map: dict[str, int], new_map: dict[str, int]) -> None:
    """Remap /indices in an HDF5 file from old_map indices to new_map indices."""
    # Build old_index → new_index lookup
    reverse_old = {v: k for k, v in old_map.items()}
    remap = {}
    for old_idx, feat in reverse_old.items():
        new_idx = new_map.get(feat)
        if new_idx is not None:
            remap[old_idx] = new_idx

    with h5py.File(hdf5_path, "r+") as f:
        indices = f["/indices"][:]
        remapped = np.vectorize(lambda x: remap.get(int(x), int(x)))(indices)
        f["/indices"][:] = remapped.astype(np.int32)
    print(f"  remapped {hdf5_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--maps",    nargs="+", required=True, help="Input FeatureMap JSON files")
    p.add_argument("--hdf5",    nargs="+", required=True, help="HDF5 files to remap (in place)")
    p.add_argument("--out-map", required=True,            help="Output merged FeatureMap JSON")
    args = p.parse_args()

    print(f"Merging {len(args.maps)} feature maps...")
    merged, old_maps = merge_maps(args.maps)
    print(f"  merged size: {len(merged)} features")

    Path(args.out_map).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_map, "w") as f:
        json.dump(merged, f, separators=(",", ":"))
    print(f"  saved → {args.out_map}")

    print(f"\nRemapping {len(args.hdf5)} HDF5 files...")
    for hdf5_path, old_map in zip(args.hdf5, old_maps):
        remap_hdf5(hdf5_path, old_map, merged)

    print("\nDone. Use the merged map for all future collect runs.")


if __name__ == "__main__":
    main()
