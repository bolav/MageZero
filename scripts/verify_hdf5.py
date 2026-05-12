"""
Verify that a MageZero HDF5 file has the correct schema and sane content.

Usage:
  python scripts/verify_hdf5.py data/test/ver1/testing/session1.hdf5
"""
import sys
import h5py
import numpy as np

ACTIONS_MAX = 128
ROW_WIDTH   = ACTIONS_MAX + 4


def verify(path: str) -> None:
    print(f"Verifying: {path}\n")
    with h5py.File(path, "r") as f:
        # --- schema ---
        missing = [k for k in ("/indices", "/offsets", "/row") if k not in f]
        if missing:
            print(f"FAIL: missing datasets: {missing}")
            sys.exit(1)

        indices = f["/indices"][:]
        offsets = f["/offsets"][:]
        row     = f["/row"][:]
        N = row.shape[0]

        print(f"Steps (N):      {N}")
        print(f"indices shape:  {indices.shape}  dtype={indices.dtype}")
        print(f"offsets shape:  {offsets.shape}  dtype={offsets.dtype}")
        print(f"row shape:      {row.shape}      dtype={row.dtype}")
        print()

        # --- row width ---
        if row.shape[1] != ROW_WIDTH:
            print(f"FAIL: row width {row.shape[1]}, expected {ROW_WIDTH}")
            sys.exit(1)

        # --- offsets consistency ---
        # CSR format: offsets has N+1 elements (matches dataset.py: N = off.shape[0] - 1)
        if len(offsets) != N + 1:
            print(f"FAIL: offsets length {len(offsets)}, expected N+1={N+1}")
            sys.exit(1)
        if offsets[-1] != len(indices):
            print(f"FAIL: offsets[-1]={offsets[-1]} != len(indices)={len(indices)}")
            sys.exit(1)

        # --- indices in range ---
        idx_min, idx_max = int(indices.min()), int(indices.max())
        print(f"Index range:    [{idx_min}, {idx_max}]")
        if idx_min < 0 or idx_max >= 2_000_000:
            print("FAIL: indices out of [0, 2_000_000) range")
            sys.exit(1)

        # --- policy labels ---
        policy = row[:, :ACTIONS_MAX]
        nonzero_per_step = (policy > 0).sum(axis=1)
        print(f"Policy nonzero per step: min={nonzero_per_step.min()}, "
              f"max={nonzero_per_step.max()}, mean={nonzero_per_step.mean():.1f}")

        # --- outcomes ---
        outcomes = row[:, 128]
        print(f"Outcomes:       {dict(zip(*np.unique(outcomes, return_counts=True)))}")
        bad_outcomes = ~np.isin(outcomes, [-1.0, 0.0, 1.0])
        if bad_outcomes.any():
            print(f"FAIL: unexpected outcome values: {outcomes[bad_outcomes]}")
            sys.exit(1)

        # --- isPlayer ---
        is_player = row[:, 130]
        print(f"isPlayer:       {dict(zip(*np.unique(is_player, return_counts=True)))}")

        # --- actionType ---
        action_types = row[:, 131]
        at_counts = dict(zip(*np.unique(action_types, return_counts=True)))
        at_names  = {0.0: "PRIORITY", 3.0: "CHOOSE_TARGET", 5.0: "CHOOSE_USE"}
        print(f"actionType:     { {at_names.get(k, k): v for k, v in at_counts.items()} }")

        print()
        print("OK — schema valid")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/test/ver1/testing/session1.hdf5"
    verify(path)
