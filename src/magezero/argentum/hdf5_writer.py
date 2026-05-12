"""
hdf5_writer.py — writes self-play data to MageZero-compatible HDF5 files.

HDF5 schema (must match dataset.py exactly):
  /indices  int32  [nnz]     concatenated sparse feature indices
  /offsets  int64  [N+1]     CSR pointer: state i uses indices[offsets[i]:offsets[i+1]]
  /row      float32 [N, 132] columns: policy[128] | result | stateScore | isPlayer | actionType

row column layout:
  row[:, 0:128]  policy label — MCTS visit counts (raw; train.py normalises)
  row[:, 128]    result label — +1.0 win / -1.0 loss / 0.0 draw (acting player POV)
  row[:, 129]    stateScore   — unused, always 0.0
  row[:, 130]    isPlayer     — 1.0 if perspective == acting player, else 0.0
  row[:, 131]    actionType   — int cast to float (0=PRIORITY, 3=TARGET, 5=BINARY)
"""
import numpy as np
import h5py
from pathlib import Path

ACTIONS_MAX = 128   # must match model.py
ROW_WIDTH   = ACTIONS_MAX + 4


class HDF5Writer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Per-game buffer — cleared after end_game()
        self._game: list[tuple[np.ndarray, np.ndarray]] = []
        # Flush buffer — cleared after flush()
        self._pending_indices: list[np.ndarray] = []
        self._pending_rows:    list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Game-level API
    # ------------------------------------------------------------------

    def begin_game(self) -> None:
        self._game = []

    def record_step(
        self,
        feature_indices: list[int],
        policy_vec: list[float],     # length ACTIONS_MAX; slot positions filled with visit counts
        is_player: bool,
        action_type: int,
    ) -> None:
        """Buffer one decision step. outcome is back-patched in end_game()."""
        idx = np.array(sorted(set(feature_indices)), dtype=np.int32)
        row = np.zeros(ROW_WIDTH, dtype=np.float32)
        row[:ACTIONS_MAX] = policy_vec
        # row[128] = result  ← filled in end_game()
        row[129] = 0.0       # stateScore (unused)
        row[130] = 1.0 if is_player else 0.0
        row[131] = float(action_type)
        self._game.append((idx, row))

    def end_game(self, outcome: float) -> None:
        """
        Back-patch the game outcome onto every buffered step and move to
        the flush buffer.

        outcome: +1.0 = win for acting player, -1.0 = loss, 0.0 = draw/truncation.
        The caller is responsible for computing this from the perspective of the
        player who was acting at that step.
        """
        for idx, row in self._game:
            row[128] = outcome
            self._pending_indices.append(idx)
            self._pending_rows.append(row)
        self._game = []

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write all buffered steps to the HDF5 file and clear the buffer."""
        if not self._pending_indices:
            return

        all_indices = np.concatenate(self._pending_indices).astype(np.int32)
        all_rows    = np.stack(self._pending_rows, axis=0).astype(np.float32)

        # Build offsets array [N+1]
        lengths  = np.array([len(x) for x in self._pending_indices], dtype=np.int64)
        offsets  = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        with h5py.File(self.path, "a") as f:
            if "/indices" not in f:
                # First write — create datasets with unlimited dimension
                f.create_dataset("/indices", data=all_indices,
                                 maxshape=(None,), chunks=(min(len(all_indices), 65536),),
                                 compression="gzip", compression_opts=1)
                f.create_dataset("/offsets", data=offsets,
                                 maxshape=(None,), chunks=(min(len(offsets), 65536),))
                f.create_dataset("/row",     data=all_rows,
                                 maxshape=(None, ROW_WIDTH),
                                 chunks=(min(len(all_rows), 512), ROW_WIDTH),
                                 compression="gzip", compression_opts=1)
            else:
                # Append — adjust offsets relative to existing index count
                old_idx_end = int(f["/offsets"][-1])
                old_n       = f["/row"].shape[0]

                f["/indices"].resize(f["/indices"].shape[0] + len(all_indices), axis=0)
                f["/indices"][-len(all_indices):] = all_indices

                # offsets[0] is 0 for the new batch; shift and drop it
                shifted = offsets[1:] + old_idx_end
                f["/offsets"].resize(f["/offsets"].shape[0] + len(shifted), axis=0)
                f["/offsets"][-len(shifted):] = shifted

                f["/row"].resize(old_n + len(all_rows), axis=0)
                f["/row"][-len(all_rows):] = all_rows

        self._pending_indices = []
        self._pending_rows    = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.flush()
