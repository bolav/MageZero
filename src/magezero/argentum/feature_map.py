"""
feature_map.py — stable string → integer index mapping.

One FeatureMap per featurizer version, shared across all self-play runs and
model checkpoints. Indices are assigned in insertion order and never change.
The map only grows; deletions and reassignments are not permitted.

Thread-safe: concurrent calls to id() are safe. save() should be called from
one thread at a time (the collector flushes after each game batch).
"""
import json
import threading
from pathlib import Path

GLOBAL_MAX = 2_000_000  # must match model.py


class FeatureMap:
    def __init__(self, path: str | Path | None = None):
        self._map: dict[str, int] = {}
        self._lock = threading.Lock()
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def id(self, feature: str) -> int:
        """Return the index for *feature*, assigning a new one if unseen."""
        try:
            return self._map[feature]
        except KeyError:
            with self._lock:
                if feature not in self._map:
                    n = len(self._map)
                    if n >= GLOBAL_MAX:
                        raise OverflowError(
                            f"FeatureMap full ({GLOBAL_MAX} limit). "
                            "Increase GLOBAL_MAX or reduce feature vocabulary."
                        )
                    self._map[feature] = n
                return self._map[feature]

    def size(self) -> int:
        return len(self._map)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._map, f, separators=(",", ":"))
        tmp.replace(self.path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with open(self.path) as f:
            self._map = json.load(f)
