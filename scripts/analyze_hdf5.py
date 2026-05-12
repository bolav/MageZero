"""
Assess the quality of MageZero training data in an HDF5 file.

Usage:
  python scripts/analyze_hdf5.py data/uwtempo/ver1/training/session1.hdf5
  python scripts/analyze_hdf5.py data/uwtempo/ver1/training/*.hdf5   # compare files
"""
import sys
import math
import numpy as np
import h5py
from pathlib import Path

ACTIONS_MAX = 128


def analyze(path: str) -> dict:
    with h5py.File(path, "r") as f:
        indices = f["/indices"][:]
        offsets = f["/offsets"][:]
        row     = f["/row"][:]

    N = row.shape[0]
    if N == 0:
        return {"path": path, "error": "empty file"}

    policy      = row[:, :ACTIONS_MAX]       # [N, 128]
    outcomes    = row[:, 128]                # [N]
    is_player   = row[:, 130]               # [N]
    action_type = row[:, 131]               # [N]

    # ------------------------------------------------------------------
    # Outcome distribution — too many draws = games hit maxTurns, not
    # terminating naturally. A good dataset has real wins and losses.
    # ------------------------------------------------------------------
    wins   = int((outcomes >  0.5).sum())
    losses = int((outcomes < -0.5).sum())
    draws  = int((np.abs(outcomes) < 0.5).sum())
    total  = N

    # ------------------------------------------------------------------
    # Policy quality — the most important signal.
    #
    # Nonzero slots per step: how many actions did MCTS actually visit?
    #   - 1 (one-hot): MCTS put all visits on one action — trivial policy,
    #     almost no learning signal for the network. Happens with very few
    #     simulations or uniform priors (gen 0 offline).
    #   - 2-8: MCTS explored a handful of alternatives — some signal.
    #   - 8+: MCTS spread visits across many actions — rich signal.
    #
    # Policy entropy: H = -sum(p * log2(p)) in bits.
    #   - 0.0: one-hot, no information.
    #   - 7.0: uniform over 128 actions, maximum entropy.
    #   - Good training data: 0.5 – 3.0 bits per step.
    # ------------------------------------------------------------------
    nonzero_per_step = (policy > 0).sum(axis=1)       # [N]
    one_hot_frac     = float((nonzero_per_step == 1).mean())

    # Entropy (only over nonzero entries to avoid log(0))
    eps = 1e-9
    p_norm = policy / (policy.sum(axis=1, keepdims=True) + eps)
    entropy_per_step = -(p_norm * np.log2(p_norm + eps) * (p_norm > eps)).sum(axis=1)
    avg_entropy      = float(entropy_per_step.mean())
    median_entropy   = float(np.median(entropy_per_step))

    # ------------------------------------------------------------------
    # Feature diversity — unique feature indices seen.
    # More unique features = more diverse game states.
    # Gen 0 (10 games): ~500 features.  Gen 1+ (200 games): 2000–5000+.
    # ------------------------------------------------------------------
    unique_features = int(np.unique(indices).shape[0])
    avg_features_per_state = float(np.diff(np.concatenate([[0], offsets])).mean()
                                   if len(offsets) > 0 else 0)

    # ------------------------------------------------------------------
    # Action diversity — how many distinct slots were ever visited.
    # A diverse dataset visits many different slots, not always the same ones.
    # ------------------------------------------------------------------
    visited_slots   = int((policy > 0).any(axis=0).sum())

    # ------------------------------------------------------------------
    # Game-level stats — infer approximate game count and length.
    # Detect boundaries by finding steps where the outcome value changes
    # OR where the feature index resets (new game starts low again).
    # Falls back to outcome-change detection which works poorly for all-draws.
    # ------------------------------------------------------------------
    # Use contiguous runs of identical outcome values
    boundaries = [0]
    for i in range(1, N):
        if outcomes[i] != outcomes[i - 1]:
            boundaries.append(i)
    boundaries.append(N)
    run_lengths = np.diff(boundaries)
    # Filter out very short runs (< 5 steps) as noise
    game_runs   = run_lengths[run_lengths >= 5]
    approx_games = max(len(game_runs), 1)
    avg_steps    = float(run_lengths.mean()) if len(run_lengths) > 0 else N

    return {
        "path":             path,
        "steps":            N,
        "approx_games":     approx_games,
        "avg_steps_game":   avg_steps,
        # Outcomes
        "wins":             wins,
        "losses":           losses,
        "draws":            draws,
        "draw_rate":        draws / total,
        # Policy quality
        "one_hot_frac":     one_hot_frac,
        "avg_nonzero":      float(nonzero_per_step.mean()),
        "avg_entropy_bits": avg_entropy,
        "median_entropy":   median_entropy,
        # Feature diversity
        "unique_features":  unique_features,
        "avg_features_state": avg_features_per_state,
        "visited_slots":    visited_slots,
    }


def grade(stats: dict) -> list[str]:
    """Return a list of warnings/notes about data quality."""
    notes = []

    if stats.get("error"):
        return [f"ERROR: {stats['error']}"]

    draw_rate = stats["draw_rate"]
    one_hot   = stats["one_hot_frac"]
    entropy   = stats["avg_entropy_bits"]
    features  = stats["unique_features"]
    steps     = stats["steps"]

    if steps < 5000:
        notes.append("⚠  Very few steps — collect more games before training.")

    if draw_rate > 0.6:
        notes.append(f"⚠  {draw_rate:.0%} draws — most games hit maxTurns. "
                     "Data is less informative; increase maxTurns or reduce deck complexity.")

    if one_hot > 0.8:
        notes.append(f"⚠  {one_hot:.0%} one-hot policies — MCTS rarely explored alternatives. "
                     "Increase simulations or use a trained network (not offline mode).")
    elif one_hot > 0.5:
        notes.append(f"ℹ  {one_hot:.0%} one-hot policies — marginal. "
                     "More simulations or a stronger network will help.")

    if entropy < 0.3:
        notes.append("⚠  Very low policy entropy — nearly all policies are trivial. "
                     "Network will struggle to learn meaningful move ordering.")
    elif entropy < 1.0:
        notes.append("ℹ  Low policy entropy — data from offline/gen-0 is expected to be low. "
                     "Should improve with trained network priors.")

    if features < 500:
        notes.append("⚠  Very few unique features — only a handful of game states seen. "
                     "Collect many more games.")

    if not notes:
        notes.append("✓  Data looks reasonable.")

    return notes


def print_report(stats: dict) -> None:
    path = Path(stats["path"]).name
    print(f"\n{'═' * 60}")
    print(f"  {path}")
    print(f"{'═' * 60}")

    if stats.get("error"):
        print(f"  ERROR: {stats['error']}")
        return

    print(f"  Steps:          {stats['steps']:,}  (~{stats['approx_games']} games, "
          f"avg {stats['avg_steps_game']:.0f} steps/game)")
    print(f"  Outcomes:       W={stats['wins']}  L={stats['losses']}  D={stats['draws']}  "
          f"(draw rate: {stats['draw_rate']:.0%})")
    print()
    print(f"  Policy quality:")
    print(f"    One-hot frac:   {stats['one_hot_frac']:.1%}   "
          f"(lower = richer MCTS exploration)")
    print(f"    Avg nonzero:    {stats['avg_nonzero']:.1f} slots/step")
    print(f"    Avg entropy:    {stats['avg_entropy_bits']:.2f} bits/step  "
          f"(0=trivial, 7=uniform over 128)")
    print(f"    Median entropy: {stats['median_entropy']:.2f} bits/step")
    print()
    print(f"  Feature diversity:")
    print(f"    Unique features:    {stats['unique_features']:,}")
    print(f"    Avg features/state: {stats['avg_features_state']:.0f}")
    print(f"    Slots ever visited: {stats['visited_slots']} / {ACTIONS_MAX}")
    print()
    print("  Assessment:")
    for note in grade(stats):
        print(f"    {note}")


def main() -> None:
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["data/uwtempo/ver1/training/session1.hdf5"]
    all_stats = [analyze(p) for p in paths]

    for s in all_stats:
        print_report(s)

    if len(all_stats) > 1:
        print(f"\n{'═' * 60}")
        print("  COMPARISON")
        print(f"{'═' * 60}")
        print(f"  {'File':<30} {'Steps':>7} {'Entropy':>8} {'1-hot%':>7} {'Features':>9}")
        print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*7} {'-'*9}")
        for s in all_stats:
            if s.get("error"):
                continue
            name = Path(s["path"]).name[:30]
            print(f"  {name:<30} {s['steps']:>7,} "
                  f"{s['avg_entropy_bits']:>7.2f}b "
                  f"{s['one_hot_frac']:>6.0%}  "
                  f"{s['unique_features']:>9,}")


if __name__ == "__main__":
    main()
