"""
collect.py — CLI entry point for Argentum self-play data collection.

Replaces the XMage JVM launch in MageZero's runner.py. Plays N games via the
Argentum gym-server and writes MageZero-compatible HDF5 training data.

Usage:
  python -m magezero.argentum.collect \\
    --games 200 \\
    --deck mono-red \\
    --version 1 \\
    --offline \\
    --gym-url http://localhost:8090 \\
    --output data/mono-red/ver1/testing/session1.hdf5 \\
    --feature-map data/features/argentum-v1.json

  # With inference server (generation 1+):
  python -m magezero.argentum.collect \\
    --games 200 \\
    --deck mono-red \\
    --version 1 \\
    --gym-url http://localhost:8090 \\
    --server-url http://localhost:50052 \\
    --output data/mono-red/ver1/testing/session2.hdf5 \\
    --feature-map data/features/argentum-v1.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

from .client import ArgentumClient, _env_id
from .encoder import ArgentumStateEncoder, _player_id
from .action_encoder import encode_actions, ACTION_TYPE_PRIORITY
from .feature_map import FeatureMap
from .hdf5_writer import HDF5Writer
from .mcts import PythonMcts


# --------------------------------------------------------------------------
# Deck config helpers
# --------------------------------------------------------------------------

def _load_deck_config(deck_json: str | None, gym_url: str) -> dict:
    """
    Load deck config from a JSON file or return a hardcoded fallback.
    The JSON file should contain an EnvConfig body for POST /envs.
    """
    if deck_json:
        with open(deck_json) as f:
            return json.load(f)
    # Default: two identical mono-red decks for quick smoke tests
    return {
        "players": [
            {
                "name": "P1",
                "deck": {
                    "type": "Explicit",
                    "cards": {"Mountain": 17, "Raging Goblin": 3}
                }
            },
            {
                "name": "P2",
                "deck": {
                    "type": "Explicit",
                    "cards": {"Mountain": 17, "Raging Goblin": 3}
                }
            }
        ],
        "skipMulligans": True,
    }


# --------------------------------------------------------------------------
# Single-game loop
# --------------------------------------------------------------------------

MAX_STEPS_PER_GAME = 5000  # safety cap — server enforces maxTurns; this only catches runaway bugs


def play_game(
    client:           ArgentumClient,
    encoder:          ArgentumStateEncoder,
    mcts:             PythonMcts,
    writer:           HDF5Writer,
    config:           dict,
    game_num:         int,
    heuristic_opponent: bool = False,
) -> dict:
    env_id, obs = client.create_env(config)
    writer.begin_game()

    perspective_id = _player_id(obs.get("perspectivePlayerId"))
    steps = 0

    while not obs.get("terminated", False) and steps < MAX_STEPS_PER_GAME:
        agent_id = _player_id(obs.get("agentToAct"))
        is_player = (agent_id == perspective_id)

        # Opponent's turn: use engine heuristic (fast, no MCTS).
        # Do this before the legalActions check — heuristic handles structured
        # decisions too, so legalActions may be empty on the opponent's turn.
        if heuristic_opponent and not is_player:
            result = client.heuristic_step(env_id)
            obs = result.get("nextObservation", result)
            steps += 1
            continue

        legal = obs.get("legalActions", [])
        if not legal:
            break

        pd            = obs.get("pendingDecision")
        decision_kind = pd["kind"] if pd else None

        # Encode state before stepping
        indices = sorted(encoder.encode(obs))

        # Select action via MCTS
        action_id, visit_vec, action_type = mcts.select_action(env_id, obs)

        # Record step before advancing
        writer.record_step(
            feature_indices = indices,
            policy_vec      = visit_vec,
            is_player       = is_player,
            action_type     = action_type,
        )

        obs = client.step(env_id, action_id)
        steps += 1

    # Determine outcome from perspective player's point of view
    winner_raw = obs.get("winnerId")
    winner_id  = _player_id(winner_raw) if winner_raw else None
    if winner_id == perspective_id:
        outcome = 1.0
    elif winner_id is None:
        outcome = 0.0   # draw / truncation
    else:
        outcome = -1.0

    writer.end_game(outcome)
    client.dispose([env_id])

    return {"steps": steps, "outcome": outcome, "winner": winner_id}


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def run(
    games:       int,
    config:      dict,
    output_path: str,
    feature_map_path: str,
    gym_url:     str,
    server_url:  str | None,
    simulations: int,
    flush_every: int,
    heuristic_opponent: bool = False,
) -> None:
    fm      = FeatureMap(feature_map_path)
    client  = ArgentumClient(gym_url)
    encoder = ArgentumStateEncoder(fm)
    offline = server_url is None
    mcts    = PythonMcts(
        client           = client,
        encoder          = encoder,
        simulations      = simulations,
        offline          = offline,
        server_url       = server_url,
        dirichlet_alpha  = 0.3,
        dirichlet_weight = 0.0 if offline else 0.25,
    )

    print(f"[collect] gym={gym_url}  server={'offline' if offline else server_url}")
    print(f"[collect] games={games}  sims={simulations}  output={output_path}")

    # Verify gym-server is reachable
    try:
        client.observe  # lazy — just do a quick create+dispose
        env_id, _ = client.create_env(config)
        client.dispose([env_id])
        print("[collect] gym-server OK")
    except Exception as e:
        print(f"[collect] ERROR: cannot reach gym-server at {gym_url}: {e}", file=sys.stderr)
        sys.exit(1)

    wins = losses = draws = 0
    t0 = time.time()
    t_game = time.time()

    with HDF5Writer(output_path) as writer:
        for i in range(games):
            result = play_game(client, encoder, mcts, writer, config, i,
                               heuristic_opponent=heuristic_opponent)
            outcome = result["outcome"]
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                draws += 1

            elapsed_game = time.time() - t_game
            t_game = time.time()
            elapsed_total = time.time() - t0
            rate = (i + 1) / elapsed_total
            outcome_sym = "W" if outcome > 0 else ("L" if outcome < 0 else "D")
            print(
                f"[{i+1}/{games}] {outcome_sym}  steps={result['steps']}  "
                f"W={wins} L={losses} D={draws}  "
                f"features={fm.size()}  "
                f"{elapsed_game:.0f}s/game  "
                f"eta={((games - i - 1) / rate):.0f}s",
                flush=True,
            )

            if (i + 1) % flush_every == 0:
                writer.flush()
                fm.save()

    fm.save()
    elapsed = time.time() - t0
    print(
        f"[collect] done  games={games}  "
        f"W={wins} L={losses} D={draws}  "
        f"features={fm.size()}  "
        f"wall={elapsed:.0f}s"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect Argentum self-play data for MageZero training."
    )
    p.add_argument("--games",        type=int,  default=200,
                   help="Number of games to play")
    p.add_argument("--simulations",  type=int,  default=100,
                   help="MCTS simulations per move")
    p.add_argument("--gym-url",      default="http://localhost:8090",
                   help="Argentum gym-server base URL")
    p.add_argument("--server-url",   default=None,
                   help="MageZero inference server URL (omit for offline mode)")
    p.add_argument("--output",       required=True,
                   help="Output HDF5 file path")
    p.add_argument("--feature-map",  required=True,
                   help="FeatureMap JSON file path (created if missing)")
    p.add_argument("--deck-config",  default=None,
                   help="JSON file with EnvConfig body for POST /envs")
    p.add_argument("--offline",             action="store_true",
                   help="Use uniform priors (no inference server) — generation 0 bootstrap")
    p.add_argument("--heuristic-opponent",  action="store_true",
                   help="Drive opponent with engine heuristic AI instead of MCTS (faster)")
    p.add_argument("--flush-every",  type=int,  default=10,
                   help="Flush HDF5 and save FeatureMap every N games")
    args = p.parse_args()

    config = _load_deck_config(args.deck_config, args.gym_url)

    server_url = None if args.offline else args.server_url

    run(
        games               = args.games,
        config              = config,
        output_path         = args.output,
        feature_map_path    = args.feature_map,
        gym_url             = args.gym_url,
        server_url          = server_url,
        simulations         = args.simulations,
        flush_every         = args.flush_every,
        heuristic_opponent  = args.heuristic_opponent,
    )


if __name__ == "__main__":
    main()
