"""
agent_service.py — Flask service that bridges the game-server and the MageZero
inference server, enabling the MageZeroAiPlayerController (Kotlin) to request
AI decisions.

Usage:
  python -m magezero.argentum.agent_service \
    --port 5005 \
    --server-url http://127.0.0.1:50052 \
    --feature-map data/features/argentum-v1.json

Endpoint:
  POST /decide
    Request:  { state: ClientGameState, legalActions: [...], pendingDecision: {...}|null, playerId: str }
    Response: { actionId: str } | { error: str }
"""
import argparse
import http.client
import sys
import urllib.request

import msgpack
from flask import Flask, request, jsonify

from .client_encoder import ClientStateEncoder, _eid
from .action_encoder import encode_actions, priority_slot, binary_slot, target_slot
from .feature_map import FeatureMap

app = Flask(__name__)

# Module-level state set by init()
_encoder: ClientStateEncoder | None = None
_server_url: str | None = None


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def _evaluate(indices: list[int]) -> dict:
    payload = msgpack.packb({"indices": indices, "offsets": [0]}, use_bin_type=True)
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                _server_url.rstrip("/") + "/evaluate",
                data=payload,
                headers={
                    "Content-Type": "application/x-msgpack",
                    "Connection": "close",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return msgpack.unpackb(resp.read(), raw=False)
        except http.client.BadStatusLine:
            if attempt == 2:
                raise
    raise RuntimeError("inference server unreachable after 3 attempts")


def _choose_action(net_out: dict, encoded_acts: list[dict],
                   is_player: bool) -> dict:
    """Pick the action with the highest policy logit for its head."""
    best = None
    best_score = float("-inf")

    for ea in encoded_acts:
        head  = ea["head"]
        slot  = ea["slot"]
        if head == "priority":
            logits = net_out["policy_player"] if is_player else net_out["policy_opponent"]
        elif head == "target":
            logits = net_out["policy_target"]
        else:
            logits = net_out["policy_binary"]

        score = logits[slot] if slot < len(logits) else float("-inf")
        if score > best_score:
            best_score = score
            best = ea

    return best


def _fallback_action(legal_actions: list[dict]) -> dict | None:
    """Return first affordable non-mana non-pass action, or pass, or None."""
    non_pass = [a for a in legal_actions
                if a.get("affordable", True)
                and not a.get("isManaAbility", False)
                and a.get("kind") != "PassPriority"]
    if non_pass:
        return non_pass[0]
    pass_action = next((a for a in legal_actions if a.get("kind") == "PassPriority"), None)
    return pass_action or (legal_actions[0] if legal_actions else None)


# --------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------

@app.post("/decide")
def decide():
    body = request.get_json(force=True)
    state            = body.get("state", {})
    legal_actions    = body.get("legalActions", [])
    pending_decision = body.get("pendingDecision")
    player_id        = body.get("playerId", "")

    if not legal_actions:
        return jsonify({"error": "no legal actions"}), 400

    # Determine perspective
    perspective_id = _eid(state.get("viewingPlayerId"))
    priority_id    = _eid(state.get("priorityPlayerId"))
    is_player      = (perspective_id == priority_id == player_id) or (priority_id == player_id)

    # Encode legal actions
    decision_kind = pending_decision.get("type") if pending_decision else None
    encoded_acts  = encode_actions(legal_actions, decision_kind, state)

    if not encoded_acts:
        fallback = _fallback_action(legal_actions)
        if fallback:
            return jsonify({"actionId": fallback["actionId"]})
        return jsonify({"error": "no affordable actions"}), 400

    # Single legal action — no need to call the network
    if len(encoded_acts) == 1:
        return jsonify({"actionId": encoded_acts[0]["actionId"]})

    # Encode state and call inference server
    try:
        indices  = sorted(_encoder.encode(state, pending_decision))
        net_out  = _evaluate(indices)
        chosen   = _choose_action(net_out, encoded_acts, is_player)
        if chosen:
            app.logger.info(
                "decided: %s (score=%.3f, head=%s, slot=%d)",
                chosen.get("description", chosen["actionId"]),
                float(net_out["value"]),
                chosen["head"],
                chosen["slot"],
            )
            return jsonify({"actionId": chosen["actionId"]})
    except Exception as e:
        app.logger.warning("inference failed: %s — using fallback", e)

    # Fallback
    fallback = _fallback_action(legal_actions)
    if fallback:
        return jsonify({"actionId": fallback["actionId"]})
    return jsonify({"error": "could not decide"}), 500


@app.get("/healthz")
def healthz():
    return "ok", 200


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def init(port: int, server_url: str, feature_map_path: str) -> None:
    global _encoder, _server_url
    _server_url = server_url
    fm = FeatureMap(feature_map_path)
    _encoder = ClientStateEncoder(fm)
    print(f"[agent] feature_map={feature_map_path} ({fm.size()} features)")
    print(f"[agent] inference_server={server_url}")
    print(f"[agent] listening on http://127.0.0.1:{port}")

    import waitress
    waitress.serve(app, host="127.0.0.1", port=port, threads=4)


def main() -> None:
    p = argparse.ArgumentParser(description="MageZero agent service")
    p.add_argument("--port",         type=int, default=5005)
    p.add_argument("--server-url",   default="http://127.0.0.1:50052")
    p.add_argument("--feature-map",  required=True)
    args = p.parse_args()
    init(args.port, args.server_url, args.feature_map)


if __name__ == "__main__":
    main()
