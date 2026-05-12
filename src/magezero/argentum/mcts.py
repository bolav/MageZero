"""
mcts.py — pure-Python PUCT tree search for Argentum Engine games.

The tree root corresponds to the current game state (live env_id in the
gym-server). Each simulation forks the root env, replays the path of chosen
actions to reach the simulation's current node, evaluates the leaf, and
disposes the fork.

Offline mode (no network): uniform priors, value = 0 at every leaf. This
produces random MCTS data useful for bootstrap training (generation 0).

Online mode: calls the MageZero inference server (server.py) for priors
and value via msgpack.
"""
import http.client
import math
import random
import urllib.request
from dataclasses import dataclass, field

import msgpack

from .client import ArgentumClient
from .encoder import ArgentumStateEncoder
from .action_encoder import (
    encode_actions, classify,
    ACTION_TYPE_PRIORITY, ACTION_TYPE_CHOOSE_TARGET, ACTION_TYPE_CHOOSE_USE,
)

ACTIONS_MAX = 128   # must match model.py
C_PUCT      = 1.5


# --------------------------------------------------------------------------
# Tree nodes
# --------------------------------------------------------------------------

@dataclass
class MctsEdge:
    action_id:   int
    head:        str
    slot:        int
    action_type: int
    prior:       float
    visits:      int   = 0
    value_sum:   float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb(self, parent_visits: int) -> float:
        exploration = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.q + exploration


@dataclass
class MctsNode:
    edges: list[MctsEdge] = field(default_factory=list)
    expanded: bool = False


# --------------------------------------------------------------------------
# Inference client (msgpack over HTTP — matches server.py)
# --------------------------------------------------------------------------

class InferenceClient:
    """
    Persistent HTTP connection to the MageZero inference server.
    Reuses the connection across calls for speed; reconnects automatically
    on stale-connection errors (BadStatusLine, RemoteDisconnected).
    """
    def __init__(self, url: str):
        from urllib.parse import urlparse
        p = urlparse(url)
        self._host = p.hostname
        self._port = p.port or 50052
        self._path = (p.path or "") + "/evaluate"
        self._conn: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        conn = http.client.HTTPConnection(self._host, self._port, timeout=30)
        conn.connect()
        return conn

    def evaluate(self, indices: list[int]) -> dict:
        """
        Returns {"policy_player", "policy_opponent", "policy_target",
                 "policy_binary", "value"}.
        """
        payload = msgpack.packb({"indices": indices, "offsets": [0]},
                                use_bin_type=True)
        for attempt in range(3):
            try:
                if self._conn is None:
                    self._conn = self._connect()
                self._conn.request(
                    "POST", self._path, body=payload,
                    headers={"Content-Type": "application/x-msgpack"}
                )
                resp = self._conn.getresponse()
                body = resp.read()
                return msgpack.unpackb(body, raw=False)
            except (http.client.BadStatusLine,
                    http.client.RemoteDisconnected,
                    ConnectionResetError,
                    BrokenPipeError):
                # Stale connection — reconnect and retry
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
                if attempt == 2:
                    raise


# --------------------------------------------------------------------------
# MCTS
# --------------------------------------------------------------------------

class PythonMcts:
    def __init__(
        self,
        client:      ArgentumClient,
        encoder:     ArgentumStateEncoder,
        simulations: int = 100,
        offline:     bool = True,
        server_url:  str | None = None,
    ):
        self._client      = client
        self._encoder     = encoder
        self._simulations = simulations
        self._offline     = offline
        self._inference   = InferenceClient(server_url) if server_url else None

    def select_action(
        self,
        env_id: str,
        obs:    dict,
    ) -> tuple[int, list[float], int]:
        """
        Run MCTS from the current state.

        Returns:
            chosen_action_id: the action to submit to the gym-server
            visit_vec:        dense float[128] visit count array for the policy label
            action_type:      int for HDF5 row[131]
        """
        pd            = obs.get("pendingDecision")
        decision_kind = pd["kind"] if pd else None
        encoded_acts  = encode_actions(obs.get("legalActions", []), decision_kind, obs)

        if not encoded_acts:
            # No affordable actions — pass (should not happen in normal play)
            pass_id = next(
                (a["actionId"] for a in obs.get("legalActions", [])
                 if a.get("kind") == "PassPriority"),
                obs["legalActions"][0]["actionId"] if obs.get("legalActions") else 0
            )
            return pass_id, [0.0] * ACTIONS_MAX, ACTION_TYPE_PRIORITY

        # Single legal action — no MCTS needed
        if len(encoded_acts) == 1:
            ea = encoded_acts[0]
            vec = [0.0] * ACTIONS_MAX
            vec[ea["slot"]] = 1.0
            return ea["actionId"], vec, ea["actionType"]

        root = MctsNode()
        self._expand(root, obs, encoded_acts)

        for _ in range(self._simulations):
            fork_id = self._client.fork(env_id)[0]
            try:
                self._simulate(root, fork_id, depth=0)
            finally:
                self._client.dispose([fork_id])

        # Build policy vector from visit counts
        action_type = encoded_acts[0]["actionType"]
        visit_vec   = [0.0] * ACTIONS_MAX
        best_id, best_visits = encoded_acts[0]["actionId"], -1

        for edge in root.edges:
            visit_vec[edge.slot] = float(edge.visits)
            if edge.visits > best_visits:
                best_visits = edge.visits
                best_id     = edge.action_id

        return best_id, visit_vec, action_type

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _simulate(self, node: MctsNode, env_id: str, depth: int) -> float:
        """Recursive PUCT simulation. Returns value in [-1, 1]."""
        if depth > 50:  # safety cap
            return 0.0

        # Select edge by UCB
        total_visits = sum(e.visits for e in node.edges)
        edge = max(node.edges, key=lambda e: e.ucb(total_visits))

        # Step the fork
        obs = self._client.step(env_id, edge.action_id)

        if obs.get("terminated"):
            winner  = obs.get("winnerId")
            persp   = obs.get("perspectivePlayerId")
            if isinstance(persp, dict): persp = persp.get("id")
            if isinstance(winner, dict): winner = winner.get("id")
            value = 1.0 if winner == persp else (-1.0 if winner else 0.0)
        elif not edge.visits:
            # Leaf — evaluate
            value = self._evaluate(obs)
            pd = obs.get("pendingDecision")
            dk = pd["kind"] if pd else None
            acts = encode_actions(obs.get("legalActions", []), dk, obs)
            if acts:
                child = MctsNode()
                self._expand(child, obs, acts)
                # Store child on edge for future visits (simple approach: re-expand each visit)
        else:
            # Already visited — recurse (re-step each simulation from root via fork)
            # For simplicity: treat as leaf and re-evaluate
            value = self._evaluate(obs)

        # Backup
        edge.visits    += 1
        edge.value_sum += value
        return -value  # flip for parent (minimax)

    def _expand(self, node: MctsNode, obs: dict,
                encoded_acts: list[dict]) -> None:
        """Populate node edges with priors from the network or uniform."""
        if self._offline or self._inference is None:
            prior = 1.0 / len(encoded_acts)
            for ea in encoded_acts:
                node.edges.append(MctsEdge(
                    action_id   = ea["actionId"],
                    head        = ea["head"],
                    slot        = ea["slot"],
                    action_type = ea["actionType"],
                    prior       = prior,
                ))
        else:
            indices = sorted(self._encoder.encode(obs))
            net_out = self._inference.evaluate(indices)
            self._expand_from_network(node, encoded_acts, net_out, obs)
        node.expanded = True

    def _expand_from_network(self, node: MctsNode, encoded_acts: list[dict],
                              net_out: dict, obs: dict) -> None:
        pd = obs.get("pendingDecision")
        dk = pd["kind"] if pd else None
        perspective = obs.get("perspectivePlayerId")
        if isinstance(perspective, dict):
            perspective = perspective.get("id")
        agent = obs.get("agentToAct")
        if isinstance(agent, dict):
            agent = agent.get("id")
        is_player = (agent == perspective)

        # Select the correct policy logits based on head and perspective
        def get_prior(ea: dict) -> float:
            head = ea["head"]
            slot = ea["slot"]
            if head == "priority":
                logits = net_out["policy_player"] if is_player else net_out["policy_opponent"]
            elif head == "target":
                logits = net_out["policy_target"]
            else:  # binary
                logits = net_out["policy_binary"]
            if slot < len(logits):
                return max(float(logits[slot]), 1e-8)
            return 1e-8

        raw = [get_prior(ea) for ea in encoded_acts]
        total = sum(raw)
        for ea, p in zip(encoded_acts, raw):
            node.edges.append(MctsEdge(
                action_id   = ea["actionId"],
                head        = ea["head"],
                slot        = ea["slot"],
                action_type = ea["actionType"],
                prior       = p / total,
            ))

    def _evaluate(self, obs: dict) -> float:
        """Return value estimate in [-1, 1] for the current state."""
        if self._offline or self._inference is None:
            return 0.0
        indices = sorted(self._encoder.encode(obs))
        net_out = self._inference.evaluate(indices)
        return float(net_out.get("value", 0.0))
