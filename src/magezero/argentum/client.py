"""
client.py — thin HTTP wrapper over the Argentum gym-server.

All methods raise on non-2xx responses. Entity IDs are normalised to plain
strings internally; the server sends them as {"id": "..."} objects.
"""
import json
import urllib.error
import urllib.request
from typing import Any


class ArgentumClient:
    def __init__(self, base_url: str = "http://localhost:8090"):
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_env(self, config: dict) -> tuple[str, dict]:
        """
        POST /envs — create a new environment.
        Returns (env_id, observation).
        """
        resp = self._post("/envs", config)
        return _env_id(resp["envId"]), resp["observation"]

    def reset(self, env_id: str, config: dict) -> dict:
        """POST /envs/{id}/reset — reset an existing env. Returns observation."""
        return self._post(f"/envs/{env_id}/reset", config)

    def dispose(self, env_ids: list[str]) -> None:
        """DELETE /envs — dispose a list of envs."""
        self._delete("/envs", {"envIds": env_ids})

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def observe(self, env_id: str, reveal_all: bool = False) -> dict:
        """GET /envs/{id} — observe without advancing."""
        url = f"/envs/{env_id}"
        if reveal_all:
            url += "?revealAll=true"
        return self._get(url)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, env_id: str, action_id: int) -> dict:
        """POST /envs/{id}/step — advance by one action. Returns observation."""
        return self._post(f"/envs/{env_id}/step", {"actionId": action_id})

    def step_batch(self, requests: list[tuple[str, int]]) -> list[dict]:
        """POST /envs/step-batch — advance multiple envs in parallel."""
        body = [{"envId": env_id, "actionId": aid}
                for env_id, aid in requests]
        results = self._post("/envs/step-batch", body)
        return [r["observation"] for r in results]

    def submit_decision(self, env_id: str, response: dict) -> dict:
        """POST /envs/{id}/decision — submit a structured decision response."""
        return self._post(f"/envs/{env_id}/decision", response)

    def heuristic_step(self, env_id: str) -> dict:
        """
        POST /envs/{id}/heuristic-step — run the engine heuristic AI for the
        acting player and advance. Returns HeuristicStepResult:
          { observation, heuristicActionId, nextObservation }
        """
        return self._post(f"/envs/{env_id}/heuristic-step", {})

    # ------------------------------------------------------------------
    # Fork / snapshot / restore
    # ------------------------------------------------------------------

    def fork(self, env_id: str, count: int = 1) -> list[str]:
        """POST /envs/{id}/fork — fork an env N times. Returns list of new env IDs."""
        result = self._post(f"/envs/{env_id}/fork?count={count}", None)
        if isinstance(result, list):
            return [_env_id(e) for e in result]
        return [_env_id(result)]

    def snapshot(self, env_id: str) -> str:
        """POST /envs/{id}/snapshot — save a snapshot. Returns handle string."""
        result = self._post(f"/envs/{env_id}/snapshot", None)
        return result.get("handle", str(result))

    def restore(self, env_id: str, handle: str) -> dict:
        """POST /envs/{id}/restore — restore from snapshot. Returns observation."""
        return self._post(f"/envs/{env_id}/restore", {"handle": handle})

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def health(self) -> bool:
        try:
            self._get("/actuator/health")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> Any:
        req = urllib.request.Request(self.base_url + path)
        return self._send(req)

    def _post(self, path: str, body: Any) -> Any:
        data = json.dumps(body).encode() if body is not None else b""
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(req)

    def _delete(self, path: str, body: Any) -> Any:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        return self._send(req)

    def _send(self, req: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(
                f"gym-server {req.method} {req.full_url} → {e.code}: {body}"
            ) from e


def _env_id(raw) -> str:
    """Normalise EnvId — inline value class serialises as a plain string."""
    if isinstance(raw, dict):
        # fallback for any future wrapper format
        return raw.get("value", raw.get("id", str(raw)))
    return str(raw)
