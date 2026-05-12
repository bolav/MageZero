"""
encoder.py — converts a TrainingObservation JSON dict into a set of sparse
feature indices compatible with MageZero's EmbeddingBag network.

Mirrors MageZero's StateEncoder.processState() but reads from the structured
JSON observation rather than walking the XMage game object tree.

Namespace hierarchy:
  {step}
  {phase}
  {decisionType}
  Player.{...}      — acting player's perspective
  Opponent.{...}    — opponent's perspective
  Stack.{name}.{...}
  Exile.{name}.{...}
"""
from .feature_map import FeatureMap
from .action_encoder import clean

# Numeric caps to keep vocabulary bounded
_LIFE_CAP    = 40
_PT_CAP      = 15
_LIBRARY_CAP = 60
_DAMAGE_CAP  = 15
_MANA_CAP    = 10


def _bucket_life(n: int) -> str:
    return str(min(max(n, 0), _LIFE_CAP))


def _bucket_library(n: int) -> str:
    if n == 0:      return "0"
    if n <= 5:      return "1-5"
    if n <= 10:     return "6-10"
    if n <= 20:     return "11-20"
    if n <= 40:     return "21-40"
    return "41+"


def _bucket_pt(n: int) -> str:
    return str(min(max(n, 0), _PT_CAP))


def _bucket_mana(n: int) -> str:
    return str(min(max(n, 0), _MANA_CAP))


class ArgentumStateEncoder:
    def __init__(self, feature_map: FeatureMap):
        self._fm = feature_map

    def encode(self, obs: dict) -> set[int]:
        """
        Return the set of active feature indices for this observation.
        Caller should sort before writing to HDF5.
        """
        indices: set[int] = set()
        f = self._fm.id

        # Turn structure
        step = obs.get("step")
        if step:
            indices.add(f(str(step)))
        phase = obs.get("phase")
        if phase:
            indices.add(f(str(phase)))

        # Decision type
        pd = obs.get("pendingDecision")
        decision_type = pd["kind"] if pd else "PRIORITY"
        indices.add(f(decision_type))

        perspective_id = _player_id(obs.get("perspectivePlayerId"))
        agent_id       = _player_id(obs.get("agentToAct"))

        # Players
        players = obs.get("players", [])
        for player in players:
            pid = _player_id(player["id"])
            is_perspective = (pid == perspective_id)
            prefix = "Player" if is_perspective else "Opponent"
            self._encode_player(player, obs.get("zones", []),
                                is_perspective, prefix, agent_id, indices)

        # Stack (bottom → top, top is last)
        for item in obs.get("stack", []):
            ns = f"Stack.{clean(item.get('name', ''))}"
            indices.add(f(f"{ns}.OnStack"))
            ctrl = _player_id(item.get("controllerId"))
            if ctrl == perspective_id:
                indices.add(f(f"{ns}.IsController"))

        return indices

    # ------------------------------------------------------------------
    # Player section
    # ------------------------------------------------------------------

    def _encode_player(self, player: dict, zones: list[dict],
                       is_perspective: bool, prefix: str,
                       agent_id: str, indices: set[int]) -> None:
        f = self._fm.id
        pid = _player_id(player["id"])

        life = player.get("lifeTotal", 0)
        indices.add(f(f"{prefix}.LifeTotal:{_bucket_life(life)}"))

        lib = player.get("librarySize", 0)
        indices.add(f(f"{prefix}.LibraryCount:{_bucket_library(lib)}"))

        if player.get("isActive"):
            indices.add(f(f"{prefix}.IsActivePlayer"))
        if player.get("hasPriority"):
            indices.add(f(f"{prefix}.HasPriority"))
        if pid == agent_id:
            indices.add(f(f"{prefix}.IsDecisionPlayer"))

        # Mana pool
        mana = player.get("manaPool", {})
        for color, key in (("white", "W"), ("blue", "U"), ("black", "B"),
                           ("red", "R"), ("green", "G"), ("colorless", "C")):
            val = mana.get(color, 0)
            if val > 0:
                indices.add(f(f"{prefix}.ManaPool.{key}:{_bucket_mana(val)}"))

        # Zones
        for zone_view in zones:
            zone_owner = _player_id(zone_view.get("ownerId"))
            if zone_owner != pid:
                continue
            zone_type = zone_view.get("zoneType", "")
            if zone_view.get("hidden"):
                # Hidden zones: only size
                size = zone_view.get("size", 0)
                indices.add(f(f"{prefix}.CardsInHand:{size}"))
                continue
            for card in zone_view.get("cards", []):
                self._encode_card(card, prefix, zone_type, indices)

    # ------------------------------------------------------------------
    # Card / permanent section
    # ------------------------------------------------------------------

    def _encode_card(self, card: dict, prefix: str,
                     zone_type: str, indices: set[int]) -> None:
        f = self._fm.id
        name = clean(card.get("name", ""))
        ns = f"{prefix}.{zone_type}.{name}"

        indices.add(f(f"{ns}.Card"))

        # Types, subtypes, colors, keywords (from projected state)
        for t in card.get("types", []):
            indices.add(f(f"{ns}.{t}"))

        for st in card.get("subtypes", []):
            indices.add(f(f"{ns}.{st}"))

        for color in card.get("colors", []):
            indices.add(f(f"{ns}.{color}Card"))

        for kw in card.get("keywords", []):
            indices.add(f(f"{ns}.Keyword.{kw}"))

        mv = card.get("manaValue", 0)
        indices.add(f(f"{ns}.ManaValue:{mv}"))

        # Mana cost pips  e.g. "{1}{R}{R}" → "{1}", "{R}", "{R}"
        mana_cost = card.get("manaCost", "")
        if mana_cost:
            import re
            for pip in re.findall(r"\{[^}]+\}", mana_cost):
                indices.add(f(f"{ns}.Pip.{pip}"))

        # Battlefield-only dynamic features
        if zone_type == "BATTLEFIELD":
            indices.add(f(f"{ns}.Permanent"))

            # Dynamic types (may differ from base due to layer effects)
            for t in card.get("types", []):
                indices.add(f(f"{ns}.{t}_dynamic"))

            if card.get("tapped"):
                indices.add(f(f"{ns}.Tapped"))
            if card.get("summoningSick"):
                indices.add(f(f"{ns}.SummoningSick"))

            power = card.get("power")
            toughness = card.get("toughness")
            if power is not None:
                indices.add(f(f"{ns}.Power:{_bucket_pt(power)}"))
            if toughness is not None:
                indices.add(f(f"{ns}.Toughness:{_bucket_pt(toughness)}"))

            dmg = card.get("damageMarked", 0)
            if dmg > 0:
                indices.add(f(f"{ns}.Damage:{min(dmg, _DAMAGE_CAP)}"))

            for counter_name, count in card.get("counters", {}).items():
                indices.add(f(f"{ns}.Counter.{counter_name}:{count}"))


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _player_id(raw) -> str:
    """Normalise EntityId (may be a dict {"id": "..."} or a plain string)."""
    if isinstance(raw, dict):
        return raw.get("id", "")
    return str(raw) if raw is not None else ""
