"""
client_encoder.py — encodes ClientGameState (game-server format) into sparse
feature indices for the MageZero EmbeddingBag network.

ClientGameState is the format the game-server sends to the web client.
It differs from TrainingObservation (gym-server format) in field names and
structure — cards are in a flat map keyed by entity ID, zones reference card
IDs rather than embedding card data directly.

Field mapping from ClientDTO.kt:
  state["currentStep"]      → step feature
  state["currentPhase"]     → phase feature
  state["viewingPlayerId"]  → perspective player
  state["priorityPlayerId"] → agentToAct
  player["life"]            → life total
  player["handSize"]        → hand size (opponent)
  player["librarySize"]     → library size
  player["manaPool"]        → mana pool (may be None for opponent)
  card["cardTypes"]         → types (Set<String>)
  card["subtypes"]          → subtypes
  card["colors"]            → colors (serialized as strings)
  card["keywords"]          → keywords
  card["isTapped"]          → tapped
  card["hasSummoningSickness"] → summoning sick
  card["damage"]            → damage marked
  card["counters"]          → counters (Map<CounterType, Int>)
  card["power"], ["toughness"] → P/T (projected values)
"""
from .feature_map import FeatureMap
from .action_encoder import clean

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


def _eid(raw) -> str:
    """Normalise EntityId — may be {"value": "..."} or a plain string."""
    if isinstance(raw, dict):
        return raw.get("value", raw.get("id", str(raw)))
    return str(raw) if raw is not None else ""


def _zone_type(zone_id: dict) -> str:
    """Extract zone type string from ZoneKey dict."""
    if isinstance(zone_id, dict):
        return str(zone_id.get("zoneType", ""))
    return str(zone_id)


def _zone_owner(zone_id: dict) -> str:
    """Extract owner player ID from ZoneKey dict."""
    if isinstance(zone_id, dict):
        raw = zone_id.get("playerId")
        return _eid(raw)
    return ""


class ClientStateEncoder:
    def __init__(self, feature_map: FeatureMap):
        self._fm = feature_map

    def encode(self, state: dict, pending_decision: dict | None = None) -> set[int]:
        """
        Encode a ClientGameState dict into a set of active feature indices.
        pending_decision: the PendingDecision dict if mid-decision, else None.
        """
        indices: set[int] = set()
        f = self._fm.id

        # Turn structure
        step = state.get("currentStep")
        if step:
            indices.add(f(str(step)))
        phase = state.get("currentPhase")
        if phase:
            indices.add(f(str(phase)))

        # Decision type
        decision_kind = pending_decision["type"] if pending_decision else "PRIORITY"
        indices.add(f(decision_kind))

        perspective_id = _eid(state.get("viewingPlayerId"))
        priority_id    = _eid(state.get("priorityPlayerId"))
        active_id      = _eid(state.get("activePlayerId"))

        # Cards map (entity ID → ClientCard dict)
        cards: dict[str, dict] = {}
        for k, v in state.get("cards", {}).items():
            cards[_eid(k)] = v

        # Players
        for player in state.get("players", []):
            pid = _eid(player.get("playerId"))
            is_perspective = (pid == perspective_id)
            prefix = "Player" if is_perspective else "Opponent"
            self._encode_player(player, pid, prefix,
                                active_id, priority_id, perspective_id,
                                indices)

        # Zones — group cards by owner+type
        zones = state.get("zones", [])
        for zone_view in zones:
            zid   = zone_view.get("zoneId", {})
            ztype = _zone_type(zid)
            owner = _zone_owner(zid)
            is_perspective_owner = (owner == perspective_id)
            prefix = "Player" if is_perspective_owner else "Opponent"

            if not zone_view.get("isVisible", True):
                # Hidden zone — only size
                size = zone_view.get("size", 0)
                indices.add(f(f"{prefix}.CardsInHand:{size}"))
                continue

            if ztype == "STACK":
                for cid in zone_view.get("cardIds", []):
                    card = cards.get(_eid(cid))
                    if card:
                        self._encode_stack_card(card, perspective_id, indices)
                continue

            for cid in zone_view.get("cardIds", []):
                card = cards.get(_eid(cid))
                if card:
                    self._encode_card(card, prefix, ztype, indices)

        return indices

    # ------------------------------------------------------------------
    # Player section
    # ------------------------------------------------------------------

    def _encode_player(self, player: dict, pid: str, prefix: str,
                       active_id: str, priority_id: str, perspective_id: str,
                       indices: set[int]) -> None:
        f = self._fm.id

        life = player.get("life", 0)
        indices.add(f(f"{prefix}.LifeTotal:{_bucket_life(life)}"))

        lib = player.get("librarySize", 0)
        indices.add(f(f"{prefix}.LibraryCount:{_bucket_library(lib)}"))

        if pid == active_id:
            indices.add(f(f"{prefix}.IsActivePlayer"))
        if pid == priority_id:
            indices.add(f(f"{prefix}.HasPriority"))
        if pid == priority_id:
            indices.add(f(f"{prefix}.IsDecisionPlayer"))

        mana = player.get("manaPool") or {}
        for color, key in (("white", "W"), ("blue", "U"), ("black", "B"),
                           ("red", "R"), ("green", "G"), ("colorless", "C")):
            val = mana.get(color, 0) or 0
            if val > 0:
                indices.add(f(f"{prefix}.ManaPool.{key}:{_bucket_mana(val)}"))

    # ------------------------------------------------------------------
    # Card / permanent section
    # ------------------------------------------------------------------

    def _encode_card(self, card: dict, prefix: str,
                     zone_type: str, indices: set[int]) -> None:
        f = self._fm.id
        name = clean(card.get("name", ""))
        ns = f"{prefix}.{zone_type}.{name}"

        indices.add(f(f"{ns}.Card"))

        for t in card.get("cardTypes", []):
            indices.add(f(f"{ns}.{t}"))
        for st in card.get("subtypes", []):
            indices.add(f(f"{ns}.{st}"))
        for color in card.get("colors", []):
            color_str = color if isinstance(color, str) else str(color)
            indices.add(f(f"{ns}.{color_str}Card"))
        for kw in card.get("keywords", []):
            kw_str = kw if isinstance(kw, str) else str(kw)
            indices.add(f(f"{ns}.Keyword.{kw_str}"))

        mv = card.get("manaValue", 0)
        indices.add(f(f"{ns}.ManaValue:{mv}"))

        import re
        mana_cost = card.get("manaCost", "") or ""
        for pip in re.findall(r"\{[^}]+\}", mana_cost):
            indices.add(f(f"{ns}.Pip.{pip}"))

        if zone_type == "BATTLEFIELD":
            indices.add(f(f"{ns}.Permanent"))

            for t in card.get("cardTypes", []):
                indices.add(f(f"{ns}.{t}_dynamic"))

            if card.get("isTapped"):
                indices.add(f(f"{ns}.Tapped"))
            if card.get("hasSummoningSickness"):
                indices.add(f(f"{ns}.SummoningSick"))

            power = card.get("power")
            toughness = card.get("toughness")
            if power is not None:
                indices.add(f(f"{ns}.Power:{_bucket_pt(power)}"))
            if toughness is not None:
                indices.add(f(f"{ns}.Toughness:{_bucket_pt(toughness)}"))

            dmg = card.get("damage") or 0
            if dmg > 0:
                indices.add(f(f"{ns}.Damage:{min(dmg, _DAMAGE_CAP)}"))

            for counter_type, count in (card.get("counters") or {}).items():
                cname = counter_type if isinstance(counter_type, str) else str(counter_type)
                indices.add(f(f"{ns}.Counter.{cname}:{count}"))

    def _encode_stack_card(self, card: dict, perspective_id: str,
                           indices: set[int]) -> None:
        f = self._fm.id
        name = clean(card.get("name", ""))
        ns = f"Stack.{name}"
        indices.add(f(f"{ns}.OnStack"))
        ctrl = _eid(card.get("controllerId"))
        if ctrl == perspective_id:
            indices.add(f(f"{ns}.IsController"))
