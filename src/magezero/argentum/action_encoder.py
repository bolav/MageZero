"""
action_encoder.py — maps Argentum LegalActionView dicts to MageZero (head, slot) pairs.

Slot assignment replicates Java's String.hashCode() to stay compatible with
ActionEncoder.java. Slot 0 is reserved for Pass (priority head) and Stop Choosing
(target head). Basic land tap abilities are hardcoded to slots 1–6.

ActionType int values match ActionEncoder.ActionType ordinals:
  PRIORITY = 0, CHOOSE_NUM = 1, BLANK = 2, CHOOSE_TARGET = 3,
  MAKE_CHOICE = 4, CHOOSE_USE = 5
"""
import re

# --------------------------------------------------------------------------
# ActionType constants (ordinals from ActionEncoder.java)
# --------------------------------------------------------------------------
ACTION_TYPE_PRIORITY      = 0
ACTION_TYPE_CHOOSE_NUM    = 1   # not currently trained
ACTION_TYPE_BLANK         = 2   # not currently trained
ACTION_TYPE_CHOOSE_TARGET = 3
ACTION_TYPE_MAKE_CHOICE   = 4   # not currently trained
ACTION_TYPE_CHOOSE_USE    = 5   # binary decisions (attackers, blockers, yes/no)

# --------------------------------------------------------------------------
# Hardcoded priority slots — Argentum's description format (no trailing period)
# ActionEncoder.java uses "{T}: Add {R}." but Argentum's AddManaEffect produces
# "{T}: Add {R}" (AbilityCost.Tap.description = "{T}", AddManaEffect.description
# = "Add {R}", joined as "{T}: Add {R}").
# --------------------------------------------------------------------------
_PRIORITY_FIXED: dict[str, int] = {
    "Pass": 0,
    "{T}: Add {B}": 1,
    "{T}: Add {G}": 2,
    "{T}: Add {R}": 3,
    "{T}: Add {U}": 4,
    "{T}: Add {W}": 5,
    "{T}: Add {C}": 6,
}

_TARGET_FIXED: dict[str, int] = {
    "Stop Choosing": 0,
    "PlayerA": 1,
    "PlayerB": 2,
}

_HEAD_SIZE_PRIORITY = 128
_HEAD_SIZE_TARGET   = 128
_HEAD_SIZE_BINARY   = 2

_UUID_RE = re.compile(r" \[[0-9a-f\-]{8,}\]")
_TAG_RE  = re.compile(r"<[^>]*>")


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------

def _java_hash(s: str) -> int:
    """Java String.hashCode() — signed 32-bit integer."""
    h = 0
    for c in s:
        h = (31 * h + ord(c)) & 0xFFFFFFFF
    return h - 0x100000000 if h >= 0x80000000 else h


def _hash_slot(name: str, modulus: int) -> int:
    """Slots 1..(modulus-1). Slot 0 is fully reserved."""
    return abs(_java_hash(name)) % (modulus - 1) + 1


# --------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------

def clean(s: str) -> str:
    """Remove UUIDs and HTML/XML tags — matches StateEncoder.cleanString()."""
    s = _UUID_RE.sub("", s)
    s = _TAG_RE.sub("", s)
    return s.strip()


def classify(kind: str, decision_kind: str | None) -> tuple[str, int]:
    """
    Return (head_name, action_type_int) for a LegalActionView.

    head_name is one of "priority", "target", "binary".
    action_type_int is written to HDF5 row[131].
    """
    if decision_kind == "CHOOSE_TARGETS":
        return "target", ACTION_TYPE_CHOOSE_TARGET
    if decision_kind == "YES_NO":
        return "binary", ACTION_TYPE_CHOOSE_USE
    if kind in ("DeclareAttackers", "DeclareBlockers"):
        return "binary", ACTION_TYPE_CHOOSE_USE
    return "priority", ACTION_TYPE_PRIORITY


def priority_slot(canonical: str) -> int:
    """Slot in [0, 127] for the priority head."""
    if canonical in _PRIORITY_FIXED:
        return _PRIORITY_FIXED[canonical]
    return _hash_slot(canonical, _HEAD_SIZE_PRIORITY)


def target_slot(canonical: str) -> int:
    """Slot in [0, 127] for the target head."""
    if canonical in _TARGET_FIXED:
        return _TARGET_FIXED[canonical]
    return _hash_slot(canonical, _HEAD_SIZE_TARGET)


def binary_slot(canonical: str) -> int:
    """Slot in {0, 1} for the binary head."""
    return abs(_java_hash(canonical)) % _HEAD_SIZE_BINARY


def slot_for(canonical: str, head: str) -> int:
    if head == "priority":
        return priority_slot(canonical)
    if head == "target":
        return target_slot(canonical)
    return binary_slot(canonical)


def canonical_priority(action: dict, obs: dict) -> str:
    """
    Stable canonical string for a priority-head action.
    Must not contain UUIDs or per-step IDs.
    """
    kind = action.get("kind", "")

    if kind == "PassPriority":
        return "Pass"

    if kind == "ActivateAbility" and action.get("isManaAbility"):
        # "{T}: Add {R}" — matches Argentum's ActivatedAbility.description format
        desc = clean(action.get("description", ""))
        if desc in _PRIORITY_FIXED:
            return desc
        return desc  # falls through to hash

    if kind == "DeclareAttackers":
        return "DeclareAttackers"

    if kind == "DeclareBlockers":
        return "DeclareBlockers"

    if kind == "DECISION":
        return clean(action.get("description", ""))

    desc = clean(action.get("description", kind))
    return desc


def canonical_target(target_id: str, obs: dict) -> str:
    """
    Stable canonical string for a target entity.
    Players are "PlayerA" / "PlayerB"; permanents use their card name.
    """
    players = obs.get("players", [])
    for i, p in enumerate(players):
        if p["id"]["id"] == target_id or p["id"] == target_id:
            return "PlayerA" if i == 0 else "PlayerB"

    for zone in obs.get("zones", []):
        for card in zone.get("cards", []):
            eid = card.get("entityId", {})
            eid_val = eid.get("id", eid) if isinstance(eid, dict) else eid
            if eid_val == target_id:
                return clean(card.get("name", target_id))

    return clean(str(target_id))


def encode_actions(legal_actions: list[dict], decision_kind: str | None,
                   obs: dict) -> list[dict]:
    """
    Return a list of dicts with actionId, head, slot, description for each
    affordable legal action.
    """
    result = []
    for action in legal_actions:
        if not action.get("affordable", True):
            continue
        kind = action.get("kind", "")
        head, action_type = classify(kind, decision_kind)
        canonical = canonical_priority(action, obs)
        s = slot_for(canonical, head)
        result.append({
            "actionId": action["actionId"],
            "head": head,
            "slot": s,
            "actionType": action_type,
            "description": action.get("description", ""),
        })
    return result
