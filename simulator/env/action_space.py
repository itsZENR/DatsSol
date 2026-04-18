from dataclasses import dataclass
from enum import IntEnum
from gymnasium import spaces

MAX_PLANT = 40
AR = 2  # Fixed by game rules (no upgrade for AR)
TARGET_OFFSET_DIM = (2 * AR + 1) ** 2   # 25
N_UPGRADES = 8


class ActionKind(IntEnum):
    END_TURN = 0
    SET_UPGRADE = 1
    SET_RELOCATE = 2
    COMMAND = 3
    NOOP = 4


@dataclass
class DecodedAction:
    kind: ActionKind
    author_idx: int
    relay_idx: int
    target_offset: int
    upgrade_id: int


def build_action_space() -> spaces.Dict:
    return spaces.Dict({
        "kind":          spaces.Discrete(len(ActionKind)),
        "author_idx":    spaces.Discrete(MAX_PLANT),
        "relay_idx":     spaces.Discrete(MAX_PLANT),
        "target_offset": spaces.Discrete(TARGET_OFFSET_DIM),
        "upgrade_id":    spaces.Discrete(N_UPGRADES),
    })


def decode_action(raw: dict) -> DecodedAction:
    return DecodedAction(
        kind=ActionKind(int(raw["kind"])),
        author_idx=int(raw["author_idx"]),
        relay_idx=int(raw["relay_idx"]),
        target_offset=int(raw["target_offset"]),
        upgrade_id=int(raw["upgrade_id"]),
    )


def target_offset_to_delta(offset: int) -> tuple[int, int]:
    """Offset in [0, 24] → (dx, dy) in [-AR, AR]."""
    side = 2 * AR + 1
    dx = (offset // side) - AR
    dy = (offset % side) - AR
    return dx, dy


def delta_to_target_offset(dx: int, dy: int) -> int:
    side = 2 * AR + 1
    return (dx + AR) * side + (dy + AR)
