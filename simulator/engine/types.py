from enum import Enum

Pos = tuple[int, int]
PlantID = str
BeaverID = str
StormID = str


class UpgradeName(str, Enum):
    REPAIR_POWER = "repair_power"
    MAX_HP = "max_hp"
    SETTLEMENT_LIMIT = "settlement_limit"
    SIGNAL_RANGE = "signal_range"
    VISION_RANGE = "vision_range"
    DECAY_MITIGATION = "decay_mitigation"
    EARTHQUAKE_MITIGATION = "earthquake_mitigation"
    BEAVER_DAMAGE_MITIGATION = "beaver_damage_mitigation"


class ActionKind(str, Enum):
    BUILD = "build"
    REPAIR = "repair"
    SABOTAGE = "sabotage"
    BEAVER = "beaver"
