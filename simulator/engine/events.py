from dataclasses import dataclass
from typing import Union
from simulator.engine.types import Pos, PlantID, BeaverID


@dataclass(frozen=True)
class Scored:
    amount: int
    source: str  # "terraform" | "sabotage" | "beaver"


@dataclass(frozen=True)
class PlantationDestroyed:
    plantation_id: PlantID
    position: Pos
    was_main: bool


@dataclass(frozen=True)
class BeaverKilled:
    beaver_id: BeaverID
    position: Pos
    reward: int


@dataclass(frozen=True)
class HQLost:
    prev_hq_id: PlantID
    penalty: int


@dataclass(frozen=True)
class ConstructionCompleted:
    position: Pos
    new_plantation_id: PlantID


@dataclass(frozen=True)
class CellTerraformed:
    position: Pos
    reward: int


@dataclass(frozen=True)
class UpgradeBought:
    name: str
    new_level: int


Event = Union[
    Scored, PlantationDestroyed, BeaverKilled, HQLost,
    ConstructionCompleted, CellTerraformed, UpgradeBought,
]
