from dataclasses import dataclass, field
from typing import Optional
from simulator.engine.types import Pos, PlantID, BeaverID, StormID, UpgradeName


@dataclass
class Plantation:
    id: PlantID
    position: Pos
    is_main: bool
    hp: int
    immunity_until_turn: int
    built_on_turn: int


@dataclass
class Construction:
    position: Pos
    progress: int
    last_progress_turn: int
    contributors_last_turn: dict[PlantID, int]  # author_id -> cs_contribution


@dataclass
class Beaver:
    id: BeaverID
    position: Pos
    hp: int
    last_hit_by: dict[str, int]  # player_id -> last_turn_hit
    total_damage_this_turn: dict[str, int]  # player_id -> damage_this_turn


@dataclass
class Storm:
    id: StormID
    position: Pos
    radius: int
    speed: int
    direction: tuple[int, int]  # (dx, dy) — normalized +1/-1
    forming_left: int  # 0 if already moving


@dataclass
class CellState:
    progress: int
    completed_on_turn: Optional[int]  # turn when reached 100% (for 30-turn timer)


@dataclass
class UpgradeTier:
    current: int
    max: int


@dataclass
class UpgradesState:
    points: int
    turns_until_points: int
    tiers: dict[UpgradeName, UpgradeTier] = field(default_factory=dict)
