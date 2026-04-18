from dataclasses import dataclass, field
from typing import Optional
from collections import deque

Pos = tuple[int, int]


@dataclass
class Plantation:
    id: str
    position: Pos
    is_main: bool
    is_isolated: bool
    immunity_until_turn: int
    hp: int


@dataclass
class Enemy:
    id: str
    position: Pos
    hp: int


@dataclass
class Cell:
    position: Pos
    terraformation_progress: float
    turns_until_degradation: int


@dataclass
class Construction:
    position: Pos
    progress: float


@dataclass
class Beaver:
    id: str
    position: Pos
    hp: int


@dataclass
class UpgradeTier:
    name: str
    current: int
    max: int


@dataclass
class PlantationUpgrades:
    points: int
    interval_turns: int
    turns_until_points: int
    max_points: int
    tiers: list[UpgradeTier]


@dataclass
class MeteoForecast:
    kind: str
    turns_until: int
    id: Optional[str] = None
    forming: Optional[bool] = None
    position: Optional[Pos] = None
    next_position: Optional[Pos] = None
    radius: Optional[int] = None


@dataclass
class GameState:
    turn_no: int
    next_turn_in: float
    size: Pos
    action_range: int
    plantations: list[Plantation]
    enemies: list[Enemy]
    mountains: list[Pos]
    cells: list[Cell]
    constructions: list[Construction]
    beavers: list[Beaver]
    upgrades: PlantationUpgrades
    meteo_forecasts: list[MeteoForecast]

    # Derived fields (populated in __post_init__)
    mountains_set: set[Pos] = field(default_factory=set)
    plantation_map: dict[Pos, Plantation] = field(default_factory=dict)
    construction_set: set[Pos] = field(default_factory=set)
    enemy_map: dict[Pos, Enemy] = field(default_factory=dict)
    connected_ids: set[str] = field(default_factory=set)
    connected_plantations: list[Plantation] = field(default_factory=list)
    main_plantation: Optional[Plantation] = None

    def __post_init__(self):
        self.mountains_set = set(map(tuple, self.mountains))
        self.plantation_map = {tuple(p.position): p for p in self.plantations}
        self.construction_set = {tuple(c.position) for c in self.constructions}
        self.enemy_map = {tuple(e.position): e for e in self.enemies}
        self.main_plantation = next((p for p in self.plantations if p.is_main), None)
        self._compute_connectivity()

    def _compute_connectivity(self):
        if not self.main_plantation:
            return
        start = tuple(self.main_plantation.position)
        visited = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            p = self.plantation_map.get(cur)
            if p:
                self.connected_ids.add(p.id)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nb = (cur[0] + dx, cur[1] + dy)
                if nb not in visited and nb in self.plantation_map:
                    visited.add(nb)
                    queue.append(nb)
        self.connected_plantations = [
            p for p in self.plantations if p.id in self.connected_ids
        ]

    def get_upgrade_tier(self, name: str) -> Optional[UpgradeTier]:
        return next((t for t in self.upgrades.tiers if t.name == name), None)

    def is_cell_free(self, pos: Pos) -> bool:
        return (
            pos not in self.mountains_set
            and pos not in self.plantation_map
            and pos not in self.construction_set
            and pos not in self.enemy_map
            and 0 <= pos[0] < self.size[0]
            and 0 <= pos[1] < self.size[1]
        )

    def is_boosted_cell(self, pos: Pos) -> bool:
        return pos[0] % 7 == 0 and pos[1] % 7 == 0


@dataclass
class Command:
    author: Pos
    relay: Pos
    target: Pos

    def to_path(self) -> list[list[int]]:
        return [list(self.author), list(self.relay), list(self.target)]


@dataclass
class TurnDecision:
    commands: list[Command] = field(default_factory=list)
    upgrade: str = ""
    relocate_main: list[Pos] = field(default_factory=list)

    def to_payload(self) -> dict:
        payload: dict = {}
        if self.commands:
            payload["command"] = [{"path": c.to_path()} for c in self.commands]
        if self.upgrade:
            payload["plantationUpgrade"] = self.upgrade
        if self.relocate_main:
            payload["relocateMain"] = [list(p) for p in self.relocate_main]
        if not payload:
            # Server requires at least one action; send empty upgrade as fallback
            payload["plantationUpgrade"] = ""
        return payload
