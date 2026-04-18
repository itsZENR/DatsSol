from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from simulator.engine.config import GameConfig
from simulator.engine.entities import (
    Plantation, Construction, Beaver, Storm, CellState,
    UpgradesState, UpgradeTier,
)
from simulator.engine.events import Event
from simulator.engine.types import Pos, PlantID, BeaverID, UpgradeName


@dataclass
class EngineState:
    config: GameConfig
    rng: np.random.Generator
    turn_no: int = 0
    plantations: dict[PlantID, Plantation] = field(default_factory=dict)
    constructions: dict[Pos, Construction] = field(default_factory=dict)
    beavers: dict[BeaverID, Beaver] = field(default_factory=dict)
    cells: dict[Pos, CellState] = field(default_factory=dict)
    mountains: frozenset[Pos] = field(default_factory=frozenset)
    storms: list[Storm] = field(default_factory=list)
    earthquake_turn: Optional[int] = None
    upgrades: UpgradesState = field(default_factory=lambda: UpgradesState(
        points=0, turns_until_points=30, tiers={},
    ))
    hq_id: Optional[PlantID] = None
    score: int = 0
    events_this_turn: list[Event] = field(default_factory=list)
    connected_ids: set[PlantID] = field(default_factory=set)
    next_plant_counter: int = 0

    @classmethod
    def new(cls, config: GameConfig, seed: int) -> "EngineState":
        rng = np.random.default_rng(seed)
        st = cls(config=config, rng=rng)
        default_max = {
            UpgradeName.REPAIR_POWER: 3,
            UpgradeName.MAX_HP: 5,
            UpgradeName.SETTLEMENT_LIMIT: 10,
            UpgradeName.SIGNAL_RANGE: 99,
            UpgradeName.VISION_RANGE: 5,
            UpgradeName.DECAY_MITIGATION: 3,
            UpgradeName.EARTHQUAKE_MITIGATION: 3,
            UpgradeName.BEAVER_DAMAGE_MITIGATION: 5,
        }
        for name, mx in default_max.items():
            st.upgrades.tiers[name] = UpgradeTier(current=0, max=mx)
        st.upgrades.turns_until_points = config.upgrade_interval
        return st

    def get_hq(self) -> Optional[Plantation]:
        if self.hq_id is None:
            return None
        return self.plantations.get(self.hq_id)

    def tier_level(self, name: UpgradeName) -> int:
        t = self.upgrades.tiers.get(name)
        return t.current if t is not None else 0

    def effective_rs(self) -> int:
        return self.config.rs + self.tier_level(UpgradeName.REPAIR_POWER)

    def effective_mhp(self) -> int:
        return self.config.mhp + 10 * self.tier_level(UpgradeName.MAX_HP)

    def effective_sr(self) -> int:
        return self.config.sr + self.tier_level(UpgradeName.SIGNAL_RANGE)

    def effective_vr(self) -> int:
        return self.config.vr + 2 * self.tier_level(UpgradeName.VISION_RANGE)

    def effective_plantation_limit(self) -> int:
        return self.config.base_plantation_limit + self.tier_level(
            UpgradeName.SETTLEMENT_LIMIT
        )

    def effective_beaver_damage(self) -> int:
        return max(
            0,
            self.config.beaver_damage - 2 * self.tier_level(
                UpgradeName.BEAVER_DAMAGE_MITIGATION
            ),
        )

    def effective_earthquake_damage(self) -> int:
        return max(
            0,
            self.config.earthquake_damage - 2 * self.tier_level(
                UpgradeName.EARTHQUAKE_MITIGATION
            ),
        )

    def effective_ds(self) -> int:
        return max(
            0,
            self.config.ds - 2 * self.tier_level(UpgradeName.DECAY_MITIGATION),
        )

    def is_boosted_cell(self, pos: Pos) -> bool:
        m = self.config.boosted_mod
        return pos[0] % m == 0 and pos[1] % m == 0

    def cell_reward(self, pos: Pos) -> int:
        return (
            self.config.boosted_cell_reward
            if self.is_boosted_cell(pos)
            else self.config.normal_cell_reward
        )

    def new_plant_id(self) -> PlantID:
        pid = f"p{self.next_plant_counter}"
        self.next_plant_counter += 1
        return pid
