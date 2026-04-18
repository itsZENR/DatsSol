from typing import Optional
from simulator.engine.state import EngineState
from simulator.engine.events import (
    Event, PlantationDestroyed, ConstructionCompleted, ConstructionProgressed,
    UpgradeBought, HQLost,
)

SCORE_SCALE = 1000.0

# Shaping bonuses — DENSE reward to escape the idle local optimum.
# Each +1 of construction progress gives a small bonus so ANY build attempt is reinforced.
# Completed construction gives big bonus. Upgrades moderate. No penalties (they masked signal).
B_BUILD_PROGRESS = 0.02   # per +1 progress; 50-progress ramp-up → +1.0
B_BUILD = 3.0             # bonus on completion
B_UPGRADE = 0.5
B_HQ_LOSS = 0.0


def compute_reward(
    prev_score: int,
    cur_score: int,
    events: list[Event],
    state_after: Optional[EngineState],
) -> float:
    delta = (cur_score - prev_score) / SCORE_SCALE

    shaping = 0.0
    for e in events:
        if isinstance(e, ConstructionCompleted):
            shaping += B_BUILD
        elif isinstance(e, ConstructionProgressed):
            shaping += B_BUILD_PROGRESS * e.amount
        elif isinstance(e, UpgradeBought):
            shaping += B_UPGRADE
        elif isinstance(e, HQLost):
            shaping -= B_HQ_LOSS

    return delta + shaping
