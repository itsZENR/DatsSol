from typing import Optional
from simulator.engine.state import EngineState
from simulator.engine.events import (
    Event, PlantationDestroyed, ConstructionCompleted, UpgradeBought, HQLost,
)

SCORE_SCALE = 1000.0

# Shaping bonuses (in reward units).
# Goal: strongly reinforce building & upgrades to escape the idle-converges-to-local-optimum
# trap. Penalties removed — they dominated early exploration and discouraged any building.
B_BUILD = 2.0           # Big bonus for each completed construction (vs cell reward = 1.0)
B_UPGRADE = 0.5
B_HQ_LOSS = 0.0         # Score delta already penalizes HQ loss (5% of current score)


def compute_reward(
    prev_score: int,
    cur_score: int,
    events: list[Event],
    state_after: Optional[EngineState],
) -> float:
    delta = (cur_score - prev_score) / SCORE_SCALE

    # Shaped bonuses for positive events.
    shaping = 0.0
    for e in events:
        if isinstance(e, ConstructionCompleted):
            shaping += B_BUILD
        elif isinstance(e, UpgradeBought):
            shaping += B_UPGRADE
        elif isinstance(e, HQLost):
            shaping -= B_HQ_LOSS

    return delta + shaping
