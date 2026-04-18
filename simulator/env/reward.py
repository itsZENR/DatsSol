from typing import Optional
from simulator.engine.state import EngineState
from simulator.engine.events import (
    Event, PlantationDestroyed, ConstructionCompleted, UpgradeBought, HQLost,
)

SCORE_SCALE = 1000.0

# Shaping bonuses (in reward units, added to Δscore/SCORE_SCALE):
#   - Each completed construction gives a dense +B_BUILD (compared to final cell reward 1000/SCORE_SCALE = 1.0).
#   - Each upgrade purchased gives +B_UPGRADE.
#   - HQ loss adds an explicit penalty beyond the score delta (to discourage suicide-builds).
B_BUILD = 0.5
B_UPGRADE = 0.2
B_HQ_LOSS = 1.0


def compute_reward(
    prev_score: int,
    cur_score: int,
    events: list[Event],
    state_after: Optional[EngineState],
) -> float:
    delta = (cur_score - prev_score) / SCORE_SCALE

    # Penalty for losing own plantations (lost cell potential)
    penalty = 0.0
    if state_after is not None:
        for e in events:
            if isinstance(e, PlantationDestroyed) and not e.was_main:
                cell = state_after.cells.get(e.position)
                progress = cell.progress if cell is not None else 0
                potential = state_after.cell_reward(e.position) * (100 - progress) / 100.0
                penalty += potential / SCORE_SCALE

    # Shaped bonuses: dense signal so PPO can credit early discoveries.
    shaping = 0.0
    for e in events:
        if isinstance(e, ConstructionCompleted):
            shaping += B_BUILD
        elif isinstance(e, UpgradeBought):
            shaping += B_UPGRADE
        elif isinstance(e, HQLost):
            shaping -= B_HQ_LOSS

    return delta - penalty + shaping
