"""Δscore + shaping. Added per-turn plantation-count bonus to push expansion."""
from typing import Optional
from simulator.engine.state import EngineState
from simulator.engine.events import (
    Event, ConstructionCompleted, ConstructionProgressed,
    BeaverKilled, BeaverDamaged,
)

SCORE_SCALE = 1000.0

B_BUILD = 0.5
B_BUILD_PROGRESS = 0.005
B_BEAVER_KILL = 30.0
B_BEAVER_DAMAGE = 0.3
# Per-plantation-per-tick bonus — pushes expansion up to limit.
B_PLANT_ALIVE = 0.01


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
        elif isinstance(e, BeaverKilled):
            shaping += B_BEAVER_KILL
        elif isinstance(e, BeaverDamaged):
            shaping += B_BEAVER_DAMAGE * e.amount
    # Per-plantation bonus applied each reward tick
    if state_after is not None:
        n_own = sum(
            1 for pid in state_after.plantations
            if pid not in state_after.enemy_ids
        )
        shaping += B_PLANT_ALIVE * n_own
    return delta + shaping
