"""Δscore + shaping. Added BEAVER_KILL bonus to push policy past Go-plateau."""
from typing import Optional
from simulator.engine.state import EngineState
from simulator.engine.events import (
    Event, ConstructionCompleted, ConstructionProgressed, BeaverKilled,
)

SCORE_SCALE = 1000.0

B_BUILD = 0.5
B_BUILD_PROGRESS = 0.005
B_BEAVER_KILL = 30.0     # Big signal. Score delta already gives +20 per kill (20K points);
                         # this bonus pushes policy to prioritize attacks over safe terraform.


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
    return delta + shaping
