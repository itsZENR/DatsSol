"""Δscore + minimal shaping (just enough to guide exploration to builds)."""
from typing import Optional
from simulator.engine.state import EngineState
from simulator.engine.events import Event, ConstructionCompleted, ConstructionProgressed

SCORE_SCALE = 1000.0

# Minimal shaping — small enough not to dominate true score signal.
B_BUILD = 0.5
B_BUILD_PROGRESS = 0.005


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
    return delta + shaping
