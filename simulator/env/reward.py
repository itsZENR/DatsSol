"""Pure-score reward — no shaping.

Rationale: shaping bonuses (build completion etc) compete with actual score gradient
and teach policy to optimize proxy rather than real objective. With attention-based
architecture and proper compute, raw Δscore is the right signal.
"""
from typing import Optional
from simulator.engine.state import EngineState
from simulator.engine.events import Event

SCORE_SCALE = 1000.0


def compute_reward(
    prev_score: int,
    cur_score: int,
    events: list[Event],
    state_after: Optional[EngineState],
) -> float:
    return (cur_score - prev_score) / SCORE_SCALE
