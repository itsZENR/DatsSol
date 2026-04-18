from typing import Optional
from simulator.engine.state import EngineState
from simulator.engine.events import Event, PlantationDestroyed

SCORE_SCALE = 1000.0


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

    return delta - penalty
