from dataclasses import dataclass, field
import numpy as np
from simulator.engine.state import EngineState
from simulator.env.action_space import (
    ActionKind, MAX_PLANT, TARGET_OFFSET_DIM, N_UPGRADES,
)


@dataclass
class TurnContext:
    """Accumulated within a single game turn."""
    authors_used: set = field(default_factory=set)
    upgrade_set: bool = False
    relocate_set: bool = False
    micro_steps: int = 0

    def ordered_author_ids(self, state: EngineState) -> list:
        return sorted(state.connected_ids)


def build_action_mask(state: EngineState, ctx: TurnContext) -> dict:
    """Return mask arrays per Dict-space key. 1 = valid, 0 = invalid."""
    kind_mask = np.zeros(len(ActionKind), dtype=np.int8)
    author_mask = np.zeros(MAX_PLANT, dtype=np.int8)
    relay_mask = np.ones(MAX_PLANT, dtype=np.int8)
    target_mask = np.ones(TARGET_OFFSET_DIM, dtype=np.int8)
    upgrade_mask = np.zeros(N_UPGRADES, dtype=np.int8)

    # END_TURN always valid
    kind_mask[int(ActionKind.END_TURN)] = 1

    # SET_UPGRADE
    if not ctx.upgrade_set and state.upgrades.points > 0:
        kind_mask[int(ActionKind.SET_UPGRADE)] = 1
        from simulator.engine.types import UpgradeName
        upgrade_list = list(UpgradeName)
        for i, u in enumerate(upgrade_list[:N_UPGRADES]):
            t = state.upgrades.tiers.get(u)
            if t and t.current < t.max:
                upgrade_mask[i] = 1

    # SET_RELOCATE — if HQ has orthogonally adjacent own connected neighbor
    if not ctx.relocate_set:
        hq = state.get_hq()
        if hq is not None:
            hq_pos = hq.position
            pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
            has_neighbor = any(
                (hq_pos[0] + dx, hq_pos[1] + dy) in pos_to_id and
                pos_to_id[(hq_pos[0] + dx, hq_pos[1] + dy)] in state.connected_ids
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            )
            if has_neighbor:
                kind_mask[int(ActionKind.SET_RELOCATE)] = 1

    # COMMAND — at least one unused author exists
    author_ids = ctx.ordered_author_ids(state)
    has_author = False
    for i, pid in enumerate(author_ids[:MAX_PLANT]):
        if pid not in ctx.authors_used:
            author_mask[i] = 1
            has_author = True
    if has_author:
        kind_mask[int(ActionKind.COMMAND)] = 1

    # NOOP always valid (allows "skip")
    kind_mask[int(ActionKind.NOOP)] = 1

    return {
        "kind": kind_mask,
        "author_idx": author_mask,
        "relay_idx": relay_mask,
        "target_offset": target_mask,
        "upgrade_id": upgrade_mask,
    }
