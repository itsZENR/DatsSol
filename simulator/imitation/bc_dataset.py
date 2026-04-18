"""
Builds (obs, micro_action) training pairs from the dump.
For each turn:
  1. Load state at turn T.
  2. Load all recorded commands at turn T.
  3. Generate micro-action sequence:
     - for each command (ordered by author position): COMMAND micro-action
     - finally END_TURN micro-action
  4. Encode obs at each step.
"""
from pathlib import Path
from typing import Iterator

from simulator.parity.db_loader import load_state_at
from simulator.parity.action_loader import load_actions_for_turn
from simulator.env.obs_encoder import encode_observation
from simulator.env.action_space import ActionKind, MAX_PLANT, delta_to_target_offset


def build_bc_samples(
    dump: Path,
    round_ids: list[int],
    max_turns: int | None = None,
) -> Iterator[tuple[dict, dict]]:
    for rid in round_ids:
        t = 0
        while True:
            try:
                state = load_state_at(dump, rid, t)
            except ValueError:
                break
            cmds = load_actions_for_turn(dump, rid, t)

            sorted_cmds = sorted(
                cmds.commands,
                key=lambda c: (c.author[0], c.author[1]),
            )

            own_ids = sorted(
                pid for pid in state.plantations.keys()
                if pid not in state.enemy_ids
            )
            pos_to_idx = {
                state.plantations[pid].position: i
                for i, pid in enumerate(own_ids[:MAX_PLANT])
            }

            for cmd in sorted_cmds:
                obs = encode_observation(state)
                if cmd.author not in pos_to_idx:
                    continue
                dx = cmd.target[0] - cmd.relay[0]
                dy = cmd.target[1] - cmd.relay[1]
                if abs(dx) > 2 or abs(dy) > 2:
                    continue
                action = {
                    "kind": int(ActionKind.COMMAND),
                    "author_idx": pos_to_idx[cmd.author],
                    "relay_idx": pos_to_idx[cmd.author],
                    "target_offset": delta_to_target_offset(dx, dy),
                    "upgrade_id": 0,
                }
                yield (obs, action)

            # End-turn micro-action
            obs = encode_observation(state)
            yield (obs, {
                "kind": int(ActionKind.END_TURN),
                "author_idx": 0, "relay_idx": 0,
                "target_offset": 0, "upgrade_id": 0,
            })

            t += 1
            if max_turns is not None and t >= max_turns:
                break
