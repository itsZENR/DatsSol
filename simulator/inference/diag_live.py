"""Diagnostic: fetch arena, print state + per-step policy decisions."""
from __future__ import annotations

import json
import sys
import numpy as np
import requests

from config import TOKEN, BASE_URL
from simulator.inference.adapter import api_to_state
from simulator.inference.live_agent import _load_model, _flat_mask_from_ctx
from simulator.env.obs_encoder import encode_observation
from simulator.env.action_mask import TurnContext, build_action_mask
from simulator.env.action_space import ActionKind, target_offset_to_delta
from simulator.engine.spatial import bfs_connected


def main(model_path: str):
    print(f"Loading model: {model_path}")
    policy = _load_model(model_path)

    r = requests.get(f"{BASE_URL}/api/arena",
                     headers={"X-Auth-Token": TOKEN}, timeout=5)
    r.raise_for_status()
    data = r.json()
    print(f"arena turnNo={data.get('turnNo')} size={data.get('size')}")
    print(f"own plants: {len(data.get('plantations', []))}, "
          f"cells: {len(data.get('cells', []))}, "
          f"beavers: {len(data.get('beavers', []))}, "
          f"constructions: {len(data.get('construction', []))}")

    state = api_to_state(data)
    own_pos = {pid: p.position for pid, p in state.plantations.items()
               if pid not in state.enemy_ids}
    state.connected_ids = bfs_connected(state.hq_id, own_pos)
    print(f"hq_id={state.hq_id}")
    print(f"all own: {len(own_pos)}, connected: {len(state.connected_ids)}")
    print(f"connected positions: {[(pid[:8], state.plantations[pid].position) for pid in state.connected_ids]}")

    ctx = TurnContext()
    mask = build_action_mask(state, ctx)
    print(f"\nInitial mask kind={mask['kind'].tolist()}  "
          f"author_sum={int(mask['author_idx'].sum())}  "
          f"relay_sum={int(mask['relay_idx'].sum())}  "
          f"target_sum={int(mask['target_offset'].sum())}  "
          f"upgrade_sum={int(mask['upgrade_id'].sum())}")

    print("\n--- Micro-steps ---")
    for step in range(20):
        obs = encode_observation(state, ctx.authors_used)
        flat_mask = _flat_mask_from_ctx(state, ctx)
        action, _ = policy.predict(obs, action_masks=flat_mask, deterministic=True)
        action = np.asarray(action).flatten()
        kind = int(action[0])
        kname = ActionKind(kind).name if kind < len(ActionKind) else f"?{kind}"
        a, ri, to, ui = int(action[1]), int(action[2]), int(action[3]), int(action[4])
        print(f"  step {step}: {kname} author={a} relay={ri} tgt_off={to} upg={ui}")
        if kind == int(ActionKind.END_TURN):
            print("  -> END_TURN")
            break
        if kind == int(ActionKind.COMMAND):
            connected = sorted(state.connected_ids)
            if a < len(connected):
                aid = connected[a]
                if aid in ctx.authors_used:
                    print(f"    (author {aid[:8]} already used)")
                else:
                    ctx.authors_used.add(aid)
        elif kind == int(ActionKind.SET_UPGRADE):
            ctx.upgrade_set = True
        elif kind == int(ActionKind.SET_RELOCATE):
            ctx.relocate_set = True
        ctx.micro_steps += 1


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ppo_s1.zip")
