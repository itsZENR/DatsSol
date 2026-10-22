"""Observation-parity: compare obs from api_to_state, db_loader, env.reset.

Takes one live arena snapshot (if available) and one dump snapshot at a
similar turn, plus a fresh sim state, builds observations for all three,
and prints field-by-field deltas.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from simulator.env.obs_encoder import (
    encode_observation, GRID_CHANNELS, MAX_PLANT, MAX_BEAVER, MAX_ENEMY
)
from simulator.env.datssol_env import DatssolEnv
from simulator.env.action_space import ActionKind
from simulator.parity.db_loader import load_state_at
from simulator.engine.rules import tick
from simulator.engine.commands import CommandSet
from simulator.engine.spatial import bfs_connected
from simulator.inference.adapter import api_to_state


def summarize(name: str, obs: dict, state_info: dict):
    print(f"\n====== {name} ======")
    print(f"  state: {state_info}")
    print(f"  self_mask sum:   {int(obs['self_mask'].sum())}")
    print(f"  enemy_mask sum:  {int(obs['enemy_mask'].sum())}")
    print(f"  beaver_mask sum: {int(obs['beaver_mask'].sum())}")
    print(f"  storm_mask sum:  {int(obs['storm_mask'].sum())}")
    print(f"  global: turn_frac={obs['global'][0]:.3f} "
          f"score_scaled={obs['global'][12]:.3f} "
          f"hq_hp={obs['global'][10]:.3f} "
          f"n_plants_frac={obs['global'][11]:.3f} "
          f"eq={obs['global'][8]:.3f}")
    g = obs['grid']
    print(f"  grid nonzero per channel (31x31):")
    chans = ["mountains", "own", "construction", "enemy", "beaver",
             "cells_progress", "boosted", "ch7", "ch8", "ch9"]
    for i, lbl in enumerate(chans):
        nz = int((g[i] > 0).sum())
        val = g[i].max() if nz else 0
        print(f"    [{i}] {lbl}: nz={nz} max={val:.2f}")

    # First plantation in self_arr
    if obs['self_mask'].sum() > 0:
        s = obs['self'][0]
        print(f"  self[0]: dx={s[0]:.2f} dy={s[1]:.2f} hp={s[2]:.2f} "
              f"is_main={s[3]:.0f} iso={s[4]:.0f} "
              f"immunity={s[5]:.2f} boosted={s[6]:.0f} "
              f"cell_prog={s[7]:.2f} age={s[8]:.3f} "
              f"oldest={s[9]:.0f} avail={s[12]:.0f}")


def try_live_obs():
    try:
        import requests
        from config import TOKEN, BASE_URL
        r = requests.get(f"{BASE_URL}/api/arena",
                         headers={"X-Auth-Token": TOKEN}, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[skip live: {e}]")
        return None, None

    state = api_to_state(data)
    own_pos = {pid: p.position for pid, p in state.plantations.items()
               if pid not in state.enemy_ids}
    state.connected_ids = bfs_connected(state.hq_id, own_pos)
    obs = encode_observation(state, set(), vision_aware=True)
    info = {
        "turn": state.turn_no, "score": state.score,
        "plants": len(state.plantations),
        "mountains": len(state.mountains),
        "beavers": len(state.beavers),
        "cells": len(state.cells),
    }
    return obs, info


def dump_obs(dump_dir: Path, round_id: int, turn_no: int):
    state = load_state_at(dump_dir, round_id, turn_no)
    obs = encode_observation(state, set())
    info = {
        "turn": state.turn_no, "score": state.score,
        "plants": len(state.plantations),
        "mountains": len(state.mountains),
        "beavers": len(state.beavers),
        "cells": len(state.cells),
    }
    return obs, info


def sim_obs(map_size, round_turns, ticks: int):
    env = DatssolEnv(seed=42, map_size=map_size, round_turns=round_turns,
                     game_config_overrides={"initial_beavers": 30,
                                            "initial_mountain_density": 0.02},
                     vision_aware=True)
    obs, info = env.reset()
    # Run a few empty ticks to let state warm up.
    for _ in range(ticks):
        env.state = tick(env.state, CommandSet())
    obs = encode_observation(env.state, set(), vision_aware=True)
    st = env.state
    info = {
        "turn": st.turn_no, "score": st.score,
        "plants": len(st.plantations),
        "mountains": len(st.mountains),
        "beavers": len(st.beavers),
        "cells": len(st.cells),
    }
    return obs, info


def main():
    dump_dir = Path("dump")
    round_id = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    turn_no = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print(f"--- dump round_id={round_id} turn_no={turn_no} ---")
    dobs, dinfo = dump_obs(dump_dir, round_id, turn_no)
    summarize("DUMP", dobs, dinfo)

    print("\n--- sim env.reset + 10 ticks ---")
    sobs, sinfo = sim_obs(map_size=(378, 378), round_turns=600, ticks=10)
    summarize("SIM", sobs, sinfo)

    print("\n--- live /api/arena ---")
    lobs, linfo = try_live_obs()
    if lobs is not None:
        summarize("LIVE", lobs, linfo)


if __name__ == "__main__":
    main()
