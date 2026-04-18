"""Render a GIF of the Go-ported scripted strategy in action (high-score demo)."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation

from simulator.env.datssol_env import DatssolEnv
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.env.action_space import ActionKind
from simulator.visualize.scripted_go import ScriptedGo
from simulator.visualize.render import _snapshot, _draw_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-size", type=int, nargs=2, default=[30, 30])
    ap.add_argument("--round-turns", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ez", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("go_scripted_demo.gif"))
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    overrides = {"ds": 0, "cs": 10} if args.ez else {}
    base = DatssolEnv(seed=args.seed, map_size=tuple(args.map_size),
                     round_turns=args.round_turns,
                     game_config_overrides=overrides)
    env = FlattenActionEnv(base)
    policy = ScriptedGo()

    obs, _ = env.reset()
    policy.reset()
    snapshots = [_snapshot(env.env.state)]
    used: set = set()
    relocated = False
    first = True
    done = False

    while not done:
        action, ends_turn = policy.act(env, used, relocated, first)
        first = False
        if action[0] == int(ActionKind.COMMAND):
            own_ids = sorted(pid for pid in env.env.state.plantations
                           if pid not in env.env.state.enemy_ids)
            if action[1] < len(own_ids):
                used.add(own_ids[action[1]])
        if action[0] == int(ActionKind.SET_RELOCATE):
            relocated = True
        if ends_turn:
            used = set()
            relocated = False
            first = True
        obs, _r, term, trunc, _ = env.step(action)
        if ends_turn:
            snapshots.append(_snapshot(env.env.state))
        done = term or trunc

    print(f"collected {len(snapshots)} frames, final score={env.env.state.score}")

    fig, ax = plt.subplots(figsize=(7, 7))
    mhp = base.state.effective_mhp() if base.state else 50
    boosted_mod = base.state.config.boosted_mod if base.state else 7

    def update(i):
        _draw_frame(ax, snapshots[i], mhp=mhp, boosted_mod=boosted_mod)
        return []

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // args.fps)
    writer = PillowWriter(fps=args.fps)
    anim.save(str(args.out), writer=writer)
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
