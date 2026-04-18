"""
Render an episode of a trained (or scripted) policy to frames + GIF.

Usage:
    python -m simulator.visualize.render --model ppo_seq_s0 \\
        --map-size 30 30 --round-turns 100 --out demo.gif

Frame content per turn:
- Grid: mountains (grey), own plantations (green gradient by HP), constructions (yellow),
  enemies (red), beavers (brown), terraformed cells (light green gradient),
  boosted cells highlighted.
- HUD line: turn_no, score, n_plantations, upgrade_points.
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import PillowWriter, FuncAnimation

from simulator.env.datssol_env import DatssolEnv
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.env.action_space import ActionKind


def _snapshot(state) -> dict:
    """Extract a renderable snapshot from EngineState."""
    own_positions = {
        p.position: p for pid, p in state.plantations.items()
        if pid not in state.enemy_ids
    }
    enemy_positions = {
        state.plantations[eid].position for eid in state.enemy_ids
    }
    beaver_positions = [b.position for b in state.beavers.values()]
    return dict(
        turn=state.turn_no,
        score=state.score,
        w=state.config.map_width,
        h=state.config.map_height,
        mountains=list(state.mountains),
        own=[(pos, p.hp) for pos, p in own_positions.items()],
        hq=state.hq_id,
        hq_pos=(state.get_hq().position if state.get_hq() else None),
        enemies=list(enemy_positions),
        beavers=beaver_positions,
        constructions=[(pos, c.progress) for pos, c in state.constructions.items()],
        cells=[(pos, c.progress) for pos, c in state.cells.items()],
        upgrade_points=state.upgrades.points,
    )


def _draw_frame(ax, snap: dict, mhp: int, boosted_mod: int = 7):
    w, h = snap["w"], snap["h"]
    ax.clear()
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f1e1")  # sand

    # Boosted cells grid hint
    for x in range(0, w, boosted_mod):
        for y in range(0, h, boosted_mod):
            ax.add_patch(Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                   facecolor="#ffe9a8", edgecolor="none", alpha=0.5))

    # Terraformed cells
    for (x, y), prog in snap["cells"]:
        alpha = min(1.0, 0.15 + prog / 130.0)
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1.0, 1.0,
                               facecolor="#6ca96c", edgecolor="none", alpha=alpha))

    # Mountains
    for (x, y) in snap["mountains"]:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1.0, 1.0,
                               facecolor="#6f6f6f", edgecolor="none"))

    # Constructions
    for (x, y), prog in snap["constructions"]:
        ax.add_patch(Rectangle((x - 0.35, y - 0.35), 0.7, 0.7,
                               facecolor="#e0c044",
                               edgecolor="#8a6c00", linewidth=0.5,
                               alpha=0.3 + 0.7 * (prog / 50)))

    # Own plantations
    for (x, y), hp in snap["own"]:
        frac = max(0.0, hp / mhp)
        color = (0.1, 0.55 + 0.35 * frac, 0.1)
        ax.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                               facecolor=color, edgecolor="#0a3d0a", linewidth=0.5))

    # HQ star
    if snap["hq_pos"] is not None:
        hx, hy = snap["hq_pos"]
        ax.plot([hx], [hy], marker="*", color="#fff",
                markeredgecolor="#0a3d0a", markersize=14)

    # Enemies
    for (x, y) in snap["enemies"]:
        ax.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                               facecolor="#c44", edgecolor="#4a0a0a", linewidth=0.5))

    # Beavers
    for (x, y) in snap["beavers"]:
        ax.plot([x], [y], marker="o", color="#8b4513",
                markeredgecolor="#000", markersize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"turn={snap['turn']}  score={snap['score']:,}  "
        f"plants={len(snap['own'])}  up_pts={snap['upgrade_points']}"
    )


def _load_policy(model_path: Optional[Path]):
    if model_path is None:
        return None
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(model_path)


def _random_masked_action(env: FlattenActionEnv, rng: np.random.Generator) -> np.ndarray:
    """Sample a random valid action with correlated sub-choices.
    Bias towards COMMAND actions and use author as relay (always valid self-relay)."""
    from simulator.env.action_space import ActionKind as _AK, MAX_PLANT, TARGET_OFFSET_DIM, N_UPGRADES
    nvec = env.NVEC
    mask = env.action_masks()
    offsets = np.concatenate([[0], np.cumsum(nvec)])
    kind_mask = mask[offsets[0]:offsets[1]]
    author_mask = mask[offsets[1]:offsets[2]]

    # Bias: if COMMAND is valid, pick it with 70% probability
    valid_kinds = np.flatnonzero(kind_mask)
    if int(_AK.COMMAND) in valid_kinds and rng.random() < 0.7:
        kind = int(_AK.COMMAND)
    else:
        kind = int(rng.choice(valid_kinds))

    out = np.zeros(len(nvec), dtype=np.int64)
    out[0] = kind

    if kind == int(_AK.COMMAND):
        valid_authors = np.flatnonzero(author_mask)
        if len(valid_authors) == 0:
            out[0] = int(_AK.END_TURN)
            return out
        a = int(rng.choice(valid_authors))
        out[1] = a
        out[2] = a  # relay = author (always-valid self-relay)
        # Pick any target offset in AR range (all 25 are geometrically valid;
        # engine will reject if hits own plant/mountain/beaver but often yields a build)
        out[3] = int(rng.integers(0, TARGET_OFFSET_DIM))
        out[4] = 0
    elif kind == int(_AK.SET_UPGRADE):
        upg_mask = mask[offsets[4]:offsets[5]]
        valid_u = np.flatnonzero(upg_mask)
        out[4] = int(rng.choice(valid_u)) if len(valid_u) else 0
    # END_TURN/NOOP/SET_RELOCATE — sub-indices don't matter
    return out


def _act(env: FlattenActionEnv, policy, obs, rng: np.random.Generator | None = None) -> np.ndarray:
    if policy == "random":
        return _random_masked_action(env, rng)
    if policy is None:
        # Scripted "end-turn always" fallback
        return np.array([int(ActionKind.END_TURN), 0, 0, 0, 0], dtype=np.int64)
    masks = env.action_masks()
    action, _ = policy.predict(obs, action_masks=masks, deterministic=False)
    return action


def render_episode(
    model_path: Optional[Path],
    map_size: tuple[int, int],
    round_turns: int,
    seed: int,
    out: Path,
    fps: int = 6,
    random_policy: bool = False,
    stage: Optional[str] = None,
):
    if stage is not None:
        from simulator.train.curriculum import make_env_for_stage
        s = int(stage) if stage.isdigit() else stage
        base = make_env_for_stage(s, seed=seed)
    else:
        base = DatssolEnv(seed=seed, map_size=map_size, round_turns=round_turns)
    env = FlattenActionEnv(base)
    policy = "random" if random_policy else _load_policy(model_path)
    rng = np.random.default_rng(seed)

    obs, _ = env.reset()
    snapshots = [_snapshot(env.env.state)]
    micro = 0
    done = False
    while not done:
        action = _act(env, policy, obs, rng)
        obs, r, term, trunc, _ = env.step(action)
        micro += 1
        # Snapshot only after turn advances (end-turn executed)
        if micro >= 1 and (action[0] == int(ActionKind.END_TURN) or env.env.state.turn_no != snapshots[-1]["turn"]):
            snapshots.append(_snapshot(env.env.state))
            micro = 0
        done = term or trunc

    print(f"collected {len(snapshots)} snapshots (final score={env.env.state.score})")

    fig, ax = plt.subplots(figsize=(7, 7))
    mhp = base.state.effective_mhp() if base.state else 50
    boosted_mod = base.state.config.boosted_mod if base.state else 7

    def update(i):
        _draw_frame(ax, snapshots[i], mhp=mhp, boosted_mod=boosted_mod)
        return []

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // fps)
    writer = PillowWriter(fps=fps)
    anim.save(str(out), writer=writer)
    plt.close(fig)
    print(f"Saved animation: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--map-size", type=int, nargs=2, default=[30, 30])
    ap.add_argument("--round-turns", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("demo.gif"))
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--random", action="store_true",
                    help="Use random valid actions (no model required)")
    ap.add_argument("--stage", type=str, default=None,
                    help="Use stage config (e.g. 0_ez) — overrides map-size/turns")
    args = ap.parse_args()

    render_episode(
        model_path=args.model,
        map_size=tuple(args.map_size),
        round_turns=args.round_turns,
        seed=args.seed,
        out=args.out,
        fps=args.fps,
        random_policy=args.random,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
