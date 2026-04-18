"""
Scripted greedy-builder policy for visualization / BC-teacher purposes.

Logic per micro-step:
  1. If HQ HP < 70% MHP → repair HQ from a connected neighbor (if any).
  2. Else if any upgrade point available → buy next upgrade in sequence.
  3. Else if adjacent free cell exists → build on it (first unused author).
  4. Else → END_TURN.

This is intentionally simple. On small maps it saturates at ~30 plantations, which
demonstrates achievable score (≥150K on stage 2).
"""
from __future__ import annotations

import numpy as np
from simulator.env.datssol_env import DatssolEnv
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.env.action_space import (
    ActionKind, MAX_PLANT, TARGET_OFFSET_DIM, delta_to_target_offset,
)
from simulator.engine.types import UpgradeName


UPGRADE_SEQUENCE = [
    UpgradeName.SETTLEMENT_LIMIT,   # more plantations = more score
    UpgradeName.REPAIR_POWER,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.MAX_HP,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.REPAIR_POWER,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.BEAVER_DAMAGE_MITIGATION,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.REPAIR_POWER,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.MAX_HP,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.EARTHQUAKE_MITIGATION,
    UpgradeName.SETTLEMENT_LIMIT,
]


class ScriptedGreedyBuilder:
    def __init__(self):
        self.upgrade_idx = 0

    def reset(self):
        self.upgrade_idx = 0

    def act(self, env: FlattenActionEnv, used_authors: set, relocated: bool) -> tuple[np.ndarray, bool, str]:
        """Return (flat-action, ends_turn_flag, description)."""
        state = env.env.state
        own_ids = sorted(
            pid for pid in state.plantations.keys() if pid not in state.enemy_ids
        )
        mhp = state.effective_mhp()

        # 0) Relocate HQ if its cell is close to 100% terraformation (prevents wipe)
        hq = state.get_hq()
        if hq is not None and not relocated:
            cell = state.cells.get(hq.position)
            progress = cell.progress if cell is not None else 0
            # Must relocate when next tick would terraform ≥100%
            if progress + state.config.ts >= 100:
                pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nb = (hq.position[0] + dx, hq.position[1] + dy)
                    if nb in pos_to_id and pos_to_id[nb] in state.connected_ids and nb != hq.position:
                        return _relocate(), False, f"relocate HQ -> {nb}"

        # 1) Repair HQ if damaged (skipped — HQ can't repair itself; needs helpers)
        if hq is not None and hq.hp < 0.7 * mhp:
            for pid in own_ids:
                if pid == hq.id or pid in used_authors:
                    continue
                p = state.plantations[pid]
                # Adjacent to HQ?
                dx = hq.position[0] - p.position[0]
                dy = hq.position[1] - p.position[1]
                if abs(dx) <= 2 and abs(dy) <= 2:
                    idx = own_ids.index(pid)
                    if idx < MAX_PLANT:
                        return _command(idx, dx, dy), False, f"repair HQ from {pid}"

        # 2) Buy upgrade if points available
        if state.upgrades.points > 0 and self.upgrade_idx < len(UPGRADE_SEQUENCE):
            upg = UPGRADE_SEQUENCE[self.upgrade_idx]
            tier = state.upgrades.tiers.get(upg)
            if tier is not None and tier.current < tier.max:
                all_upgrades = list(UpgradeName)
                upg_id = all_upgrades.index(upg)
                self.upgrade_idx += 1
                return _upgrade(upg_id), False, f"upgrade {upg.value}"
            self.upgrade_idx += 1  # skip maxed

        # 3) Build — priority:
        #    (a) finish existing construction reachable from any free author
        #    (b) start new build on adjacent free cell
        w, h = state.config.map_width, state.config.map_height
        pos_to_id = {p.position: pid for pid, p in state.plantations.items()}

        # (a) Direct existing constructions
        for idx, pid in enumerate(own_ids):
            if idx >= MAX_PLANT or pid in used_authors or pid not in state.connected_ids:
                continue
            p = state.plantations[pid]
            for (cx, cy), con in state.constructions.items():
                dx, dy = cx - p.position[0], cy - p.position[1]
                if abs(dx) <= state.config.ar and abs(dy) <= state.config.ar and (dx or dy):
                    return _command(idx, dx, dy), False, f"finish constr {pid}->({cx},{cy})"

        # (b) Start new construction
        for idx, pid in enumerate(own_ids):
            if idx >= MAX_PLANT or pid in used_authors or pid not in state.connected_ids:
                continue
            p = state.plantations[pid]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                tx, ty = p.position[0] + dx, p.position[1] + dy
                if not (0 <= tx < w and 0 <= ty < h):
                    continue
                if (tx, ty) in state.mountains:
                    continue
                if (tx, ty) in pos_to_id:
                    continue
                if (tx, ty) in state.constructions:
                    continue
                if (tx, ty) in state.cells and state.cells[(tx, ty)].progress >= 100:
                    continue
                if any(b.position == (tx, ty) for b in state.beavers.values()):
                    continue
                return _command(idx, dx, dy), False, f"build from {pid} -> ({tx},{ty})"

        # 4) End turn
        return np.array([int(ActionKind.END_TURN), 0, 0, 0, 0], dtype=np.int64), True, "end-turn"


def _command(author_idx: int, dx: int, dy: int) -> np.ndarray:
    return np.array([
        int(ActionKind.COMMAND),
        author_idx,
        author_idx,   # relay = author
        delta_to_target_offset(dx, dy),
        0,
    ], dtype=np.int64)


def _upgrade(upgrade_id: int) -> np.ndarray:
    return np.array([
        int(ActionKind.SET_UPGRADE),
        0, 0, 0,
        upgrade_id,
    ], dtype=np.int64)


def _relocate() -> np.ndarray:
    return np.array([
        int(ActionKind.SET_RELOCATE),
        0, 0, 0, 0,
    ], dtype=np.int64)


def run_scripted_episode(
    map_size: tuple[int, int],
    round_turns: int,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[int, list]:
    """Run one scripted episode, return (final_score, [per-turn score log])."""
    base = DatssolEnv(seed=seed, map_size=map_size, round_turns=round_turns)
    env = FlattenActionEnv(base)
    policy = ScriptedGreedyBuilder()
    obs, _ = env.reset()
    policy.reset()

    score_log = [env.env.state.score]
    used_authors: set = set()
    relocated = False
    done = False
    while not done:
        action, ends_turn, _desc = policy.act(env, used_authors, relocated)
        # Track authors used in this turn (reset on end-turn)
        if action[0] == int(ActionKind.COMMAND):
            used_authors.add(sorted(
                pid for pid in env.env.state.plantations
                if pid not in env.env.state.enemy_ids
            )[action[1]])
        if action[0] == int(ActionKind.SET_RELOCATE):
            relocated = True
        if ends_turn:
            used_authors = set()
            relocated = False
        obs, _r, term, trunc, _ = env.step(action)
        if ends_turn:
            score_log.append(env.env.state.score)
            if verbose and env.env.state.turn_no % 50 == 0:
                print(f"turn {env.env.state.turn_no}: score={env.env.state.score}, "
                      f"plants={len(env.env.state.plantations)}")
        done = term or trunc
    return env.env.state.score, score_log


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-size", type=int, nargs=2, default=[100, 100])
    ap.add_argument("--round-turns", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    final, log = run_scripted_episode(
        tuple(args.map_size), args.round_turns, args.seed, args.verbose
    )
    print(f"\n=== FINAL SCORE: {final:,} ===")
    print(f"Score progression (every ~50 turns): "
          f"{[log[i] for i in range(0, len(log), max(1, len(log)//10))]}")
