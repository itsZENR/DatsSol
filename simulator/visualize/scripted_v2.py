"""
Improved scripted teacher v2 — shared build target across plantations.

Key heuristic (ported from Go conveyor):
- Maintain ONE active build target; all free plantations build on it until completion.
- Rotate HQ proactively when own cell progress >= 80%.
- Build expansion prefers cells adjacent to existing plantations (stays compact).
- After completion, pick NEW target far from current HQ to expand.
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
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.REPAIR_POWER,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.MAX_HP,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.REPAIR_POWER,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.MAX_HP,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.REPAIR_POWER,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.MAX_HP,
    UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.MAX_HP,
    UpgradeName.SETTLEMENT_LIMIT,
]


class ScriptedV2:
    def __init__(self):
        self.upgrade_idx = 0
        self.current_target: tuple[int, int] | None = None
        self.target_age = 0

    def reset(self):
        self.upgrade_idx = 0
        self.current_target = None
        self.target_age = 0

    def _pick_new_target(self, state) -> tuple[int, int] | None:
        """Find best empty cell adjacent to any connected plantation, preferring
        boosted cells and cells away from HQ for spread."""
        w, h = state.config.map_width, state.config.map_height
        pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
        hq = state.get_hq()
        if hq is None:
            return None
        hq_pos = hq.position

        candidates: list[tuple[float, tuple[int, int]]] = []
        for pid in state.connected_ids:
            if pid in state.enemy_ids:
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
                # Respect: no build on 100% cells
                if (tx, ty) in state.cells and state.cells[(tx, ty)].progress >= 100:
                    continue
                if any(b.position == (tx, ty) for b in state.beavers.values()):
                    continue
                # Score: boosted cells + manhattan distance from HQ (spread)
                score = 0.0
                if state.is_boosted_cell((tx, ty)):
                    score += 100.0
                score += abs(tx - hq_pos[0]) + abs(ty - hq_pos[1])
                candidates.append((score, (tx, ty)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    def act(self, env: FlattenActionEnv, used_authors: set, relocated: bool) -> tuple[np.ndarray, bool, str]:
        state = env.env.state
        own_ids = sorted(
            pid for pid in state.plantations.keys() if pid not in state.enemy_ids
        )
        mhp = state.effective_mhp()
        ar = state.config.ar

        # 0) Relocate HQ if cell progress >= 80% (earlier than just "next tick hits 100")
        hq = state.get_hq()
        if hq is not None and not relocated:
            cell = state.cells.get(hq.position)
            progress = cell.progress if cell is not None else 0
            if progress >= 80:
                pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
                # Pick neighbor with lowest cell progress
                best_nb = None
                best_prog = 999
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nb = (hq.position[0] + dx, hq.position[1] + dy)
                    if nb not in pos_to_id or pos_to_id[nb] not in state.connected_ids:
                        continue
                    if nb == hq.position:
                        continue
                    nb_cell = state.cells.get(nb)
                    nb_prog = nb_cell.progress if nb_cell else 0
                    if nb_prog < best_prog:
                        best_prog = nb_prog
                        best_nb = nb
                if best_nb is not None:
                    return _relocate(), False, f"relocate HQ→{best_nb}"

        # 1) Upgrade if points
        if state.upgrades.points > 0 and self.upgrade_idx < len(UPGRADE_SEQUENCE):
            upg = UPGRADE_SEQUENCE[self.upgrade_idx]
            tier = state.upgrades.tiers.get(upg)
            if tier is not None and tier.current < tier.max:
                all_upgrades = list(UpgradeName)
                upg_id = all_upgrades.index(upg)
                self.upgrade_idx += 1
                return _upgrade(upg_id), False, f"upgrade {upg.value}"
            self.upgrade_idx += 1

        # 2) Check/update shared build target
        # Validate current target
        if self.current_target is not None:
            pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
            still_valid = (
                self.current_target not in pos_to_id and
                self.current_target not in state.mountains and
                not (self.current_target in state.cells and
                     state.cells[self.current_target].progress >= 100)
            )
            if not still_valid:
                self.current_target = None
                self.target_age = 0

        if self.current_target is None:
            self.current_target = self._pick_new_target(state)
            self.target_age = 0

        # 3) All free builders push current target
        if self.current_target is not None:
            for idx, pid in enumerate(own_ids):
                if idx >= MAX_PLANT or pid in used_authors or pid not in state.connected_ids:
                    continue
                p = state.plantations[pid]
                dx = self.current_target[0] - p.position[0]
                dy = self.current_target[1] - p.position[1]
                if abs(dx) <= ar and abs(dy) <= ar and (dx or dy):
                    return _command(idx, dx, dy), False, f"build→{self.current_target}"

        # 4) End turn. Track target age — if no construction at target for 3+ turns, drop.
        return np.array([int(ActionKind.END_TURN), 0, 0, 0, 0], dtype=np.int64), True, "end-turn"


def _command(author_idx: int, dx: int, dy: int) -> np.ndarray:
    return np.array([
        int(ActionKind.COMMAND),
        author_idx, author_idx,
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


def run_episode(map_size: tuple[int, int], round_turns: int, seed: int = 0,
                game_config_overrides: dict | None = None) -> int:
    base = DatssolEnv(seed=seed, map_size=map_size, round_turns=round_turns,
                     game_config_overrides=game_config_overrides or {})
    env = FlattenActionEnv(base)
    policy = ScriptedV2()
    obs, _ = env.reset()
    policy.reset()
    used: set = set()
    relocated = False
    done = False
    last_turn_score = 0
    while not done:
        action, ends_turn, _ = policy.act(env, used, relocated)
        if action[0] == int(ActionKind.COMMAND):
            own_ids = sorted(
                pid for pid in env.env.state.plantations
                if pid not in env.env.state.enemy_ids
            )
            if action[1] < len(own_ids):
                used.add(own_ids[action[1]])
        if action[0] == int(ActionKind.SET_RELOCATE):
            relocated = True
        if ends_turn:
            # Track target age based on progress
            if policy.current_target is not None:
                if policy.current_target in env.env.state.constructions:
                    con = env.env.state.constructions[policy.current_target]
                    if con.progress > policy.target_age:
                        policy.target_age = con.progress
                    else:
                        # No progress this turn — drop target
                        policy.current_target = None
            used = set()
            relocated = False
        obs, _r, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.env.state.score


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-size", type=int, nargs=2, default=[30, 30])
    ap.add_argument("--round-turns", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--ez", action="store_true", help="Use stage 0_ez overrides (ds=0, cs=10)")
    args = ap.parse_args()
    overrides = {"ds": 0, "cs": 10} if args.ez else {}
    scores = []
    for i in range(args.episodes):
        s = run_episode(tuple(args.map_size), args.round_turns, seed=i,
                       game_config_overrides=overrides)
        scores.append(s)
        print(f"ep {i}: score={s}")
    print(f"\nMean: {sum(scores)/len(scores):.0f}  Max: {max(scores)}")
