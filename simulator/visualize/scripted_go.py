"""
Python port of the Go "conveyor" strategy (visual/sdfsdfs-main/strategy/conveyor.go).

Core heuristics:
- Upgrade sequence: signal_range x3, vision_range, decay_mitigation x3, settlement_limit x6, max_hp x2
- CU rotation: opportunistic at 30%+ if fresher neighbor (-20% gap), emergency at 80%+
- tryContinueBuild: finish closest-to-done; adaptive maxPerConstruction
- tryNewBuild: scoring by cluster centroid, CU proximity, neighbor density, boosted bonus, edge penalty
- Sticky build target: prefer last-turn target if still valid

Simplifications for stage0_ez: no beavers, no meteo, no enemies, no sabotage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from simulator.env.datssol_env import DatssolEnv
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.env.action_space import (
    ActionKind, MAX_PLANT, delta_to_target_offset,
)
from simulator.engine.types import UpgradeName


UPGRADE_SEQUENCE = [
    UpgradeName.SIGNAL_RANGE, UpgradeName.SIGNAL_RANGE, UpgradeName.SIGNAL_RANGE,
    UpgradeName.VISION_RANGE,
    UpgradeName.DECAY_MITIGATION, UpgradeName.DECAY_MITIGATION, UpgradeName.DECAY_MITIGATION,
    UpgradeName.SETTLEMENT_LIMIT, UpgradeName.SETTLEMENT_LIMIT, UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.SETTLEMENT_LIMIT, UpgradeName.SETTLEMENT_LIMIT, UpgradeName.SETTLEMENT_LIMIT,
    UpgradeName.MAX_HP, UpgradeName.MAX_HP,
]


def _cardinal():
    return [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _nearest_boosted_distance(pos):
    rx = pos[0] % 7
    if rx > 3: rx = 7 - rx
    ry = pos[1] % 7
    if ry > 3: ry = 7 - ry
    return rx + ry


@dataclass
class ScriptedGo:
    last_build_target: tuple | None = None
    assigned_builds: dict = field(default_factory=dict)
    new_builds_this_turn: int = 0

    def reset(self):
        self.last_build_target = None
        self.assigned_builds = {}
        self.new_builds_this_turn = 0

    def _pick_upgrade(self, state):
        ups = state.upgrades
        if ups.points <= 0:
            return None
        total = sum(t.current for t in ups.tiers.values())
        all_upgrades = list(UpgradeName)
        for idx in range(total, len(UPGRADE_SEQUENCE)):
            name = UPGRADE_SEQUENCE[idx]
            tier = ups.tiers.get(name)
            if tier is not None and (tier.max < 0 or tier.current < tier.max):
                return all_upgrades.index(name)
        return None

    def _maybe_relocate(self, state):
        hq = state.get_hq()
        if hq is None:
            return None
        cell = state.cells.get(hq.position)
        progress = cell.progress if cell else 0
        if progress < 30:
            return None
        pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
        own_positions = [p.position for pid, p in state.plantations.items() if pid not in state.enemy_ids]
        if not own_positions:
            return None
        cx = sum(p[0] for p in own_positions) // len(own_positions)
        cy = sum(p[1] for p in own_positions) // len(own_positions)
        candidates = []
        for dx, dy in _cardinal():
            adj = (hq.position[0] + dx, hq.position[1] + dy)
            apid = pos_to_id.get(adj)
            if apid is None or apid in state.enemy_ids or apid == state.hq_id:
                continue
            if apid not in state.connected_ids:
                continue
            aprog = state.cells.get(adj).progress if adj in state.cells else 0
            score = progress - aprog
            cur_d = abs(hq.position[0] - cx) + abs(hq.position[1] - cy)
            cand_d = abs(adj[0] - cx) + abs(adj[1] - cy)
            if cand_d < cur_d:
                score += 10
            candidates.append((score, aprog, adj))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best_score, best_prog, best_pos = candidates[0]
        if progress >= 80:
            if best_prog >= progress:
                return None
            return best_pos
        if best_prog > progress - 20:
            return None
        return best_pos

    def _try_continue_build(self, state, author, ar):
        constructions = list(state.constructions.values())
        if not constructions:
            return None
        connected_count = len(state.connected_ids)
        max_per = max(3, (connected_count + len(constructions) - 1) // len(constructions))
        ranked = []
        for con in constructions:
            dx = con.position[0] - author.position[0]
            dy = con.position[1] - author.position[1]
            if abs(dx) > ar or abs(dy) > ar:
                continue
            if con.progress >= 50:
                continue
            remaining = 50 - con.progress
            turns_with_3 = (remaining + 14) // 15
            if connected_count > 5 and turns_with_3 > 5 and con.progress <= 5:
                continue
            ranked.append((con.progress, con.position))
        ranked.sort(key=lambda x: -x[0])
        for prog, pos in ranked:
            if self.assigned_builds.get(pos, 0) >= max_per:
                continue
            self.assigned_builds[pos] = self.assigned_builds.get(pos, 0) + 1
            return pos
        return None

    def _try_new_build(self, state, author, ar):
        w, h = state.config.map_width, state.config.map_height
        pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
        hq = state.get_hq()
        if hq is None:
            return None
        own_positions = [p.position for pid, p in state.plantations.items() if pid not in state.enemy_ids]
        cx = sum(p[0] for p in own_positions) // max(1, len(own_positions))
        cy = sum(p[1] for p in own_positions) // max(1, len(own_positions))
        constructing = set(state.constructions.keys())
        occupied = set(pos_to_id.keys()) | constructing
        best_pos = None
        best_score = 0
        for dx in range(-ar, ar + 1):
            for dy in range(-ar, ar + 1):
                target = (author.position[0] + dx, author.position[1] + dy)
                if not (0 <= target[0] < w and 0 <= target[1] < h):
                    continue
                if target in occupied or target in state.mountains:
                    continue
                prog = state.cells.get(target).progress if target in state.cells else 0
                if prog >= 100:
                    continue
                turns_alive = (100 - prog) // 5
                if prog > 0 and turns_alive < 6:
                    continue
                adj_plants = 0
                adj_cons = 0
                has_living = False
                adj_to_cu = False
                for adx, ady in _cardinal():
                    adj = (target[0] + adx, target[1] + ady)
                    apid = pos_to_id.get(adj)
                    if apid is not None:
                        adj_plants += 1
                        aprog = state.cells.get(adj).progress if adj in state.cells else 0
                        ttl = (100 - aprog) // 5
                        if apid == state.hq_id:
                            adj_to_cu = True
                            if aprog < 50:
                                has_living = True
                        elif ttl >= 10:
                            has_living = True
                    elif adj in constructing:
                        adj_cons += 1
                        has_living = True
                adj_total = adj_plants + adj_cons
                if adj_total == 0:
                    continue
                if not has_living and len(own_positions) > 1:
                    continue
                score = 100
                d_centroid = abs(target[0] - cx) + abs(target[1] - cy)
                score += 300 - d_centroid * 30
                d_cu = abs(target[0] - hq.position[0]) + abs(target[1] - hq.position[1])
                score += 200 - d_cu * 20
                if adj_to_cu:
                    score += 150
                score += adj_plants * 100
                score += adj_cons * 30
                if prog > 0:
                    score -= prog * 2
                if state.is_boosted_cell(target):
                    score += 100
                nbd = _nearest_boosted_distance(target)
                if nbd <= 3:
                    score += 40
                edge_d = min(target[0], w - 1 - target[0], target[1], h - 1 - target[1])
                if edge_d < 10:
                    score -= (10 - edge_d) * 30
                if score > 0 and score > best_score:
                    best_score = score
                    best_pos = target
        if self.last_build_target is not None and best_pos is not None:
            lp = self.last_build_target
            if lp != best_pos:
                if (lp not in occupied and lp not in state.mountains and
                    abs(lp[0] - author.position[0]) <= ar and
                    abs(lp[1] - author.position[1]) <= ar):
                    prog = state.cells.get(lp).progress if lp in state.cells else 0
                    if prog < 100:
                        ttl = (100 - prog) // 5
                        if prog == 0 or ttl >= 6:
                            best_pos = lp
        if best_pos is None:
            self.last_build_target = None
            return None
        self.last_build_target = best_pos
        return best_pos

    def act(self, env, used_authors, relocated, is_first_micro_of_turn):
        state = env.env.state
        own_ids = sorted(pid for pid in state.plantations.keys() if pid not in state.enemy_ids)
        ar = state.config.ar
        if is_first_micro_of_turn:
            self.assigned_builds = {}
            self.new_builds_this_turn = 0
        if not relocated:
            dst = self._maybe_relocate(state)
            if dst is not None:
                return _relocate(), False
        upg_id = self._pick_upgrade(state)
        if upg_id is not None:
            return _upgrade(upg_id), False
        connected_plants = [
            (idx, pid) for idx, pid in enumerate(own_ids[:MAX_PLANT])
            if pid in state.connected_ids and pid not in used_authors
        ]
        active_cons = len(state.constructions)
        free_plants = len(connected_plants)
        conn_total = len([p for p in own_ids if p in state.connected_ids])
        for idx, pid in connected_plants:
            p = state.plantations[pid]
            t = self._try_continue_build(state, p, ar)
            if t is not None:
                return _command(idx, t[0] - p.position[0], t[1] - p.position[1]), False
            if active_cons == 0:
                max_new = free_plants
            elif conn_total <= 4:
                max_new = free_plants
            else:
                max_new = max(1, free_plants // 3)
            plant_limit = state.effective_plantation_limit()
            total_future = len(state.plantations) + active_cons + self.new_builds_this_turn
            if self.new_builds_this_turn < max_new and total_future < plant_limit:
                t = self._try_new_build(state, p, ar)
                if t is not None:
                    self.new_builds_this_turn += 1
                    return _command(idx, t[0] - p.position[0], t[1] - p.position[1]), False
        return np.array([int(ActionKind.END_TURN), 0, 0, 0, 0], dtype=np.int64), True


def _command(author_idx, dx, dy):
    return np.array([int(ActionKind.COMMAND), author_idx, author_idx,
                     delta_to_target_offset(dx, dy), 0], dtype=np.int64)


def _upgrade(upgrade_id):
    return np.array([int(ActionKind.SET_UPGRADE), 0, 0, 0, upgrade_id], dtype=np.int64)


def _relocate():
    return np.array([int(ActionKind.SET_RELOCATE), 0, 0, 0, 0], dtype=np.int64)


def run_episode(map_size, round_turns, seed=0, game_config_overrides=None, verbose=False):
    base = DatssolEnv(seed=seed, map_size=map_size, round_turns=round_turns,
                     game_config_overrides=game_config_overrides or {})
    env = FlattenActionEnv(base)
    policy = ScriptedGo()
    obs, _ = env.reset()
    policy.reset()
    used = set()
    relocated = False
    first = True
    done = False
    while not done:
        action, ends = policy.act(env, used, relocated, first)
        first = False
        if action[0] == int(ActionKind.COMMAND):
            own_ids = sorted(pid for pid in env.env.state.plantations if pid not in env.env.state.enemy_ids)
            if action[1] < len(own_ids):
                used.add(own_ids[action[1]])
        if action[0] == int(ActionKind.SET_RELOCATE):
            relocated = True
        if ends:
            used = set()
            relocated = False
            first = True
        obs, _r, term, trunc, _ = env.step(action)
        if ends and verbose:
            s = env.env.state
            if s.turn_no % 20 == 0:
                print(f"turn {s.turn_no}: score={s.score}, plants={len(s.plantations)}, "
                      f"cons={len(s.constructions)}")
        done = term or trunc
    return env.env.state.score


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-size", type=int, nargs=2, default=[30, 30])
    ap.add_argument("--round-turns", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--ez", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    overrides = {"ds": 0, "cs": 10} if args.ez else {}
    scores = []
    for i in range(args.episodes):
        s = run_episode(tuple(args.map_size), args.round_turns, seed=i,
                       game_config_overrides=overrides, verbose=args.verbose)
        scores.append(s)
        print(f"ep {i}: score={s}")
    print(f"\nMean: {sum(scores)/len(scores):.0f}  Max: {max(scores)}")
