"""ScriptedGo variant that uses a trained MLP ranker for build target selection."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from simulator.visualize.scripted_go import ScriptedGo, _cardinal, _nearest_boosted_distance
from simulator.imitation.train_build_ranker import BuildRankerMLP, FEATURE_KEYS


class ScriptedGoRanker(ScriptedGo):
    """Same heuristic except _try_new_build uses MLP to rank candidates."""

    def __init__(self, ranker_path: Path):
        super().__init__()
        ckpt = torch.load(str(ranker_path), weights_only=False)
        self.ranker = BuildRankerMLP(input_dim=len(FEATURE_KEYS))
        self.ranker.load_state_dict(ckpt["state_dict"])
        self.ranker.eval()
        self.mean = torch.from_numpy(ckpt["mean"])
        self.std = torch.from_numpy(ckpt["std"])

    def _score_candidate(self, feat_dict: dict) -> float:
        feat = np.array([feat_dict[k] for k in FEATURE_KEYS], dtype=np.float32)
        x = torch.from_numpy(feat).unsqueeze(0)
        xn = (x - self.mean) / self.std
        with torch.no_grad():
            return float(self.ranker(xn).item())

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
        best_score = -1e9

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
                if adj_plants + adj_cons == 0:
                    continue
                if not has_living and len(own_positions) > 1:
                    continue
                edge_d = min(target[0], w - 1 - target[0], target[1], h - 1 - target[1])
                feat = {
                    "dx_to_author": dx,
                    "dy_to_author": dy,
                    "adj_plants": adj_plants,
                    "adj_constructions": adj_cons,
                    "has_living_neighbor": int(has_living),
                    "adj_to_cu": int(adj_to_cu),
                    "cell_progress": prog,
                    "turns_alive_after": turns_alive if prog > 0 else 20,
                    "is_boosted": int(state.is_boosted_cell(target)),
                    "nearest_boosted_dist": _nearest_boosted_distance(target),
                    "edge_dist": edge_d,
                    "dist_to_cu": abs(target[0] - hq.position[0]) + abs(target[1] - hq.position[1]),
                    "dist_to_centroid": abs(target[0] - cx) + abs(target[1] - cy),
                    "current_n_plants": len(own_positions),
                    "turn_norm": state.turn_no / state.config.round_turns,
                    "score_norm": state.score / 1e5,
                }
                score = self._score_candidate(feat)
                if score > best_score:
                    best_score = score
                    best_pos = target

        # Sticky target handling
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


if __name__ == "__main__":
    import argparse
    from simulator.env.datssol_env import DatssolEnv
    from simulator.env.flatten_action_wrapper import FlattenActionEnv
    from simulator.env.action_space import ActionKind
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranker", type=Path, default=Path("build_ranker.pt"))
    ap.add_argument("--map-size", type=int, nargs=2, default=[30, 30])
    ap.add_argument("--round-turns", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--ez", action="store_true")
    args = ap.parse_args()

    overrides = {"ds": 0, "cs": 10} if args.ez else {}
    scores = []
    for i in range(args.episodes):
        base = DatssolEnv(seed=i, map_size=tuple(args.map_size),
                         round_turns=args.round_turns,
                         game_config_overrides=overrides)
        env = FlattenActionEnv(base)
        policy = ScriptedGoRanker(args.ranker)
        obs, _ = env.reset()
        policy.reset()
        used, relocated, first = set(), False, True
        done = False
        while not done:
            action, ends = policy.act(env, used, relocated, first)
            first = False
            if action[0] == int(ActionKind.COMMAND):
                own_ids = sorted(pid for pid in env.env.state.plantations
                               if pid not in env.env.state.enemy_ids)
                if action[1] < len(own_ids):
                    used.add(own_ids[action[1]])
            if action[0] == int(ActionKind.SET_RELOCATE):
                relocated = True
            if ends:
                used, relocated, first = set(), False, True
            obs, _r, term, trunc, _ = env.step(action)
            done = term or trunc
        scores.append(env.env.state.score)
        print(f"ep {i}: {env.env.state.score}")
    print(f"\nRanker mean: {sum(scores)/len(scores):.0f}  max: {max(scores)}")
