"""
Collect (build_candidate_features, final_score_delta) dataset from scripted Go episodes.

For each turn where Go decides to build, we record:
- Features of each valid candidate (the ones Go scored)
- Which candidate was chosen
- Final episode score

Supervised target: predict "score delta from this turn till end_of_episode" given features.

This lets us train an MLP that knows which candidates historically led to better outcomes.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import numpy as np

from simulator.env.datssol_env import DatssolEnv
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.env.action_space import ActionKind
from simulator.visualize.scripted_go import ScriptedGo, _nearest_boosted_distance, _cardinal


FEATURE_NAMES = [
    "dx_to_author",        # relative offset
    "dy_to_author",
    "adj_plants",
    "adj_constructions",
    "has_living_neighbor",
    "adj_to_cu",
    "cell_progress",
    "turns_alive_after",
    "is_boosted",
    "nearest_boosted_dist",
    "edge_dist",
    "dist_to_cu",
    "dist_to_centroid",
    "current_n_plants",
    "turn_norm",           # turn_no / round_turns
    "score_norm",          # score / 100000
    "is_chosen",           # bool — did Go pick this?
]


def extract_candidates(state, author, ar: int) -> list[dict]:
    """Extract all valid candidates with features — matching scripted_go._try_new_build logic."""
    w, h = state.config.map_width, state.config.map_height
    pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
    hq = state.get_hq()
    if hq is None:
        return []
    own_positions = [p.position for pid, p in state.plantations.items() if pid not in state.enemy_ids]
    cx = sum(p[0] for p in own_positions) // max(1, len(own_positions))
    cy = sum(p[1] for p in own_positions) // max(1, len(own_positions))
    constructing = set(state.constructions.keys())
    occupied = set(pos_to_id.keys()) | constructing

    candidates = []
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
            candidates.append({
                "position": target,
                "features": {
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
                },
            })
    return candidates


def collect_dataset(episodes: int, map_size=(30, 30), round_turns=100,
                    game_config_overrides=None) -> list[dict]:
    """Run N episodes of ScriptedGo, logging candidates + chosen + final score."""
    dataset: list[dict] = []

    for ep in range(episodes):
        base = DatssolEnv(seed=ep, map_size=map_size, round_turns=round_turns,
                         game_config_overrides=game_config_overrides or {})
        env = FlattenActionEnv(base)
        policy = ScriptedGo()
        obs, _ = env.reset()
        policy.reset()
        used: set = set()
        relocated = False
        first = True
        done = False

        # For each build decision this episode: record (candidates, chosen_pos, score_at_decision_turn)
        ep_decisions: list[dict] = []

        while not done:
            # Pre-action: if it's going to be a P3 new-build, enumerate candidates first
            # We detect "new build decision" by running the policy, then checking the action kind.
            state_before = env.env.state
            own_ids_before = sorted(
                pid for pid in state_before.plantations if pid not in state_before.enemy_ids
            )
            action, ends = policy.act(env, used, relocated, first)
            first = False

            # If COMMAND and target is a new cell (not a construction continuation), it's P3
            if action[0] == int(ActionKind.COMMAND):
                author_idx = action[1]
                if author_idx < len(own_ids_before):
                    author_id = own_ids_before[author_idx]
                    author = state_before.plantations[author_id]
                    # Target absolute
                    side = 2 * state_before.config.ar + 1
                    offset = int(action[3])
                    dx = (offset // side) - state_before.config.ar
                    dy = (offset % side) - state_before.config.ar
                    target_pos = (author.position[0] + dx, author.position[1] + dy)
                    # P3 new build: target not in constructions
                    if target_pos not in state_before.constructions:
                        candidates = extract_candidates(state_before, author,
                                                        state_before.config.ar)
                        ep_decisions.append({
                            "turn": state_before.turn_no,
                            "score": state_before.score,
                            "candidates": candidates,
                            "chosen": target_pos,
                        })

                own_ids = sorted(
                    pid for pid in env.env.state.plantations
                    if pid not in env.env.state.enemy_ids
                )
                if action[1] < len(own_ids):
                    used.add(own_ids[action[1]])
            if action[0] == int(ActionKind.SET_RELOCATE):
                relocated = True
            if ends:
                used = set()
                relocated = False
                first = True
            obs, _r, term, trunc, _ = env.step(action)
            done = term or trunc

        final_score = env.env.state.score
        # Label each decision with score delta till end
        for d in ep_decisions:
            d["final_score"] = final_score
            d["score_delta"] = final_score - d["score"]
            dataset.append(d)
        print(f"ep {ep}: score={final_score}, decisions={len(ep_decisions)}", flush=True)

    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--map-size", type=int, nargs=2, default=[30, 30])
    ap.add_argument("--round-turns", type=int, default=100)
    ap.add_argument("--ez", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("build_ranker_dataset.pkl"))
    args = ap.parse_args()

    overrides = {"ds": 0, "cs": 10} if args.ez else {}
    ds = collect_dataset(args.episodes, tuple(args.map_size), args.round_turns, overrides)
    print(f"\nTotal decisions: {len(ds)}")
    with args.out.open("wb") as f:
        pickle.dump(ds, f)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
