"""
Counterfactual dataset: fork env at build decisions, try top-K Go-scored candidates,
measure score_delta after a short rollout. Produces data where we can truly compare candidates.
"""
from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path
import numpy as np

from simulator.env.datssol_env import DatssolEnv
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.env.action_space import ActionKind, delta_to_target_offset
from simulator.visualize.scripted_go import ScriptedGo, _cardinal, _nearest_boosted_distance
from simulator.imitation.build_ranker_dataset import extract_candidates, FEATURE_NAMES


def go_score_candidate(state, target, author_pos, adj_plants, adj_cons, adj_to_cu,
                       has_living, prog, is_boosted, near_boosted, edge_dist,
                       dist_cu, dist_centroid) -> int:
    """Replicate Go's scoring for candidate ranking."""
    score = 100
    score += 300 - dist_centroid * 30
    score += 200 - dist_cu * 20
    if adj_to_cu:
        score += 150
    score += adj_plants * 100
    score += adj_cons * 30
    if prog > 0:
        score -= prog * 2
    if is_boosted:
        score += 100
    if near_boosted <= 3:
        score += 40
    if edge_dist < 10:
        score -= (10 - edge_dist) * 30
    return score


def rank_by_go_score(candidates):
    """Return candidates sorted by Go heuristic score, descending."""
    def feat_score(c):
        f = c["features"]
        return go_score_candidate(
            None, c["position"], None,
            f["adj_plants"], f["adj_constructions"], f["adj_to_cu"],
            f["has_living_neighbor"], f["cell_progress"],
            f["is_boosted"], f["nearest_boosted_dist"], f["edge_dist"],
            f["dist_to_cu"], f["dist_to_centroid"],
        )
    return sorted(candidates, key=feat_score, reverse=True)


def rollout_with_forced_build(env_snapshot, forced_target, rollout_turns, author_idx):
    """Take a deep-copied env, force the next build to use forced_target, then let
    Go finish the episode (or rollout_turns turns). Return score_delta."""
    env = copy.deepcopy(env_snapshot)
    state = env.env.state
    start_score = state.score
    start_turn = state.turn_no

    # Build absolute dx/dy from current author position
    own_ids = sorted(pid for pid in state.plantations if pid not in state.enemy_ids)
    if author_idx >= len(own_ids):
        return 0
    author = state.plantations[own_ids[author_idx]]
    dx = forced_target[0] - author.position[0]
    dy = forced_target[1] - author.position[1]
    if abs(dx) > state.config.ar or abs(dy) > state.config.ar:
        return 0
    # Force this action
    forced_action = np.array([
        int(ActionKind.COMMAND), author_idx, author_idx,
        delta_to_target_offset(dx, dy), 0,
    ], dtype=np.int64)
    obs, r, term, trunc, _ = env.step(forced_action)
    if term or trunc:
        return env.env.state.score - start_score

    # Use ScriptedGo for remainder
    policy = ScriptedGo()
    used = {own_ids[author_idx]}
    relocated = False
    first = False  # not first micro, we already used one author
    done = False
    while not done and env.env.state.turn_no - start_turn < rollout_turns:
        action, ends = policy.act(env, used, relocated, first)
        first = False
        if action[0] == int(ActionKind.COMMAND):
            oids = sorted(pid for pid in env.env.state.plantations
                         if pid not in env.env.state.enemy_ids)
            if action[1] < len(oids):
                used.add(oids[action[1]])
        if action[0] == int(ActionKind.SET_RELOCATE):
            relocated = True
        if ends:
            used = set()
            relocated = False
            first = True
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.env.state.score - start_score


def collect_counterfactual(episodes: int, map_size=(30, 30), round_turns=100,
                           game_config_overrides=None, top_k: int = 5,
                           rollout_turns: int = 20,
                           decision_stride: int = 3) -> list[dict]:
    """Collect (candidate_features, score_delta) pairs comparable within same decision."""
    dataset = []
    for ep in range(episodes):
        base = DatssolEnv(seed=ep, map_size=map_size, round_turns=round_turns,
                         game_config_overrides=game_config_overrides or {})
        env = FlattenActionEnv(base)
        policy = ScriptedGo()
        obs, _ = env.reset()
        policy.reset()
        used, relocated, first = set(), False, True
        done = False
        decision_count = 0
        while not done:
            # Check if Go wants to do a COMMAND action
            state_before = env.env.state
            own_ids_before = sorted(
                pid for pid in state_before.plantations if pid not in state_before.enemy_ids
            )
            # Peek what Go would do (don't mutate policy state much)
            policy_snap = copy.deepcopy(policy)
            action, ends = policy.act(env, used, relocated, first)
            first = False

            # If P3 (new build), maybe fork
            if (action[0] == int(ActionKind.COMMAND) and
                action[1] < len(own_ids_before)):
                author = state_before.plantations[own_ids_before[action[1]]]
                side = 2 * state_before.config.ar + 1
                off = int(action[3])
                dx = (off // side) - state_before.config.ar
                dy = (off % side) - state_before.config.ar
                target_pos = (author.position[0] + dx, author.position[1] + dy)
                is_new_build = target_pos not in state_before.constructions

                if is_new_build and decision_count % decision_stride == 0:
                    # Enumerate candidates and rank via Go scoring
                    candidates = extract_candidates(state_before, author,
                                                    state_before.config.ar)
                    ranked = rank_by_go_score(candidates)[:top_k]
                    # For each top candidate, fork env and rollout
                    cf_records = []
                    for cand in ranked:
                        delta = rollout_with_forced_build(
                            env, cand["position"], rollout_turns, action[1]
                        )
                        cf_records.append({
                            "features": cand["features"],
                            "position": cand["position"],
                            "score_delta_rollout": delta,
                        })
                    if len(cf_records) >= 2:
                        dataset.append({
                            "turn": state_before.turn_no,
                            "candidates": cf_records,
                        })
                if is_new_build:
                    decision_count += 1

            # Apply Go's actual action
            if action[0] == int(ActionKind.COMMAND):
                oids = sorted(pid for pid in env.env.state.plantations
                             if pid not in env.env.state.enemy_ids)
                if action[1] < len(oids):
                    used.add(oids[action[1]])
            if action[0] == int(ActionKind.SET_RELOCATE):
                relocated = True
            if ends:
                used, relocated, first = set(), False, True
            obs, _r, term, trunc, _ = env.step(action)
            done = term or trunc
        print(f"ep {ep}: score={env.env.state.score}, cf_decisions={sum(1 for d in dataset if d.get('_ep_mark') == ep) or 'logged'}", flush=True)

    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--map-size", type=int, nargs=2, default=[30, 30])
    ap.add_argument("--round-turns", type=int, default=100)
    ap.add_argument("--ez", action="store_true")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--rollout-turns", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("cf_dataset.pkl"))
    args = ap.parse_args()

    overrides = {"ds": 0, "cs": 10} if args.ez else {}
    ds = collect_counterfactual(
        args.episodes, tuple(args.map_size), args.round_turns,
        overrides, args.top_k, args.rollout_turns, args.stride,
    )
    print(f"\nTotal counterfactual decisions: {len(ds)}")
    total_pairs = sum(len(d["candidates"]) * (len(d["candidates"]) - 1) // 2 for d in ds)
    print(f"Total pairwise comparisons: {total_pairs}")
    with args.out.open("wb") as f:
        pickle.dump(ds, f)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
