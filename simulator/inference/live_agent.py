"""
Live agent loop: reads /api/arena, runs policy autoregressive, posts commands.
"""
import argparse
import logging
import time
from pathlib import Path

import requests
from sb3_contrib import MaskablePPO


def _load_model(path):
    try:
        return MaskablePPO.load(path)
    except ValueError:
        # Frozen-backbone optimizer mismatch — load params only via lower-level API
        import torch
        from stable_baselines3.common.save_util import load_from_zip_file
        from simulator.env.attention_extractor import AttentionExtractor
        from simulator.env.datssol_env import DatssolEnv
        from simulator.env.flatten_action_wrapper import FlattenActionEnv
        data, params, _ = load_from_zip_file(
            str(path), device="cuda" if torch.cuda.is_available() else "cpu"
        )
        policy_kwargs = data.get("policy_kwargs") or {
            "features_extractor_class": AttentionExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
        }
        dummy = FlattenActionEnv(DatssolEnv(seed=0, map_size=(30, 30), round_turns=10))
        m = MaskablePPO(
            "MultiInputPolicy", dummy,
            policy_kwargs=policy_kwargs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        m.policy.load_state_dict(params["policy"])
        return m

import numpy as np
from simulator.inference.adapter import api_to_state, state_to_payload
from simulator.env.obs_encoder import encode_observation
from simulator.env.action_mask import TurnContext, build_action_mask
from simulator.env.action_space import (
    ActionKind, MAX_PLANT, TARGET_OFFSET_DIM, N_UPGRADES,
    target_offset_to_delta,
)
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.engine.commands import Command, CommandSet
from simulator.engine.types import UpgradeName


def _flat_mask_from_ctx(state, ctx):
    """Build flat mask vector matching FlattenActionEnv NVEC order."""
    m = build_action_mask(state, ctx)
    return np.concatenate([
        m["kind"], m["author_idx"], m["relay_idx"],
        m["target_offset"], m["upgrade_id"],
    ]).astype(np.int8)
# Re-use the existing rule-based bot's HTTP config/client
from config import TOKEN, BASE_URL

log = logging.getLogger("live_agent")


def compose_turn(state, policy, max_micro: int = 45) -> CommandSet:
    ctx = TurnContext()
    commands: list[Command] = []
    upgrade = None
    relocate: list = []

    # Recompute connectivity because api_to_state doesn't run engine.tick.
    from simulator.engine.spatial import bfs_connected
    own_pos = {pid: p.position for pid, p in state.plantations.items()
               if pid not in state.enemy_ids}
    state.connected_ids = bfs_connected(state.hq_id, own_pos)
    # Must match TurnContext.ordered_author_ids → sorted(state.connected_ids).
    own_ids = sorted(state.connected_ids)

    for _ in range(max_micro):
        obs = encode_observation(state, ctx.authors_used, vision_aware=True)
        flat_mask = _flat_mask_from_ctx(state, ctx)
        action, _ = policy.predict(obs, action_masks=flat_mask, deterministic=True)
        action = np.asarray(action).flatten()

        kind = int(action[0])
        author_idx = int(action[1])
        relay_idx = int(action[2])
        target_offset = int(action[3])
        upgrade_id = int(action[4])

        if kind == int(ActionKind.END_TURN):
            break
        elif kind == int(ActionKind.SET_UPGRADE) and upgrade is None:
            upgrade_list = list(UpgradeName)
            if upgrade_id < len(upgrade_list):
                upgrade = upgrade_list[upgrade_id]
            ctx.upgrade_set = True
        elif kind == int(ActionKind.SET_RELOCATE) and not relocate:
            hq = state.get_hq()
            if hq:
                pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nb = (hq.position[0] + dx, hq.position[1] + dy)
                    if nb in pos_to_id and pos_to_id[nb] in state.connected_ids:
                        relocate = [hq.position, nb]
                        break
            ctx.relocate_set = True
        elif kind == int(ActionKind.COMMAND):
            if author_idx < len(own_ids):
                author_id = own_ids[author_idx]
                if author_id not in ctx.authors_used:
                    author = state.plantations[author_id]
                    relay_id = (
                        own_ids[relay_idx]
                        if relay_idx < len(own_ids) else author_id
                    )
                    relay = state.plantations[relay_id]
                    dx, dy = target_offset_to_delta(target_offset)
                    target = (relay.position[0] + dx, relay.position[1] + dy)
                    commands.append(Command(author.position, relay.position, target))
                    ctx.authors_used.add(author_id)
        # NOOP — simply counts

    return CommandSet(commands=commands, upgrade=upgrade, relocate=relocate)


ARENA_SNAPSHOT_PATH = Path("tmp/arena_snapshot.json")
ACTIONS_SNAPSHOT_PATH = Path("tmp/actions_snapshot.json")


def fetch_arena(timeout: float = 3.0) -> dict:
    r = requests.get(f"{BASE_URL}/api/arena",
                     headers={"X-Auth-Token": TOKEN}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    _write_snapshot(data)
    return data


def _write_snapshot(data: dict) -> None:
    _atomic_write_json(ARENA_SNAPSHOT_PATH, data)


def _write_actions(turn_no: int, cs, server_errors: list | None) -> None:
    """Record the agent's actions for this turn in a shape matching ActionData[]."""
    actions = []
    for c in cs.commands:
        actions.append({
            "authorX": int(c.author[0]), "authorY": int(c.author[1]),
            "targetX": int(c.target[0]), "targetY": int(c.target[1]),
            "action": "build",
            "reason": "",
        })
    payload = {
        "turnNo": int(turn_no),
        "actions": actions,
        "upgrade": cs.upgrade.value if cs.upgrade is not None else None,
        "relocate": [list(p) for p in cs.relocate],
        "errors": server_errors or [],
    }
    _atomic_write_json(ACTIONS_SNAPSHOT_PATH, payload)


def _atomic_write_json(path: Path, data) -> None:
    import json
    import os
    import tempfile
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.stem + "_", suffix=".json",
                                   dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        pass


def post_cmd(payload: dict) -> dict:
    last_exc = None
    for attempt in range(2):
        try:
            r = requests.post(f"{BASE_URL}/api/command",
                              json=payload,
                              headers={"X-Auth-Token": TOKEN, "Content-Type": "application/json"},
                              timeout=10)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectTimeout, requests.ReadTimeout) as e:
            last_exc = e
    raise last_exc


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--poll-interval", type=float, default=0.5)
    args = ap.parse_args()

    policy = _load_model(args.model)

    import threading
    latest: dict = {"data": None, "lock": threading.Lock()}
    stop_evt = threading.Event()

    def poller():
        while not stop_evt.is_set():
            try:
                data = fetch_arena()  # writes arena_snapshot.json atomically
                with latest["lock"]:
                    latest["data"] = data
            except Exception as e:
                log.warning("poll error: %s", e)
            stop_evt.wait(args.poll_interval)

    th = threading.Thread(target=poller, daemon=True)
    th.start()

    last_turn = -1
    try:
        while True:
            try:
                with latest["lock"]:
                    data = latest["data"]
                if data is None:
                    time.sleep(0.1)
                    continue
                state = api_to_state(data)
                if state.turn_no == last_turn:
                    time.sleep(0.1)
                    continue
                last_turn = state.turn_no
                cs = compose_turn(state, policy)
                errors: list = []
                if not cs.commands and cs.upgrade is None and not cs.relocate:
                    log.info("turn %d: empty CommandSet — skipping POST", state.turn_no)
                else:
                    payload = state_to_payload(cs)
                    log.info("turn %d: commands=%d upgrade=%s relocate=%s",
                             state.turn_no, len(cs.commands), cs.upgrade, cs.relocate)
                    resp = post_cmd(payload)
                    errors = list(resp.get("errors") or [])
                    if errors:
                        log.warning("server errors: %s", errors)
                _write_actions(state.turn_no, cs, errors)
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error("live loop error: %s", e, exc_info=True)
                time.sleep(1.0)
    finally:
        stop_evt.set()


if __name__ == "__main__":
    main()
