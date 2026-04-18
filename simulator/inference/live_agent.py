"""
Live agent loop: reads /api/arena, runs policy autoregressive, posts commands.
"""
import argparse
import logging
import time
from pathlib import Path

import requests
from stable_baselines3 import PPO

from simulator.inference.adapter import api_to_state, state_to_payload
from simulator.env.obs_encoder import encode_observation
from simulator.env.action_mask import TurnContext
from simulator.env.action_space import (
    ActionKind, decode_action, target_offset_to_delta,
)
from simulator.engine.commands import Command, CommandSet
from simulator.engine.types import UpgradeName
# Re-use the existing rule-based bot's HTTP config/client
from config import TOKEN, BASE_URL

log = logging.getLogger("live_agent")


def compose_turn(state, policy, max_micro: int = 45) -> CommandSet:
    ctx = TurnContext()
    commands: list[Command] = []
    upgrade = None
    relocate: list = []

    own_ids = sorted(
        pid for pid in state.plantations.keys() if pid not in state.enemy_ids
    )

    for _ in range(max_micro):
        obs = encode_observation(state)
        action, _ = policy.predict(obs, deterministic=True)
        decoded = decode_action(action)

        if decoded.kind == ActionKind.END_TURN:
            break
        elif decoded.kind == ActionKind.SET_UPGRADE and upgrade is None:
            upgrade_list = list(UpgradeName)
            if decoded.upgrade_id < len(upgrade_list):
                upgrade = upgrade_list[decoded.upgrade_id]
            ctx.upgrade_set = True
        elif decoded.kind == ActionKind.SET_RELOCATE and not relocate:
            hq = state.get_hq()
            if hq:
                pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nb = (hq.position[0] + dx, hq.position[1] + dy)
                    if nb in pos_to_id and pos_to_id[nb] in state.connected_ids:
                        relocate = [hq.position, nb]
                        break
            ctx.relocate_set = True
        elif decoded.kind == ActionKind.COMMAND:
            if decoded.author_idx < len(own_ids):
                author_id = own_ids[decoded.author_idx]
                if author_id not in ctx.authors_used:
                    author = state.plantations[author_id]
                    relay_id = (
                        own_ids[decoded.relay_idx]
                        if decoded.relay_idx < len(own_ids) else author_id
                    )
                    relay = state.plantations[relay_id]
                    dx, dy = target_offset_to_delta(decoded.target_offset)
                    target = (relay.position[0] + dx, relay.position[1] + dy)
                    commands.append(Command(author.position, relay.position, target))
                    ctx.authors_used.add(author_id)
        # NOOP — simply counts

    return CommandSet(commands=commands, upgrade=upgrade, relocate=relocate)


def fetch_arena() -> dict:
    r = requests.get(f"{BASE_URL}/api/arena",
                     headers={"X-Auth-Token": TOKEN}, timeout=5)
    r.raise_for_status()
    return r.json()


def post_cmd(payload: dict) -> dict:
    r = requests.post(f"{BASE_URL}/api/command",
                      json=payload,
                      headers={"X-Auth-Token": TOKEN, "Content-Type": "application/json"},
                      timeout=5)
    r.raise_for_status()
    return r.json()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--poll-interval", type=float, default=0.5)
    args = ap.parse_args()

    policy = PPO.load(args.model)
    last_turn = -1

    while True:
        try:
            data = fetch_arena()
            state = api_to_state(data)
            if state.turn_no == last_turn:
                time.sleep(args.poll_interval)
                continue
            last_turn = state.turn_no
            cs = compose_turn(state, policy)
            payload = state_to_payload(cs)
            resp = post_cmd(payload)
            if resp.get("errors"):
                log.warning("server errors: %s", resp["errors"])
            sleep_for = max(0.1, float(data.get("nextTurnIn", 1.0)) - 0.1)
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error("live loop error: %s", e, exc_info=True)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
