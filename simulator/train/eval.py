"""Evaluate a trained MaskablePPO agent on DatssolEnv."""
import argparse
from pathlib import Path
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from simulator.env.datssol_env import DatssolEnv
from simulator.env.flatten_action_wrapper import FlattenActionEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--map-size", type=int, nargs=2, default=[30, 30])
    ap.add_argument("--round-turns", type=int, default=100)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    base = DatssolEnv(
        seed=42, map_size=tuple(args.map_size), round_turns=args.round_turns
    )
    env = FlattenActionEnv(base)
    model = MaskablePPO.load(args.model, env=env)

    total_rewards = []
    scores = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(
                obs, action_masks=masks, deterministic=args.deterministic
            )
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        total_rewards.append(total)
        scores.append(env.env.state.score)
        print(f"episode {ep}: reward={total:.2f} score={env.env.state.score}")

    avg = sum(total_rewards) / len(total_rewards)
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    print(f"\nMean reward: {avg:.2f}")
    print(f"Mean score:  {avg_score:.0f}")
    print(f"Max score:   {max_score:.0f}")


if __name__ == "__main__":
    main()
