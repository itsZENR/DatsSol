"""Evaluate a trained PPO agent on DatssolEnv."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from simulator.env.datssol_env import DatssolEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--map-size", type=int, nargs=2, default=[30, 30])
    ap.add_argument("--round-turns", type=int, default=100)
    args = ap.parse_args()

    env = DatssolEnv(
        seed=42, map_size=tuple(args.map_size), round_turns=args.round_turns
    )
    model = PPO.load(args.model, env=env)

    total_rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        total_rewards.append(total)
        print(f"episode {ep}: reward={total:.2f} score={env.state.score}")

    avg = sum(total_rewards) / len(total_rewards)
    print(f"\nMean reward: {avg:.2f}")


if __name__ == "__main__":
    main()
