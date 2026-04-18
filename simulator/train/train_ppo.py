"""PPO training on DatssolEnv with curriculum (MaskablePPO + flattened Dict actions)."""
import argparse
from pathlib import Path
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from simulator.train.curriculum import make_env_for_stage
from simulator.env.flatten_action_wrapper import FlattenActionEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("ppo_model"))
    args = ap.parse_args()

    def env_factory(i: int):
        def _make():
            base = make_env_for_stage(args.stage, seed=args.seed + i)
            return FlattenActionEnv(base)
        return _make

    vec = DummyVecEnv([env_factory(i) for i in range(args.n_envs)])

    model = MaskablePPO(
        "MultiInputPolicy",
        vec,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
