"""Long PPO training with entropy bonus, checkpointing, and progress logging.

Launches background-friendly training with:
- higher entropy coefficient to encourage exploration out of idle trap
- checkpoint saved every 50K steps so any snapshot is usable
- eval-on-checkpoint prints mean score to stdout (pick up via Monitor)

Usage:
    python -m simulator.train.train_long --stage 0 --total-timesteps 5000000 --n-envs 8
"""
import argparse
import os
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from simulator.train.curriculum import make_env_for_stage
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.env.datssol_env import DatssolEnv
from simulator.env.action_space import ActionKind


class EvalAndLogCallback(BaseCallback):
    """Every `eval_freq` training steps, run 2 eval episodes and print mean score."""
    def __init__(self, eval_freq: int, stage_cfg: dict, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.stage_cfg = stage_cfg
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        base = DatssolEnv(
            seed=99,
            map_size=tuple(self.stage_cfg["map_size"]),
            round_turns=self.stage_cfg["round_turns"],
        )
        env = FlattenActionEnv(base)
        scores = []
        for ep in range(2):
            obs, _ = env.reset()
            done = False
            while not done:
                mask = env.action_masks()
                a, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
                obs, _r, term, trunc, _ = env.step(a)
                done = term or trunc
            scores.append(env.env.state.score)
        mean = sum(scores) / len(scores)
        best = max(scores)
        print(f">>> EVAL step={self.num_timesteps} mean_score={mean:.0f} best={best:.0f} "
              f"scores={scores}", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--total-timesteps", type=int, default=5_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("ppo_long"))
    ap.add_argument("--checkpoint-freq", type=int, default=50_000)
    ap.add_argument("--eval-freq", type=int, default=50_000)
    ap.add_argument("--ent-coef", type=float, default=0.05,
                    help="Entropy coefficient — higher = more exploration. 0.05 encourages actual building.")
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Resume from a saved model")
    args = ap.parse_args()

    from simulator.train.curriculum import load_stage_config
    stage_cfg = load_stage_config(args.stage)

    def env_factory(i: int):
        def _make():
            base = make_env_for_stage(args.stage, seed=args.seed + i)
            return FlattenActionEnv(base)
        return _make

    vec_cls = SubprocVecEnv if args.n_envs > 1 and os.name != "nt" else DummyVecEnv
    vec = vec_cls([env_factory(i) for i in range(args.n_envs)])

    ckpt_dir = args.out.parent / f"{args.out.name}_ckpts"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.n_envs),
            save_path=str(ckpt_dir),
            name_prefix="ppo",
        ),
        EvalAndLogCallback(eval_freq=args.eval_freq, stage_cfg=stage_cfg),
    ]

    if args.resume is not None:
        model = MaskablePPO.load(str(args.resume), env=vec)
        print(f"Resumed from {args.resume}", flush=True)
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            vec,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=args.ent_coef,
            seed=args.seed,
        )

    print(f"Training: stage={args.stage} timesteps={args.total_timesteps} "
          f"n_envs={args.n_envs} ent_coef={args.ent_coef}", flush=True)
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                reset_num_timesteps=(args.resume is None))
    model.save(args.out)
    print(f"Final model saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
