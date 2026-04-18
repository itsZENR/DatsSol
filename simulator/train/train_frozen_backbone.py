"""
Resume from BC checkpoint, freeze the feature extractor (attention backbone),
train only the policy/value heads via standard PPO.

Rationale: BC-attention fits Go teacher well (loss 0.95). The representation IS good.
RL drift happens because advantage gradient changes the representation, destroying BC.
Freezing the backbone forces RL to improve via head-only adjustments — BC structure preserved.
"""
import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from simulator.train.curriculum import make_env_for_stage, load_stage_config, CONFIG_DIR
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.env.datssol_env import DatssolEnv
from simulator.env.action_space import ActionKind


from stable_baselines3.common.callbacks import BaseCallback


class EvalBC(BaseCallback):
    def __init__(self, stage_cfg: dict, eval_freq: int):
        super().__init__()
        self.stage_cfg = stage_cfg
        self.eval_freq = eval_freq
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.eval_freq:
            return True
        self._last = self.num_timesteps
        base = DatssolEnv(
            seed=99,
            map_size=tuple(self.stage_cfg["map_size"]),
            round_turns=self.stage_cfg["round_turns"],
            game_config_overrides=self.stage_cfg.get("game_config_overrides", {}),
        )
        env = FlattenActionEnv(base)
        scores = []
        for _ in range(2):
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


def freeze_backbone(model):
    """Freeze the features extractor (AttentionExtractor), keep heads trainable."""
    # Policy has: features_extractor, mlp_extractor, action_net, value_net
    # Freeze features_extractor (the big attention module).
    frozen = 0
    trainable = 0
    for name, p in model.policy.named_parameters():
        if name.startswith("features_extractor."):
            p.requires_grad = False
            frozen += p.numel()
        else:
            trainable += p.numel()
    print(f"Frozen: {frozen} params (features_extractor)", flush=True)
    print(f"Trainable: {trainable} params (heads/mlp_extractor)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default="0_ez")
    ap.add_argument("--resume", type=Path, required=True)
    ap.add_argument("--total-timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--eval-freq", type=int, default=30000)
    ap.add_argument("--out", type=Path, default=Path("ppo_frozen"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.stage.isdigit():
        stage_cfg = load_stage_config(int(args.stage))
    else:
        import yaml
        with (CONFIG_DIR / f"stage{args.stage}.yaml").open() as f:
            stage_cfg = yaml.safe_load(f)

    def factory(i):
        def _mk():
            stage = int(args.stage) if args.stage.isdigit() else args.stage
            return FlattenActionEnv(make_env_for_stage(stage, seed=args.seed + i))
        return _mk

    vec = DummyVecEnv([factory(i) for i in range(args.n_envs)])

    try:
        model = MaskablePPO.load(str(args.resume), env=vec)
    except ValueError:
        # Optimizer-state mismatch (from prior frozen training). Load params only.
        import torch
        from stable_baselines3.common.save_util import load_from_zip_file
        from simulator.env.attention_extractor import AttentionExtractor
        data, params, _ = load_from_zip_file(
            str(args.resume), device="cuda" if torch.cuda.is_available() else "cpu"
        )
        policy_kwargs = data.get("policy_kwargs") or {
            "features_extractor_class": AttentionExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
        }
        model = MaskablePPO(
            "MultiInputPolicy", vec,
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr, ent_coef=args.ent_coef,
            n_steps=2048, batch_size=256, n_epochs=6,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.policy.load_state_dict(params["policy"])
    # Update hyperparameters for fine-tune
    model.learning_rate = args.lr
    model.ent_coef = args.ent_coef
    # Reinstantiate optimizer to respect frozen params
    freeze_backbone(model)
    import torch
    trainable_params = [p for p in model.policy.parameters() if p.requires_grad]
    model.policy.optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    ckpt_dir = args.out.parent / f"{args.out.name}_ckpts"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 50000 // args.n_envs),
        save_path=str(ckpt_dir),
        name_prefix="ppo",
    )

    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([ckpt_cb, EvalBC(stage_cfg, args.eval_freq)])

    print(f"Training from {args.resume} (frozen backbone): "
          f"timesteps={args.total_timesteps} n_envs={args.n_envs}", flush=True)
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                reset_num_timesteps=True)
    model.save(args.out)
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
