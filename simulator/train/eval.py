"""Evaluate a trained MaskablePPO agent on DatssolEnv (optionally via stage config)."""
import argparse
from pathlib import Path
from sb3_contrib import MaskablePPO
from simulator.env.datssol_env import DatssolEnv
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.train.curriculum import make_env_for_stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stage", type=str, default=None,
                    help="If set, use stage config (overrides map-size/round-turns)")
    ap.add_argument("--map-size", type=int, nargs=2, default=[30, 30])
    ap.add_argument("--round-turns", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    if args.stage is not None:
        stage = int(args.stage) if args.stage.isdigit() else args.stage
        base = make_env_for_stage(stage, seed=args.seed)
    else:
        base = DatssolEnv(
            seed=args.seed, map_size=tuple(args.map_size),
            round_turns=args.round_turns
        )
    env = FlattenActionEnv(base)
    try:
        model = MaskablePPO.load(args.model, env=env)
    except ValueError:
        # Frozen-backbone checkpoints have optimizer-state mismatch. Load model params
        # via lower-level API and skip optimizer loading entirely.
        import torch
        from stable_baselines3.common.save_util import load_from_zip_file
        from sb3_contrib import MaskablePPO as _MPPO
        from simulator.env.attention_extractor import AttentionExtractor
        data, params, _ = load_from_zip_file(
            str(args.model), device="cuda" if torch.cuda.is_available() else "cpu",
        )
        policy_kwargs = data.get("policy_kwargs") or {
            "features_extractor_class": AttentionExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
        }
        model = _MPPO(
            "MultiInputPolicy", env,
            policy_kwargs=policy_kwargs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.policy.load_state_dict(params["policy"])

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
