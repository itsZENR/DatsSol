"""
BC warm-start for MaskablePPO from the scripted greedy-builder teacher.

Pipeline:
1. Collect N episodes from ScriptedGreedyBuilder → (obs, flat_action) pairs.
2. Create a MaskablePPO model (same architecture as used in RL training).
3. Train the policy network via cross-entropy on the teacher actions.
4. Save as a MaskablePPO checkpoint that can be resumed by train_long.py.

Usage:
    python -m simulator.imitation.bc_from_scripted --stage 0_ez --episodes 50 \\
        --epochs 3 --out ppo_bc_init
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from simulator.env.datssol_env import DatssolEnv
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.env.action_space import ActionKind
from simulator.visualize.scripted_policy import ScriptedGreedyBuilder
from simulator.visualize.scripted_go import ScriptedGo
from simulator.train.curriculum import make_env_for_stage


def collect_scripted_trajectories(
    make_env, n_episodes: int, seed_base: int = 0, teacher: str = "go",
) -> list[tuple[dict, np.ndarray]]:
    """Run scripted teacher for n episodes, collect (obs, action) pairs.

    teacher: 'v1' (ScriptedGreedyBuilder) or 'go' (ScriptedGo — ported from Go conveyor).
    """
    samples: list[tuple[dict, np.ndarray]] = []
    for ep in range(n_episodes):
        env = make_env(seed_base + ep)
        if teacher == "go":
            policy = ScriptedGo()
        else:
            policy = ScriptedGreedyBuilder()
        obs, _ = env.reset()
        policy.reset()
        used_authors: set = set()
        relocated = False
        first_of_turn = True
        done = False
        while not done:
            if teacher == "go":
                action, ends_turn = policy.act(env, used_authors, relocated, first_of_turn)
            else:
                action, ends_turn, _ = policy.act(env, used_authors, relocated)
            first_of_turn = False
            sample_obs = {k: np.array(v, copy=True) for k, v in obs.items()}
            samples.append((sample_obs, action.copy()))
            if action[0] == int(ActionKind.COMMAND):
                own_ids = sorted(
                    pid for pid in env.env.state.plantations
                    if pid not in env.env.state.enemy_ids
                )
                if action[1] < len(own_ids):
                    used_authors.add(own_ids[action[1]])
            if action[0] == int(ActionKind.SET_RELOCATE):
                relocated = True
            if ends_turn:
                used_authors = set()
                relocated = False
                first_of_turn = True
            obs, _r, term, trunc, _ = env.step(action)
            done = term or trunc
        final_score = env.env.state.score
        print(f"  ep {ep}: collected, final_score={final_score}", flush=True)
    return samples


class BCDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    obs_keys = list(batch[0][0].keys())
    out_obs = {}
    for k in obs_keys:
        out_obs[k] = torch.from_numpy(np.stack([b[0][k] for b in batch]))
    actions = torch.from_numpy(np.stack([b[1] for b in batch])).long()
    return out_obs, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, default="0_ez")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=Path, default=Path("ppo_bc_init"))
    ap.add_argument("--teacher", type=str, default="go", choices=["v1", "go"])
    args = ap.parse_args()

    def make_env(seed):
        base = make_env_for_stage(
            int(args.stage) if args.stage.isdigit() else args.stage,
            seed=seed
        )
        return FlattenActionEnv(base)

    print(f"Collecting {args.episodes} scripted ({args.teacher}) episodes on stage {args.stage}...", flush=True)
    samples = collect_scripted_trajectories(make_env, args.episodes, teacher=args.teacher)
    print(f"\nTotal samples: {len(samples)}", flush=True)

    # Init MaskablePPO with the right env so weights shape is correct.
    vec = DummyVecEnv([lambda: make_env(999)])
    from simulator.env.attention_extractor import AttentionExtractor
    policy_kwargs = {
        "features_extractor_class": AttentionExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
    }
    model = MaskablePPO(
        "MultiInputPolicy", vec, verbose=0,
        n_steps=256, batch_size=64, learning_rate=args.lr, gamma=0.99,
        ent_coef=0.03, policy_kwargs=policy_kwargs, seed=0,
    )
    policy = model.policy
    device = policy.device
    print(f"Policy on device: {device}", flush=True)

    # BC training: cross-entropy on action_net outputs.
    ds = BCDataset(samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    # MaskablePPO with MultiDiscrete action: policy outputs `action_net` which is a
    # MultiCategoricalDistribution; its logits come from the action_net Linear layer.
    # Best approach: use policy.evaluate_actions(obs, actions) style but simpler is to
    # extract features → action_net → logits, split per sub-action, CE per dim.

    for epoch in range(args.epochs):
        total = 0.0
        n = 0
        for obs_batch, act_batch in dl:
            obs_batch = {k: v.to(device).float() for k, v in obs_batch.items()}
            act_batch = act_batch.to(device)

            # Forward through policy to get action logits.
            features = policy.extract_features(obs_batch)
            if isinstance(features, tuple):
                features = features[0]
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)  # [B, sum(nvec)]

            # Split logits per sub-action
            nvec = vec.envs[0].NVEC.tolist()
            loss = 0.0
            offset = 0
            for i, n_choices in enumerate(nvec):
                sub_logits = logits[:, offset:offset + n_choices]
                sub_targets = act_batch[:, i]
                loss = loss + F.cross_entropy(sub_logits, sub_targets)
                offset += n_choices

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"epoch {epoch}: mean_loss={total / max(1, n):.4f}", flush=True)

    model.save(args.out)
    print(f"Saved BC-init to {args.out}", flush=True)


if __name__ == "__main__":
    main()
