"""Supervised pretraining of policy on BC dataset."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from simulator.imitation.bc_dataset import build_bc_samples
from simulator.env.action_space import (
    ActionKind, MAX_PLANT, TARGET_OFFSET_DIM,
)


class BCBuffer(Dataset):
    def __init__(self, samples: list[tuple[dict, dict]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, action = self.samples[idx]
        return obs, action


def collate(batch):
    obs_keys = ["self", "global", "grid"]
    out = {k: torch.from_numpy(np.stack([b[0][k] for b in batch])) for k in obs_keys}
    actions = {
        "kind": torch.tensor([b[1]["kind"] for b in batch], dtype=torch.long),
        "author_idx": torch.tensor([b[1]["author_idx"] for b in batch], dtype=torch.long),
        "target_offset": torch.tensor([b[1]["target_offset"] for b in batch], dtype=torch.long),
    }
    return out, actions


class SimplePolicy(nn.Module):
    def __init__(self, f_self=13, max_plant=MAX_PLANT, f_global=13):
        super().__init__()
        self.self_mlp = nn.Sequential(
            nn.Linear(f_self, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(f_global, 32), nn.ReLU(),
        )
        self.backbone = nn.Sequential(
            nn.Linear(64 + 32, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
        )
        self.head_kind = nn.Linear(128, len(ActionKind))
        self.head_author = nn.Linear(128, max_plant)
        self.head_target = nn.Linear(128, TARGET_OFFSET_DIM)

    def forward(self, obs):
        self_feat = self.self_mlp(obs["self"]).mean(dim=1)  # pool over entities
        g = self.global_mlp(obs["global"])
        h = self.backbone(torch.cat([self_feat, g], dim=-1))
        return (
            self.head_kind(h),
            self.head_author(h),
            self.head_target(h),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, default=Path("dump"))
    ap.add_argument("--rounds", type=str, default="1,2,3")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("bc_weights.pt"))
    args = ap.parse_args()

    round_ids = [int(x) for x in args.rounds.split(",")]
    samples = list(build_bc_samples(args.dump, round_ids))
    print(f"Loaded {len(samples)} BC samples")

    if not samples:
        print("No samples — aborting without training.")
        return

    ds = BCBuffer(samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    policy = SimplePolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total = 0.0
        for obs, actions in dl:
            kind_l, author_l, target_l = policy(obs)
            loss = (
                loss_fn(kind_l, actions["kind"])
                + loss_fn(author_l, actions["author_idx"])
                + loss_fn(target_l, actions["target_offset"])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch}: loss={total / max(1, len(dl)):.4f}")

    torch.save(policy.state_dict(), args.out)
    print(f"Saved weights to {args.out}")


if __name__ == "__main__":
    main()
