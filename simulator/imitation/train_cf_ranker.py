"""Train MLP with pairwise ranking loss on counterfactual dataset.

For each decision, we have multiple candidates with rollout score_delta.
Loss: pairwise margin — encourage score(better_candidate) > score(worse_candidate).
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_KEYS = [
    "dx_to_author", "dy_to_author", "adj_plants", "adj_constructions",
    "has_living_neighbor", "adj_to_cu", "cell_progress", "turns_alive_after",
    "is_boosted", "nearest_boosted_dist", "edge_dist", "dist_to_cu",
    "dist_to_centroid", "current_n_plants", "turn_norm", "score_norm",
]


class RankerMLP(nn.Module):
    def __init__(self, input_dim=16, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_pairs(cf_data: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each decision, build all candidate pairs and their delta-diff sign."""
    X1, X2, labels = [], [], []
    for d in cf_data:
        cands = d["candidates"]
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                delta_i = cands[i]["score_delta_rollout"]
                delta_j = cands[j]["score_delta_rollout"]
                if delta_i == delta_j:
                    continue
                f_i = np.array([cands[i]["features"][k] for k in FEATURE_KEYS], dtype=np.float32)
                f_j = np.array([cands[j]["features"][k] for k in FEATURE_KEYS], dtype=np.float32)
                X1.append(f_i)
                X2.append(f_j)
                labels.append(1.0 if delta_i > delta_j else 0.0)
    return np.stack(X1), np.stack(X2), np.array(labels, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("cf_ez.pkl"))
    ap.add_argument("--out", type=Path, default=Path("cf_ranker.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    with args.data.open("rb") as f:
        cf_data = pickle.load(f)

    X1, X2, y = build_pairs(cf_data)
    print(f"Pairs: {len(X1)}, frac label=1: {y.mean():.3f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X1_t = torch.from_numpy(X1).to(device)
    X2_t = torch.from_numpy(X2).to(device)
    y_t = torch.from_numpy(y).to(device)

    # Normalize features based on X1 distribution
    mean = X1_t.mean(dim=0, keepdim=True)
    std = X1_t.std(dim=0, keepdim=True) + 1e-6

    X1n = (X1_t - mean) / std
    X2n = (X2_t - mean) / std

    model = RankerMLP(input_dim=len(FEATURE_KEYS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = len(X1n)
    idx = torch.randperm(n)
    val_size = n // 5
    val_idx, train_idx = idx[:val_size], idx[val_size:]

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_idx))
        total, nb = 0.0, 0
        for i in range(0, len(train_idx), args.batch):
            b = train_idx[perm[i:i + args.batch]]
            s1 = model(X1n[b])
            s2 = model(X2n[b])
            # Pairwise logistic: if y=1, prefer s1>s2; else s2>s1
            # BCE-style: p = sigmoid(s1 - s2), target = y
            loss = F.binary_cross_entropy_with_logits(s1 - s2, y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            s1v = model(X1n[val_idx])
            s2v = model(X2n[val_idx])
            val_preds = (s1v > s2v).float()
            val_acc = (val_preds == y_t[val_idx]).float().mean().item()
            val_loss = F.binary_cross_entropy_with_logits(s1v - s2v, y_t[val_idx]).item()
        print(f"epoch {epoch}: train_loss={total/nb:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean.cpu().numpy(),
        "std": std.cpu().numpy(),
        "feature_keys": FEATURE_KEYS,
    }, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
