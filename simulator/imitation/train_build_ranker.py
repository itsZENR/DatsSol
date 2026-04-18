"""Train a tiny MLP to rank build candidates by expected future score delta.

Approach: for each recorded decision, features of CHOSEN candidate → score_delta.
This is a regression problem: predict final_score - current_score.
After training, use the MLP's score to pick the best candidate.
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


class BuildRankerMLP(nn.Module):
    def __init__(self, input_dim=16, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_chosen_dataset(pkl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """For each decision, return features of chosen candidate + score_delta."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    X, y = [], []
    for d in data:
        chosen = d["chosen"]
        for c in d["candidates"]:
            if c["position"] == chosen:
                feat = np.array([c["features"][k] for k in FEATURE_KEYS], dtype=np.float32)
                X.append(feat)
                y.append(float(d["score_delta"]) / 1e4)  # normalize
                break
    return np.stack(X), np.array(y, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("build_ranker_ez.pkl"))
    ap.add_argument("--out", type=Path, default=Path("build_ranker.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    X, y = build_chosen_dataset(args.data)
    print(f"Samples: {len(X)}, mean target: {y.mean():.3f}, std: {y.std():.3f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    # Normalize features
    mean = X_t.mean(dim=0, keepdim=True)
    std = X_t.std(dim=0, keepdim=True) + 1e-6
    X_norm = (X_t - mean) / std

    model = BuildRankerMLP(input_dim=len(FEATURE_KEYS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = len(X_norm)
    idx = torch.randperm(n)
    val_size = n // 5
    val_idx, train_idx = idx[:val_size], idx[val_size:]

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_idx))
        batches = 0
        total_loss = 0.0
        for i in range(0, len(train_idx), args.batch):
            b = train_idx[perm[i:i + args.batch]]
            xb, yb = X_norm[b], y_t[b]
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1
        model.eval()
        with torch.no_grad():
            val_pred = model(X_norm[val_idx])
            val_loss = F.mse_loss(val_pred, y_t[val_idx]).item()
        print(f"epoch {epoch}: train={total_loss/batches:.4f} val={val_loss:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean.cpu().numpy(),
        "std": std.cpu().numpy(),
        "feature_keys": FEATURE_KEYS,
    }, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
