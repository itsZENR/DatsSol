"""
Attention-based feature extractor for DatssolEnv Dict observation.

Architecture (inspired by AlphaStar entity encoder):
- self entities → per-entity Linear → MultiHead self-attention → masked mean pool
- enemy, beaver, storm entities → per-entity Linear → attention → pool
- global vector → Linear
- local grid → small CNN
- concat all → MLP head → features_dim

This replaces SB3's default MultiInputPolicy flat MLP, which can't exploit entity
invariances or local spatial structure.
"""
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [B, N, D], mask: [B, N] bool. Returns [B, D] — mean over unmasked entities."""
    m = mask.float().unsqueeze(-1)  # [B, N, 1]
    s = (x * m).sum(dim=1)  # [B, D]
    n = m.sum(dim=1).clamp_min(1.0)
    return s / n


class EntitySetEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], mask: [B, N] (True = valid)
        h = self.proj(x)
        key_padding_mask = ~mask  # MultiheadAttention expects True=ignore
        # If all masked, attn errors — pad with at least one valid index
        all_masked = (~mask).all(dim=1)
        if all_masked.any():
            # fake a valid entry at 0 to avoid NaN; real contribution gated by mask later
            fake = key_padding_mask.clone()
            fake[all_masked, 0] = False
            key_padding_mask = fake
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.norm(h + attn_out)
        return masked_mean_pool(h, mask)


class AttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self_f = observation_space["self"].shape[1]
        enemy_f = observation_space["enemy"].shape[1]
        beaver_f = observation_space["beaver"].shape[1]
        storm_f = observation_space["storm"].shape[1]
        global_f = observation_space["global"].shape[0]
        grid_shape = observation_space["grid"].shape  # (C, H, W)

        self.self_enc = EntitySetEncoder(self_f, embed_dim=64, n_heads=4)
        self.enemy_enc = EntitySetEncoder(enemy_f, embed_dim=32, n_heads=2)
        self.beaver_enc = EntitySetEncoder(beaver_f, embed_dim=32, n_heads=2)
        self.storm_enc = EntitySetEncoder(storm_f, embed_dim=32, n_heads=2)

        self.global_mlp = nn.Sequential(
            nn.Linear(global_f, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )

        c = grid_shape[0]
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Determine grid flat size
        with torch.no_grad():
            dummy = torch.zeros(1, *grid_shape)
            grid_flat = self.grid_cnn(dummy).shape[1]
        self.grid_mlp = nn.Sequential(
            nn.Linear(grid_flat, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )

        cat_dim = 64 + 32 + 32 + 32 + 64 + 64
        self.head = nn.Sequential(
            nn.Linear(cat_dim, features_dim), nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        self_mask = obs["self_mask"].bool()
        enemy_mask = obs["enemy_mask"].bool()
        beaver_mask = obs["beaver_mask"].bool()
        storm_mask = obs["storm_mask"].bool()

        s = self.self_enc(obs["self"], self_mask)
        e = self.enemy_enc(obs["enemy"], enemy_mask)
        b = self.beaver_enc(obs["beaver"], beaver_mask)
        st = self.storm_enc(obs["storm"], storm_mask)
        g = self.global_mlp(obs["global"])
        gr = self.grid_mlp(self.grid_cnn(obs["grid"]))

        x = torch.cat([s, e, b, st, g, gr], dim=-1)
        return self.head(x)
