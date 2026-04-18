"""Wraps DatssolEnv to expose a MultiDiscrete flat action space for MaskablePPO."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulator.env.datssol_env import DatssolEnv
from simulator.env.action_space import (
    ActionKind, MAX_PLANT, TARGET_OFFSET_DIM, N_UPGRADES,
)


class FlattenActionEnv(gym.Wrapper):
    """Dict action space -> MultiDiscrete action space.

    The order is: [kind, author_idx, relay_idx, target_offset, upgrade_id].
    `action_masks()` returns a concatenated binary vector compatible with
    sb3-contrib MaskablePPO.
    """

    NVEC = np.array([
        len(ActionKind), MAX_PLANT, MAX_PLANT, TARGET_OFFSET_DIM, N_UPGRADES,
    ], dtype=np.int64)

    def __init__(self, env: DatssolEnv):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete(self.NVEC)
        self._last_info: dict = {}

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_info = info
        return obs, info

    def step(self, action):
        a = {
            "kind": int(action[0]),
            "author_idx": int(action[1]),
            "relay_idx": int(action[2]),
            "target_offset": int(action[3]),
            "upgrade_id": int(action[4]),
        }
        obs, r, term, trunc, info = self.env.step(a)
        self._last_info = info
        return obs, r, term, trunc, info

    def action_masks(self) -> np.ndarray:
        """Return a flat 1-D mask concatenated over all sub-actions,
        as expected by sb3-contrib MaskablePPO with MultiDiscrete."""
        m = self._last_info.get("action_mask")
        if m is None:
            # fallback: all valid
            return np.ones(int(self.NVEC.sum()), dtype=np.int8)
        return np.concatenate([
            m["kind"], m["author_idx"], m["relay_idx"],
            m["target_offset"], m["upgrade_id"],
        ]).astype(np.int8)
