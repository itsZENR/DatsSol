import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional

from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick
from simulator.env.action_space import (
    ActionKind, build_action_space, decode_action,
    target_offset_to_delta, MAX_PLANT,
)
from simulator.env.action_mask import build_action_mask, TurnContext
from simulator.env.obs_encoder import (
    encode_observation, F_SELF, F_ENEMY, F_BEAVER, F_STORM, F_GLOBAL,
    MAX_ENEMY, MAX_BEAVER, MAX_STORM, GRID_CHANNELS, GRID_SIZE,
)
from simulator.env.reward import compute_reward
from simulator.engine.types import UpgradeName


class DatssolEnv(gym.Env):
    """Single-agent Gymnasium env with autoregressive action space."""

    def __init__(
        self,
        seed: int = 0,
        map_size: tuple[int, int] = (278, 278),
        round_turns: int = 600,
        max_micro_per_turn: int = 45,
        game_config_overrides: dict | None = None,
    ):
        super().__init__()
        self._seed = seed
        self._map_size = map_size
        self._round_turns = round_turns
        self._max_micro_per_turn = max_micro_per_turn
        self._game_config_overrides = game_config_overrides or {}

        self.action_space = build_action_space()
        self.observation_space = spaces.Dict({
            "self": spaces.Box(-10, 10, (MAX_PLANT, F_SELF), np.float32),
            "self_mask": spaces.MultiBinary(MAX_PLANT),
            "enemy": spaces.Box(-10, 10, (MAX_ENEMY, F_ENEMY), np.float32),
            "enemy_mask": spaces.MultiBinary(MAX_ENEMY),
            "beaver": spaces.Box(-10, 10, (MAX_BEAVER, F_BEAVER), np.float32),
            "beaver_mask": spaces.MultiBinary(MAX_BEAVER),
            "storm": spaces.Box(-10, 10, (MAX_STORM, F_STORM), np.float32),
            "storm_mask": spaces.MultiBinary(MAX_STORM),
            "global": spaces.Box(-10, 10, (F_GLOBAL,), np.float32),
            "grid": spaces.Box(0, 1, (GRID_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32),
        })

        self.state: Optional[EngineState] = None
        self._ctx: TurnContext = TurnContext()
        self._pending_commands: list[Command] = []
        self._pending_upgrade: Optional[UpgradeName] = None
        self._pending_relocate: list = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        cfg = GameConfig(
            map_width=self._map_size[0],
            map_height=self._map_size[1],
            round_turns=self._round_turns,
            **self._game_config_overrides,
        )
        self.state = EngineState.new(cfg, seed=self._seed)
        self._populate_initial_map()
        # First tick spawns the HQ via _resolve_respawn
        self.state = tick(self.state, CommandSet())
        self._reset_turn_context()
        obs = encode_observation(self.state, self._ctx.authors_used)
        info = {"action_mask": build_action_mask(self.state, self._ctx)}
        return obs, info

    def _populate_initial_map(self):
        """Spawn initial beavers and mountains based on config."""
        from simulator.engine.entities import Beaver
        cfg = self.state.config
        w, h = cfg.map_width, cfg.map_height
        rng = self.state.rng

        # Mountains: uniform random cells
        if cfg.initial_mountain_density > 0:
            n = int(w * h * cfg.initial_mountain_density)
            mtn = set()
            while len(mtn) < n:
                x = int(rng.integers(0, w))
                y = int(rng.integers(0, h))
                # Keep (1,1) and nearby clear for HQ spawn
                if abs(x - 1) + abs(y - 1) < 5:
                    continue
                mtn.add((x, y))
            self.state.mountains = frozenset(mtn)

        # Beavers: placed at random non-mountain cells, not too close to HQ (1,1)
        if cfg.initial_beavers > 0:
            placed = 0
            attempts = 0
            while placed < cfg.initial_beavers and attempts < 1000:
                attempts += 1
                x = int(rng.integers(0, w))
                y = int(rng.integers(0, h))
                if (x, y) in self.state.mountains:
                    continue
                if abs(x - 1) + abs(y - 1) < 8:  # not adjacent to HQ
                    continue
                bid = f"b{placed}"
                self.state.beavers[bid] = Beaver(
                    id=bid, position=(x, y), hp=cfg.beaver_max_hp,
                    last_hit_by={}, total_damage_this_turn={},
                )
                placed += 1

    def _reset_turn_context(self):
        self._ctx = TurnContext()
        self._pending_commands = []
        self._pending_upgrade = None
        self._pending_relocate = []

    def step(self, action: dict):
        assert self.state is not None
        decoded = decode_action(action)
        self._ctx.micro_steps += 1

        end_turn = False
        if decoded.kind == ActionKind.END_TURN:
            end_turn = True
        elif decoded.kind == ActionKind.NOOP:
            pass
        elif decoded.kind == ActionKind.SET_UPGRADE:
            upgrade_list = list(UpgradeName)
            if decoded.upgrade_id < len(upgrade_list):
                self._pending_upgrade = upgrade_list[decoded.upgrade_id]
                self._ctx.upgrade_set = True
        elif decoded.kind == ActionKind.SET_RELOCATE:
            hq = self.state.get_hq()
            if hq is not None:
                pos_to_id = {p.position: pid for pid, p in self.state.plantations.items()}
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nb = (hq.position[0] + dx, hq.position[1] + dy)
                    if nb in pos_to_id and pos_to_id[nb] in self.state.connected_ids:
                        self._pending_relocate = [hq.position, nb]
                        break
                self._ctx.relocate_set = True
        elif decoded.kind == ActionKind.COMMAND:
            author_ids = self._ctx.ordered_author_ids(self.state)
            if decoded.author_idx < len(author_ids):
                author_id = author_ids[decoded.author_idx]
                if author_id not in self._ctx.authors_used:
                    author = self.state.plantations[author_id]
                    relay_id = (
                        author_ids[decoded.relay_idx]
                        if decoded.relay_idx < len(author_ids)
                        else author_id
                    )
                    relay = self.state.plantations[relay_id]
                    dx, dy = target_offset_to_delta(decoded.target_offset)
                    target = (relay.position[0] + dx, relay.position[1] + dy)
                    self._pending_commands.append(
                        Command(author=author.position,
                                relay=relay.position, target=target)
                    )
                    self._ctx.authors_used.add(author_id)

        if self._ctx.micro_steps >= self._max_micro_per_turn:
            end_turn = True

        if end_turn:
            prev_score = self.state.score
            cs = CommandSet(
                commands=self._pending_commands,
                upgrade=self._pending_upgrade,
                relocate=self._pending_relocate,
            )
            self.state = tick(self.state, cs)
            reward = compute_reward(
                prev_score, self.state.score,
                self.state.events_this_turn,
                state_after=self.state,
            )
            self._reset_turn_context()
        else:
            reward = 0.0

        terminated = (self.state.turn_no >= self._round_turns) or (
            self.state.hq_id is None and self.state.score < 0
        )
        truncated = False
        obs = encode_observation(self.state, self._ctx.authors_used)
        info = {"action_mask": build_action_mask(self.state, self._ctx)}
        return obs, float(reward), terminated, truncated, info
