import numpy as np
from gymnasium import spaces
from simulator.env.datssol_env import DatssolEnv
from simulator.env.flatten_action_wrapper import FlattenActionEnv
from simulator.env.action_space import (
    ActionKind, MAX_PLANT, TARGET_OFFSET_DIM, N_UPGRADES,
)


def _mk():
    base = DatssolEnv(seed=0, map_size=(20, 20), round_turns=10)
    return FlattenActionEnv(base)


def test_action_space_is_multidiscrete():
    env = _mk()
    assert isinstance(env.action_space, spaces.MultiDiscrete)
    expected = np.array([
        len(ActionKind), MAX_PLANT, MAX_PLANT, TARGET_OFFSET_DIM, N_UPGRADES,
    ])
    assert np.array_equal(env.action_space.nvec, expected)


def test_step_with_flat_action():
    env = _mk()
    env.reset()
    prev_turn = env.env.state.turn_no
    # END_TURN flat vector
    flat = np.array([int(ActionKind.END_TURN), 0, 0, 0, 0], dtype=np.int64)
    obs, r, term, trunc, info = env.step(flat)
    assert env.env.state.turn_no == prev_turn + 1


def test_action_masks_flat_concat():
    env = _mk()
    _, info = env.reset()
    masks = env.action_masks()
    expected_len = (len(ActionKind) + MAX_PLANT + MAX_PLANT +
                    TARGET_OFFSET_DIM + N_UPGRADES)
    assert masks.shape == (expected_len,)
    # END_TURN always valid
    assert masks[int(ActionKind.END_TURN)] == 1
