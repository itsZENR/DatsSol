from simulator.env.datssol_env import DatssolEnv
from simulator.env.action_space import ActionKind


def test_reset_returns_obs_and_info():
    env = DatssolEnv(seed=0, map_size=(30, 30), round_turns=20)
    obs, info = env.reset()
    assert "self" in obs
    assert "action_mask" in info


def test_step_with_end_turn_advances_turn():
    env = DatssolEnv(seed=0, map_size=(30, 30), round_turns=20)
    obs, info = env.reset()
    prev_turn = env.state.turn_no
    action = {
        "kind": int(ActionKind.END_TURN),
        "author_idx": 0, "relay_idx": 0,
        "target_offset": 0, "upgrade_id": 0,
    }
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.state.turn_no == prev_turn + 1


def test_episode_terminates_at_round_turns():
    env = DatssolEnv(seed=0, map_size=(30, 30), round_turns=3)
    env.reset()
    end = {"kind": int(ActionKind.END_TURN), "author_idx": 0,
           "relay_idx": 0, "target_offset": 0, "upgrade_id": 0}
    terminated = False
    truncated = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(end)
        if terminated or truncated:
            break
    assert terminated or truncated


def test_max_micro_per_turn_forces_end_turn():
    env = DatssolEnv(seed=0, map_size=(30, 30), round_turns=100,
                    max_micro_per_turn=5)
    env.reset()
    noop = {"kind": int(ActionKind.NOOP), "author_idx": 0,
            "relay_idx": 0, "target_offset": 0, "upgrade_id": 0}
    prev_turn = env.state.turn_no
    for _ in range(6):
        env.step(noop)
    assert env.state.turn_no == prev_turn + 1
