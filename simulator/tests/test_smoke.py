"""High-level sanity: env runs a short round without errors; reward is finite."""
from simulator.env.datssol_env import DatssolEnv
from simulator.env.action_space import ActionKind


def test_env_completes_short_round():
    env = DatssolEnv(seed=0, map_size=(20, 20), round_turns=30)
    obs, _ = env.reset()
    end = {
        "kind": int(ActionKind.END_TURN),
        "author_idx": 0, "relay_idx": 0,
        "target_offset": 0, "upgrade_id": 0,
    }
    total = 0.0
    for _ in range(50):
        obs, r, term, trunc, _ = env.step(end)
        total += r
        if term or trunc:
            break
    assert abs(total) < 1e6
