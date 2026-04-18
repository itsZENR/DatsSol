from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.env.obs_encoder import (
    encode_observation, F_SELF, F_ENEMY, F_BEAVER, F_GLOBAL,
    MAX_PLANT, MAX_ENEMY, MAX_BEAVER, GRID_CHANNELS, GRID_SIZE,
)


def _mk():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (50, 50), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (50, 51), False, 40, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    return s, cfg


def test_obs_shapes_correct():
    s, _ = _mk()
    obs = encode_observation(s)
    assert obs["self"].shape == (MAX_PLANT, F_SELF)
    assert obs["self_mask"].shape == (MAX_PLANT,)
    assert obs["enemy"].shape == (MAX_ENEMY, F_ENEMY)
    assert obs["beaver"].shape == (MAX_BEAVER, F_BEAVER)
    assert obs["global"].shape == (F_GLOBAL,)
    assert obs["grid"].shape == (GRID_CHANNELS, GRID_SIZE, GRID_SIZE)


def test_obs_self_masks_fits_count():
    s, _ = _mk()
    obs = encode_observation(s)
    assert obs["self_mask"].sum() == 2


def test_self_positions_relative_to_hq():
    s, _ = _mk()
    obs = encode_observation(s)
    positions = obs["self"][:2, :2]
    assert abs(positions).sum() < 0.1


def test_global_contains_turn_number():
    s, cfg = _mk()
    s.turn_no = 60
    obs = encode_observation(s)
    assert abs(obs["global"][0] - 0.1) < 1e-6
