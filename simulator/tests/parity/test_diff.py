from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.parity.diff import diff_states


def test_diff_identical_states_is_empty():
    s1 = EngineState.new(GameConfig(), seed=0)
    s2 = EngineState.new(GameConfig(), seed=0)
    d = diff_states(s1, s2)
    assert d.is_empty()


def test_diff_plantation_hp_delta():
    s1 = EngineState.new(GameConfig(), seed=0)
    s2 = EngineState.new(GameConfig(), seed=0)
    s1.plantations["p1"] = Plantation("p1", (5, 5), True, 50, 0, 0)
    s2.plantations["p1"] = Plantation("p1", (5, 5), True, 45, 0, 0)
    d = diff_states(s1, s2)
    assert not d.is_empty()
    assert "plantation_hp" in d.field_counts


def test_diff_score_mismatch():
    s1 = EngineState.new(GameConfig(), seed=0)
    s2 = EngineState.new(GameConfig(), seed=0)
    s1.score = 100
    s2.score = 200
    d = diff_states(s1, s2)
    assert "score" in d.field_counts
