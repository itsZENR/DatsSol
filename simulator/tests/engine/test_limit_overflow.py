from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def _setup_at_limit(limit=3):
    cfg = GameConfig(base_plantation_limit=limit, max_plantation_limit=limit)
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, built_on_turn=10)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, built_on_turn=5)
    s.plantations["p2"] = Plantation("p2", (5, 4), False, 50, 0, built_on_turn=7)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1", "p2"}
    return s, cfg


def test_overflow_removes_oldest_non_main():
    s, cfg = _setup_at_limit(limit=3)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(6, 5))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert "p1" not in s2.plantations
    assert "p2" in s2.plantations
    assert "hq" in s2.plantations


def test_overflow_can_destroy_hq():
    cfg = GameConfig(base_plantation_limit=2, max_plantation_limit=2)
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, built_on_turn=0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, built_on_turn=5)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 4))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert "hq" not in s2.plantations


def test_no_overflow_when_under_limit():
    cfg = GameConfig(base_plantation_limit=10, max_plantation_limit=10)
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, built_on_turn=0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert "hq" in s2.plantations
    assert (5, 6) in s2.constructions
