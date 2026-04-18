from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def _mk():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 30, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    return s, cfg


def test_relocate_to_orthogonal_neighbor():
    s, cfg = _mk()
    s2 = tick(s, CommandSet(relocate=[(5, 5), (5, 6)]))
    assert s2.hq_id == "p1"
    assert s2.plantations["p1"].is_main is True
    assert s2.plantations["hq"].is_main is False


def test_hp_unchanged_after_relocate():
    s, cfg = _mk()
    s2 = tick(s, CommandSet(relocate=[(5, 5), (5, 6)]))
    assert s2.plantations["hq"].hp == 50
    assert s2.plantations["p1"].hp == 30


def test_relocate_to_non_neighbor_ignored():
    s, cfg = _mk()
    s.plantations["p1"].position = (7, 8)
    s2 = tick(s, CommandSet(relocate=[(5, 5), (7, 8)]))
    assert s2.hq_id == "hq"


def test_relocate_to_diagonal_ignored():
    s, cfg = _mk()
    s.plantations["p1"].position = (6, 6)
    s2 = tick(s, CommandSet(relocate=[(5, 5), (6, 6)]))
    assert s2.hq_id == "hq"
