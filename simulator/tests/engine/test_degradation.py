from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, Construction
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def test_isolated_plantation_loses_ds_per_turn():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["iso"] = Plantation("iso", (20, 20), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}  # iso NOT connected
    s2 = tick(s, CommandSet())
    assert s2.plantations["iso"].hp == 50 - cfg.ds


def test_connected_plantation_not_degraded():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    s2 = tick(s, CommandSet())
    assert s2.plantations["p1"].hp == 50


def test_construction_without_progress_degrades():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.constructions[(10, 10)] = Construction(
        position=(10, 10), progress=20,
        last_progress_turn=s.turn_no - 2,
        contributors_last_turn={},
    )
    s2 = tick(s, CommandSet())
    assert s2.constructions[(10, 10)].progress == 20 - cfg.ds


def test_construction_with_recent_progress_not_degraded():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.constructions[(10, 10)] = Construction(
        position=(10, 10), progress=20,
        last_progress_turn=s.turn_no,  # progress this turn
        contributors_last_turn={},
    )
    s2 = tick(s, CommandSet())
    assert s2.constructions[(10, 10)].progress == 20
