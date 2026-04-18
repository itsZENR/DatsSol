from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, Storm
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def test_earthquake_damages_all_plantations():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (20, 20), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    s.earthquake_turn = s.turn_no
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50 - cfg.earthquake_damage
    assert s2.plantations["p1"].hp == 50 - cfg.earthquake_damage


def test_storm_damages_plantation_inside_disc():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (100, 100), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.storms.append(Storm(id="s", position=(100, 100), radius=10,
                          speed=5, direction=(1, 1), forming_left=0))
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50 - cfg.storm_damage


def test_storm_does_not_damage_outside_radius():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (100, 100), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.storms.append(Storm("s", (50, 50), radius=10,
                          speed=5, direction=(1, 1), forming_left=0))
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50


def test_storm_cannot_reduce_below_1():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (100, 100), True, 1, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.storms.append(Storm("s", (100, 100), 10, 5, (1, 1), 0))
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 1


def test_forming_storm_does_no_damage():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (100, 100), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.storms.append(Storm("s", (100, 100), 10, 5, (1, 1), forming_left=3))
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50
    assert s2.storms[0].forming_left == 2
