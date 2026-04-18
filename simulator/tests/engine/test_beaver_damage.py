from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, Beaver
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def _mk(beaver_hp=100, plant_hp=50):
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, plant_hp, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.beavers["b1"] = Beaver(id="b1", position=(5, 7), hp=beaver_hp,
                             last_hit_by={}, total_damage_this_turn={})
    return s, cfg


def test_beaver_damages_plantation_in_range():
    # AR=2: (5,5) -> (5,7) distance 2, in range
    s, cfg = _mk(plant_hp=50)
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50 - cfg.beaver_damage


def test_beaver_does_not_damage_out_of_range():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.beavers["b1"] = Beaver("b1", (5, 8), 100, {}, {})
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50


def test_beaver_regenerates_up_to_max():
    s, cfg = _mk(beaver_hp=90)
    s2 = tick(s, CommandSet())
    assert s2.beavers["b1"].hp == min(cfg.beaver_max_hp, 90 + cfg.beaver_regen)


def test_beaver_does_not_damage_immune_plantation():
    s, cfg = _mk(plant_hp=50)
    s.plantations["hq"].immunity_until_turn = 999
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50
