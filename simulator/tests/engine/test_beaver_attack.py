from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, Beaver
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def _mk(beaver_hp=100):
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.beavers["b1"] = Beaver(id="b1", position=(5, 6), hp=beaver_hp,
                             last_hit_by={}, total_damage_this_turn={})
    return s, cfg


def test_beaver_attack_reduces_hp_by_be():
    s, cfg = _mk(beaver_hp=100)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    # TODO: after Task 16 (_resolve_beaver_damage with regen), change to 100 - cfg.be + cfg.beaver_regen
    assert s2.beavers["b1"].hp == 100 - cfg.be


def test_beaver_kill_awards_20x():
    s, cfg = _mk(beaver_hp=3)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert "b1" not in s2.beavers
    # (5,6) not boosted → normal_cell_reward * 20
    assert s2.score == cfg.normal_cell_reward * cfg.beaver_kill_multiplier


def test_beaver_kill_on_boosted_cell():
    s, cfg = _mk(beaver_hp=3)
    s.beavers["b1"].position = (7, 7)
    s.plantations["hq"].position = (7, 6)
    cmd = Command(author=(7, 6), relay=(7, 6), target=(7, 7))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.score == cfg.boosted_cell_reward * cfg.beaver_kill_multiplier
