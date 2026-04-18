from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def _mk(hp_p1: int = 30, hp_hq: int = 50):
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, hp_hq, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, hp_p1, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    return s, cfg


def test_repair_heals_by_rs():
    s, cfg = _mk(hp_p1=30)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["p1"].hp == 30 + cfg.rs


def test_repair_capped_at_mhp():
    s, cfg = _mk(hp_p1=48)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["p1"].hp == cfg.mhp


def test_cannot_repair_self():
    s, cfg = _mk(hp_hq=30)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 5))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["hq"].hp == 30


def test_invalid_command_noop():
    s, cfg = _mk(hp_p1=30)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 10))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["p1"].hp == 30


def test_relay_penalty_reduces_repair():
    s, cfg = _mk(hp_p1=20)
    s.plantations["p2"] = Plantation("p2", (4, 5), False, 50, 0, 0)
    s.connected_ids.add("p2")
    c1 = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    c2 = Command(author=(4, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[c1, c2]))
    assert s2.plantations["p1"].hp == 20 + cfg.rs + max(0, cfg.rs - 1)
