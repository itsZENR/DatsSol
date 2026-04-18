from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, Construction
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def _mk():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    return s, cfg


def test_build_creates_construction_with_cs_progress():
    s, cfg = _mk()
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert (5, 6) in s2.constructions
    assert s2.constructions[(5, 6)].progress == cfg.cs


def test_two_builders_accumulate_progress():
    s, cfg = _mk()
    s.plantations["p1"] = Plantation("p1", (4, 5), False, 50, 0, 0)
    s.connected_ids.add("p1")
    c1 = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    c2 = Command(author=(4, 5), relay=(4, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[c1, c2]))
    assert s2.constructions[(5, 6)].progress == 2 * cfg.cs


def test_construction_completes_at_50_and_becomes_plantation():
    s, cfg = _mk()
    s.constructions[(5, 6)] = Construction(
        position=(5, 6), progress=45, last_progress_turn=0,
        contributors_last_turn={},
    )
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert (5, 6) not in s2.constructions
    new_ids = [p.id for p in s2.plantations.values() if p.position == (5, 6)]
    assert len(new_ids) == 1
    p = s2.plantations[new_ids[0]]
    assert p.hp == cfg.mhp
    assert p.immunity_until_turn == s.turn_no + cfg.immunity_turns + 1


def test_build_on_mountain_noop():
    s, cfg = _mk()
    s.mountains = frozenset({(5, 6)})
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert (5, 6) not in s2.constructions


def test_build_on_own_plantation_noop():
    s, cfg = _mk()
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 5))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert (5, 5) not in s2.constructions
