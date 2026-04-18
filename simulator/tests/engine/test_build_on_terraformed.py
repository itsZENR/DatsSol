from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, CellState
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def _mk():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    return s, cfg


def test_cannot_build_on_fully_terraformed_cell():
    s, cfg = _mk()
    # (5,6) already terraformed to 100%
    s.cells[(5, 6)] = CellState(progress=100, completed_on_turn=s.turn_no)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    # Build blocked — no construction created
    assert (5, 6) not in s2.constructions


def test_can_build_on_partially_degraded_cell():
    s, cfg = _mk()
    # Cell was completed but already degraded back to 80
    s.cells[(5, 6)] = CellState(progress=80, completed_on_turn=None)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    # Build proceeds
    assert (5, 6) in s2.constructions
    assert s2.constructions[(5, 6)].progress == cfg.cs


def test_can_build_on_never_terraformed_cell():
    s, cfg = _mk()
    # No cell entry at all
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert (5, 6) in s2.constructions
