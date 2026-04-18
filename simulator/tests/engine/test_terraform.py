from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, CellState
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def test_connected_plantation_terraforms_and_scores():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s2 = tick(s, CommandSet())
    assert s2.cells[(5, 5)].progress == cfg.ts
    # Points: 10 per % on normal cell → 10 * TS
    assert s2.score == 10 * cfg.ts


def test_boosted_cell_gives_15x_per_percent():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (7, 7), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s2 = tick(s, CommandSet())
    assert s2.score == 15 * cfg.ts


def test_isolated_plantation_terraforms_but_no_score():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["iso"] = Plantation("iso", (20, 20), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s2 = tick(s, CommandSet())
    assert s2.cells[(20, 20)].progress == cfg.ts
    assert s2.score == 10 * cfg.ts


def test_cell_reaches_100_plantation_disappears():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.cells[(5, 5)] = CellState(progress=97, completed_on_turn=None)
    s2 = tick(s, CommandSet())
    assert s2.cells[(5, 5)].progress == 100
    assert s2.cells[(5, 5)].completed_on_turn == s.turn_no
    assert "hq" not in s2.plantations


def test_cell_degrades_after_delay():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    # Cell completed exactly cell_degradation_delay turns ago → starts degrading
    s.cells[(10, 10)] = CellState(
        progress=100, completed_on_turn=s.turn_no - cfg.cell_degradation_delay
    )
    s2 = tick(s, CommandSet())
    assert s2.cells[(10, 10)].progress == 100 - cfg.cell_degradation_per_turn
