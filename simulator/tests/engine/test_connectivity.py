from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def test_bfs_marks_connected():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, 0)
    s.plantations["iso"] = Plantation("iso", (20, 20), False, 50, 0, 0)
    s.hq_id = "hq"
    s2 = tick(s, CommandSet())
    assert "hq" in s2.connected_ids
    assert "p1" in s2.connected_ids
    assert "iso" not in s2.connected_ids
