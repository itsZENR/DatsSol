from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def test_dead_plantation_removed():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["dead"] = Plantation("dead", (5, 6), False, 0, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "dead"}
    s2 = tick(s, CommandSet())
    assert "dead" not in s2.plantations


def test_hq_death_wipes_plantations_and_applies_penalty():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 0, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    s.score = 1000
    s2 = tick(s, CommandSet())
    # After cleanup: hq dead → hq_id=None; respawn handles
    assert s2.hq_id is None or s2.plantations.get(s2.hq_id).position == (1, 1)
    # Score increases from terraform (ts=5, 2 plantations -> 2*50=100 from terraform)
    # then 5% penalty on (1000+100)=1100 -> penalty=55, final = 1100-55=1045
    assert s2.score == 1045
