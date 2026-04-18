from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def test_full_turn_repair_and_score_together():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=42)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 30, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    # 1) repair: p1.hp 30 -> 30 + RS
    assert s2.plantations["p1"].hp == 30 + cfg.rs
    # 2) score from terraform of BOTH plantations: 10 * TS * 2
    assert s2.score == 10 * cfg.ts * 2
    assert s2.turn_no == 1


def test_determinism_two_runs_same_seed():
    def run(seed):
        cfg = GameConfig()
        s = EngineState.new(cfg, seed=seed)
        s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
        s.hq_id = "hq"
        s.connected_ids = {"hq"}
        for _ in range(50):
            s = tick(s, CommandSet())
        return s.score, s.turn_no, len(s.plantations), s.earthquake_turn

    r1 = run(123)
    r2 = run(123)
    assert r1 == r2
