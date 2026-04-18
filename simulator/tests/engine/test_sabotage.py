from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def _mk_with_enemy(enemy_hp=20):
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.enemy_ids = {"e1"}
    s.plantations["e1"] = Plantation("e1", (5, 6), False, enemy_hp, 0, 0)
    return s, cfg


def test_sabotage_reduces_enemy_hp_by_se():
    s, cfg = _mk_with_enemy(enemy_hp=20)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["e1"].hp == 20 - cfg.se


def test_sabotage_kill_awards_cell_reward():
    s, cfg = _mk_with_enemy(enemy_hp=3)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert "e1" not in s2.plantations
    # HQ at (5,5) terraforms for 10 * ts = 50
    assert s2.score == cfg.normal_cell_reward + 10 * cfg.ts


def test_sabotage_boosted_cell_reward():
    s, cfg = _mk_with_enemy(enemy_hp=3)
    s.plantations["hq"].position = (6, 7)
    s.plantations["e1"].position = (7, 7)
    cmd = Command(author=(6, 7), relay=(6, 7), target=(7, 7))
    s2 = tick(s, CommandSet(commands=[cmd]))
    # Enemy at (7,7) boosted → 1500 reward
    # HQ at (6,7) is not boosted (6%7!=0), so terraforms for 10 * ts = 50
    assert s2.score == cfg.boosted_cell_reward + 10 * cfg.ts


def test_cannot_sabotage_immune_enemy():
    s, cfg = _mk_with_enemy(enemy_hp=20)
    s.plantations["e1"].immunity_until_turn = 999
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["e1"].hp == 20
