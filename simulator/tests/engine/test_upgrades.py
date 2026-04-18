from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.commands import CommandSet
from simulator.engine.entities import UpgradeTier
from simulator.engine.types import UpgradeName
from simulator.engine.rules import tick


def test_upgrade_points_accrue_each_interval():
    cfg = GameConfig(upgrade_interval=3, round_turns=10)
    s = EngineState.new(cfg, seed=0)
    for _ in range(3):
        s = tick(s, CommandSet())
    assert s.upgrades.points == 1
    assert s.upgrades.turns_until_points == cfg.upgrade_interval


def test_upgrade_points_capped_at_max():
    cfg = GameConfig(upgrade_interval=1, max_upgrade_points=2)
    s = EngineState.new(cfg, seed=0)
    for _ in range(10):
        s = tick(s, CommandSet())
    assert s.upgrades.points == 2


def test_apply_upgrade_increments_tier_and_spends_point():
    cfg = GameConfig(upgrade_interval=1)
    s = EngineState.new(cfg, seed=0)
    s.upgrades.points = 1
    s2 = tick(s, CommandSet(upgrade=UpgradeName.REPAIR_POWER))
    assert s2.upgrades.tiers[UpgradeName.REPAIR_POWER].current == 1
    assert s2.upgrades.points == 0


def test_cannot_upgrade_beyond_max():
    cfg = GameConfig(upgrade_interval=1)
    s = EngineState.new(cfg, seed=0)
    s.upgrades.points = 1
    s.upgrades.tiers[UpgradeName.REPAIR_POWER] = UpgradeTier(current=3, max=3)
    s2 = tick(s, CommandSet(upgrade=UpgradeName.REPAIR_POWER))
    assert s2.upgrades.tiers[UpgradeName.REPAIR_POWER].current == 3
    assert s2.upgrades.points == 1  # point NOT spent


def test_cannot_upgrade_without_points():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.upgrades.points = 0
    s2 = tick(s, CommandSet(upgrade=UpgradeName.REPAIR_POWER))
    assert s2.upgrades.tiers[UpgradeName.REPAIR_POWER].current == 0
