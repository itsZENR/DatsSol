import numpy as np
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, UpgradesState, UpgradeTier
from simulator.engine.types import UpgradeName


def test_engine_state_seeded_rng_deterministic():
    s1 = EngineState.new(config=GameConfig(), seed=42)
    s2 = EngineState.new(config=GameConfig(), seed=42)
    a1 = [s1.rng.random() for _ in range(5)]
    a2 = [s2.rng.random() for _ in range(5)]
    assert a1 == a2


def test_engine_state_new_empty():
    s = EngineState.new(config=GameConfig(), seed=0)
    assert s.turn_no == 0
    assert s.plantations == {}
    assert s.score == 0
    assert s.hq_id is None
    assert isinstance(s.rng, np.random.Generator)


def test_get_hq_returns_plantation_or_none():
    s = EngineState.new(config=GameConfig(), seed=0)
    assert s.get_hq() is None
    p = Plantation(id="hq", position=(3, 3), is_main=True, hp=50,
                   immunity_until_turn=0, built_on_turn=0)
    s.plantations[p.id] = p
    s.hq_id = "hq"
    assert s.get_hq() is p


def test_compute_effective_values_from_upgrades():
    cfg = GameConfig()
    s = EngineState.new(config=cfg, seed=0)
    s.upgrades.tiers[UpgradeName.REPAIR_POWER] = UpgradeTier(current=2, max=3)
    s.upgrades.tiers[UpgradeName.MAX_HP] = UpgradeTier(current=1, max=5)
    s.upgrades.tiers[UpgradeName.SIGNAL_RANGE] = UpgradeTier(current=1, max=99)
    s.upgrades.tiers[UpgradeName.SETTLEMENT_LIMIT] = UpgradeTier(current=3, max=10)
    s.upgrades.tiers[UpgradeName.BEAVER_DAMAGE_MITIGATION] = UpgradeTier(current=2, max=5)
    s.upgrades.tiers[UpgradeName.EARTHQUAKE_MITIGATION] = UpgradeTier(current=1, max=3)

    assert s.effective_rs() == cfg.rs + 2
    assert s.effective_mhp() == cfg.mhp + 10
    assert s.effective_sr() == cfg.sr + 1
    assert s.effective_plantation_limit() == cfg.base_plantation_limit + 3
    assert s.effective_beaver_damage() == cfg.beaver_damage - 4
    assert s.effective_earthquake_damage() == cfg.earthquake_damage - 2


def test_is_boosted_cell():
    s = EngineState.new(config=GameConfig(), seed=0)
    assert s.is_boosted_cell((0, 0)) is True
    assert s.is_boosted_cell((7, 14)) is True
    assert s.is_boosted_cell((3, 7)) is False
    assert s.is_boosted_cell((14, 5)) is False
