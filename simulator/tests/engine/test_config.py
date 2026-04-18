from simulator.engine.config import GameConfig


def test_default_config_values():
    cfg = GameConfig()
    assert cfg.mhp == 50
    assert cfg.ts == 5
    assert cfg.cs == 5
    assert cfg.rs == 5
    assert cfg.se == 5
    assert cfg.be == 5
    assert cfg.ds == 10
    assert cfg.ar == 2
    assert cfg.sr == 3
    assert cfg.vr == 3
    assert cfg.base_plantation_limit == 30
    assert cfg.max_plantation_limit == 40
    assert cfg.upgrade_interval == 30
    assert cfg.max_upgrade_points == 15
    assert cfg.beaver_max_hp == 100
    assert cfg.beaver_regen == 5
    assert cfg.beaver_damage == 15
    assert cfg.earthquake_damage == 10
    assert cfg.storm_damage == 2
    assert cfg.storm_forming_turns == 5
    assert cfg.storm_prob == 0.10
    assert cfg.earthquake_prob == 0.05
    assert cfg.immunity_turns == 3
    assert cfg.cell_degradation_delay == 20
    assert cfg.cell_degradation_per_turn == 10
    assert cfg.round_turns == 600
    assert cfg.normal_cell_reward == 1000
    assert cfg.boosted_cell_reward == 1500
    assert cfg.boosted_mod == 7
    assert cfg.map_width == 278
    assert cfg.map_height == 278
    assert cfg.beaver_kill_multiplier == 20


def test_config_frozen_immutable():
    cfg = GameConfig()
    import dataclasses
    assert dataclasses.is_dataclass(cfg)
    try:
        cfg.mhp = 9999
        raised = False
    except Exception:
        raised = True
    assert raised, "GameConfig must be frozen"
