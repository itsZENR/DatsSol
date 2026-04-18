from simulator.engine.types import Pos, UpgradeName
from simulator.engine.entities import (
    Plantation, Construction, Beaver, Storm, CellState, UpgradesState, UpgradeTier,
)


def test_pos_is_int_tuple():
    p: Pos = (3, 5)
    assert isinstance(p[0], int) and isinstance(p[1], int)


def test_upgrade_names_complete():
    names = {u.value for u in UpgradeName}
    expected = {
        "repair_power", "max_hp", "settlement_limit",
        "signal_range", "vision_range",
        "decay_mitigation", "earthquake_mitigation", "beaver_damage_mitigation",
    }
    assert names == expected


def test_plantation_roundtrip():
    p = Plantation(id="p1", position=(3, 4), is_main=True, hp=50,
                   immunity_until_turn=3, built_on_turn=0)
    assert p.position == (3, 4)
    assert p.is_main is True


def test_construction_roundtrip():
    c = Construction(position=(5, 5), progress=10, last_progress_turn=2,
                     contributors_last_turn={})
    assert c.progress == 10


def test_beaver_roundtrip():
    b = Beaver(id="b1", position=(10, 10), hp=95,
               last_hit_by={}, total_damage_this_turn={})
    assert b.hp == 95


def test_storm_roundtrip():
    s = Storm(id="s1", position=(100, 100), radius=30, speed=10,
              direction=(1, 1), forming_left=0)
    assert s.radius == 30


def test_cellstate_defaults():
    c = CellState(progress=0, completed_on_turn=None)
    assert c.progress == 0
    assert c.completed_on_turn is None


def test_upgrades_state():
    tiers = {UpgradeName.REPAIR_POWER: UpgradeTier(current=1, max=3)}
    u = UpgradesState(points=2, turns_until_points=25, tiers=tiers)
    assert u.tiers[UpgradeName.REPAIR_POWER].current == 1
