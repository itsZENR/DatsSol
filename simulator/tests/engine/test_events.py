from simulator.engine.events import (
    Event, Scored, PlantationDestroyed, BeaverKilled, HQLost,
    ConstructionCompleted, CellTerraformed, UpgradeBought,
)


def test_events_are_frozen_dataclasses():
    e = Scored(amount=100, source="terraform")
    assert e.amount == 100
    try:
        e.amount = 999
        raised = False
    except Exception:
        raised = True
    assert raised


def test_events_union_iterable():
    events: list[Event] = [
        Scored(amount=50, source="terraform"),
        PlantationDestroyed(plantation_id="p1", position=(3, 3), was_main=False),
        BeaverKilled(beaver_id="b1", position=(10, 10), reward=10000),
        HQLost(prev_hq_id="p0", penalty=5),
        ConstructionCompleted(position=(5, 5), new_plantation_id="p2"),
        CellTerraformed(position=(7, 7), reward=1500),
        UpgradeBought(name="repair_power", new_level=1),
    ]
    assert len(events) == 7
