from simulator.inference.adapter import api_to_state, state_to_payload
from simulator.engine.commands import Command, CommandSet
from simulator.engine.types import UpgradeName


def test_api_to_state_basic():
    resp = {
        "turnNo": 5,
        "size": [100, 100],
        "actionRange": 2,
        "plantations": [{
            "id": "p1", "position": [10, 10], "isMain": True,
            "hp": 50, "isIsolated": False, "immunityUntilTurn": 0,
        }],
        "enemy": [], "mountains": [], "cells": [], "construction": [],
        "beavers": [], "meteoForecasts": [],
        "plantationUpgrades": {
            "points": 0, "intervalTurns": 30, "turnsUntilPoints": 25,
            "maxPoints": 15, "tiers": [],
        },
    }
    s = api_to_state(resp)
    assert s.turn_no == 5
    assert s.hq_id == "p1"


def test_state_to_payload_build_command():
    cs = CommandSet(commands=[Command((1, 1), (1, 1), (2, 1))])
    payload = state_to_payload(cs)
    assert payload["command"][0]["path"] == [[1, 1], [1, 1], [2, 1]]
