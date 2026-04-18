from simulator.engine.commands import Command, CommandSet
from simulator.engine.types import UpgradeName


def test_command_path_format():
    c = Command(author=(1, 1), relay=(1, 1), target=(2, 1))
    assert c.to_path() == [[1, 1], [1, 1], [2, 1]]


def test_commandset_empty_payload_fallback():
    cs = CommandSet()
    p = cs.to_payload()
    assert p == {"plantationUpgrade": ""}


def test_commandset_with_build():
    cs = CommandSet(commands=[Command((1, 1), (1, 1), (2, 1))])
    p = cs.to_payload()
    assert p["command"] == [{"path": [[1, 1], [1, 1], [2, 1]]}]


def test_commandset_with_upgrade():
    cs = CommandSet(upgrade=UpgradeName.REPAIR_POWER)
    p = cs.to_payload()
    assert p["plantationUpgrade"] == "repair_power"


def test_commandset_with_relocate():
    cs = CommandSet(relocate=[(1, 1), (1, 2)])
    p = cs.to_payload()
    assert p["relocateMain"] == [[1, 1], [1, 2]]


def test_from_api_payload_roundtrip():
    payload = {
        "command": [{"path": [[1, 1], [1, 1], [2, 1]]}],
        "plantationUpgrade": "repair_power",
        "relocateMain": [[0, 0], [0, 1]],
    }
    cs = CommandSet.from_payload(payload)
    assert cs.commands[0].author == (1, 1)
    assert cs.upgrade == UpgradeName.REPAIR_POWER
    assert cs.relocate == [(0, 0), (0, 1)]
