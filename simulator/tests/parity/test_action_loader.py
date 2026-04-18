from pathlib import Path
from simulator.parity.action_loader import load_actions_for_turn

DUMP = Path("dump")


def test_load_actions_returns_commandset():
    cs = load_actions_for_turn(DUMP, round_id=1, turn_no=1)
    assert hasattr(cs, "commands")
