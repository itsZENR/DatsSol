from pathlib import Path
from simulator.parity.db_loader import load_round_ids, load_state_at


DUMP = Path("dump")


def test_load_round_ids_non_empty():
    ids = load_round_ids(DUMP)
    assert len(ids) > 0


def test_load_state_at_turn_zero():
    ids = load_round_ids(DUMP)
    state = load_state_at(DUMP, ids[0], 0)
    assert state.turn_no == 0
    assert len(state.plantations) >= 0
