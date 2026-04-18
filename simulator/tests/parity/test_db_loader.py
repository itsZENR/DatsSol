from pathlib import Path
from simulator.parity.db_loader import load_round_ids, load_state_at, _load_round_cache


DUMP = Path("dump")


def test_load_round_ids_non_empty():
    ids = load_round_ids(DUMP)
    assert len(ids) > 0


def test_load_state_at_turn_zero():
    ids = load_round_ids(DUMP)
    state = load_state_at(DUMP, ids[0], 0)
    assert state.turn_no == 0
    assert len(state.plantations) >= 0


def test_load_state_cache_reuses_reads():
    ids = load_round_ids(DUMP)
    c1 = _load_round_cache(DUMP, ids[0])
    c2 = _load_round_cache(DUMP, ids[0])
    # Must return the SAME dict object (cached)
    assert c1 is c2
