from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command
from simulator.engine.validation import is_valid_command


def _mk_state_with_two_plants():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    hq = Plantation(id="hq", position=(5, 5), is_main=True, hp=50,
                    immunity_until_turn=0, built_on_turn=0)
    p2 = Plantation(id="p2", position=(5, 6), is_main=False, hp=50,
                    immunity_until_turn=0, built_on_turn=0)
    s.plantations["hq"] = hq
    s.plantations["p2"] = p2
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p2"}
    return s


def test_author_must_be_own_and_connected():
    s = _mk_state_with_two_plants()
    c = Command(author=(99, 99), relay=(5, 5), target=(5, 4))
    assert is_valid_command(s, c) is False


def test_target_within_ar_of_relay():
    s = _mk_state_with_two_plants()
    c = Command(author=(5, 5), relay=(5, 5), target=(8, 5))
    assert is_valid_command(s, c) is False


def test_relay_within_sr_of_author():
    s = _mk_state_with_two_plants()
    p3 = Plantation(id="p3", position=(5, 10), is_main=False, hp=50,
                    immunity_until_turn=0, built_on_turn=0)
    s.plantations["p3"] = p3
    s.connected_ids.add("p3")
    c = Command(author=(5, 5), relay=(5, 10), target=(5, 10))
    assert is_valid_command(s, c) is False


def test_valid_direct_command():
    s = _mk_state_with_two_plants()
    c = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    assert is_valid_command(s, c) is True
