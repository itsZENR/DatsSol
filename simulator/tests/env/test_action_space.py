from gymnasium import spaces
from simulator.env.action_space import (
    ActionKind, build_action_space, decode_action,
    MAX_PLANT, TARGET_OFFSET_DIM, N_UPGRADES,
    target_offset_to_delta, delta_to_target_offset,
)


def test_action_space_is_dict():
    sp = build_action_space()
    assert isinstance(sp, spaces.Dict)
    assert "kind" in sp.spaces
    assert sp["kind"].n == len(ActionKind)
    assert sp["author_idx"].n == MAX_PLANT
    assert sp["relay_idx"].n == MAX_PLANT
    assert sp["target_offset"].n == TARGET_OFFSET_DIM
    assert sp["upgrade_id"].n == N_UPGRADES


def test_decode_action_end_turn():
    a = {"kind": ActionKind.END_TURN.value, "author_idx": 0,
         "relay_idx": 0, "target_offset": 0, "upgrade_id": 0}
    decoded = decode_action(a)
    assert decoded.kind == ActionKind.END_TURN


def test_decode_action_command():
    a = {"kind": ActionKind.COMMAND.value, "author_idx": 2,
         "relay_idx": 3, "target_offset": 12, "upgrade_id": 0}
    decoded = decode_action(a)
    assert decoded.kind == ActionKind.COMMAND
    assert decoded.author_idx == 2
    assert decoded.relay_idx == 3
    assert decoded.target_offset == 12


def test_target_offset_roundtrip():
    # AR=2 → 5x5 grid, 25 offsets. Center offset = (0,0).
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            off = delta_to_target_offset(dx, dy)
            assert 0 <= off < TARGET_OFFSET_DIM
            back_dx, back_dy = target_offset_to_delta(off)
            assert (back_dx, back_dy) == (dx, dy)
