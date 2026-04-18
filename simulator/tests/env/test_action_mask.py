from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.env.action_mask import build_action_mask, TurnContext
from simulator.env.action_space import ActionKind, MAX_PLANT, TARGET_OFFSET_DIM


def _mk():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    return s


def test_end_turn_always_valid():
    s = _mk()
    ctx = TurnContext()
    mask = build_action_mask(s, ctx)
    assert mask["kind"][int(ActionKind.END_TURN)] == 1


def test_used_author_masked_out():
    s = _mk()
    ctx = TurnContext(authors_used={"hq"})
    mask = build_action_mask(s, ctx)
    hq_idx = sorted(s.connected_ids).index("hq")
    assert mask["author_idx"][hq_idx] == 0
    p1_idx = sorted(s.connected_ids).index("p1")
    assert mask["author_idx"][p1_idx] == 1


def test_upgrade_valid_when_points_available():
    s = _mk()
    s.upgrades.points = 1
    ctx = TurnContext()
    mask = build_action_mask(s, ctx)
    assert mask["kind"][int(ActionKind.SET_UPGRADE)] == 1


def test_upgrade_masked_when_no_points():
    s = _mk()
    s.upgrades.points = 0
    ctx = TurnContext()
    mask = build_action_mask(s, ctx)
    assert mask["kind"][int(ActionKind.SET_UPGRADE)] == 0
