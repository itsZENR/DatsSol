from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import CellState
from simulator.engine.events import Scored, PlantationDestroyed, HQLost
from simulator.env.reward import compute_reward, SCORE_SCALE


def test_score_delta_to_reward():
    prev_score = 100
    cur_score = 250
    events = [Scored(amount=150, source="terraform")]
    r = compute_reward(prev_score, cur_score, events, state_after=None)
    assert abs(r - 0.15) < 1e-6


def test_plantation_loss_no_longer_penalized():
    # Per shaping redesign: lost plantations no longer add penalty to reward.
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.cells[(5, 6)] = CellState(progress=40, completed_on_turn=None)
    events = [
        PlantationDestroyed(plantation_id="p1", position=(5, 6), was_main=False),
    ]
    r = compute_reward(0, 0, events, state_after=s)
    assert abs(r - 0.0) < 1e-6


def test_hq_loss_reward_is_pure_score_delta():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    events = [HQLost(prev_hq_id="hq", penalty=50)]
    r = compute_reward(1000, 950, events, state_after=s)
    # Only the −50 score delta applies.
    assert abs(r - (-0.05)) < 1e-6


def test_construction_completed_no_longer_shaped():
    # Pure-score reward: construction completion itself gives no bonus.
    from simulator.engine.events import ConstructionCompleted
    r = compute_reward(0, 0, [ConstructionCompleted(position=(1, 2), new_plantation_id="p1")], None)
    assert abs(r - 0.0) < 1e-6
