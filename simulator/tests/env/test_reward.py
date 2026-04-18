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


def test_penalty_for_lost_plantation():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.cells[(5, 6)] = CellState(progress=40, completed_on_turn=None)
    events = [
        PlantationDestroyed(plantation_id="p1", position=(5, 6), was_main=False),
    ]
    r = compute_reward(0, 0, events, state_after=s)
    # Lost plantation with 40% progress → potential = 1000 * 60/100 = 600
    # reward = 0 - 600/1000 = -0.6
    assert abs(r - (-0.6)) < 1e-6


def test_hq_loss_penalty():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    events = [HQLost(prev_hq_id="hq", penalty=50)]
    r = compute_reward(1000, 950, events, state_after=s)
    # score_delta = -50 / 1000 = -0.05
    assert abs(r - (-0.05)) < 1e-6
