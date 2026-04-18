from pathlib import Path
import yaml
from collections import deque
from simulator.env.datssol_env import DatssolEnv

CONFIG_DIR = Path(__file__).parent / "configs"


def load_stage_config(stage: int) -> dict:
    p = CONFIG_DIR / f"stage{stage}.yaml"
    with p.open() as f:
        return yaml.safe_load(f)


def make_env_for_stage(stage: int, seed: int) -> DatssolEnv:
    cfg = load_stage_config(stage)
    return DatssolEnv(
        seed=seed,
        map_size=tuple(cfg["map_size"]),
        round_turns=cfg["round_turns"],
    )


class CurriculumScheduler:
    def __init__(self, start_stage: int = 0, history: int = 100):
        self.stage = start_stage
        self.history = deque(maxlen=history)

    def record_episode(self, episode_reward: float) -> None:
        self.history.append(episode_reward)

    def maybe_promote(self) -> bool:
        if len(self.history) < self.history.maxlen:
            return False
        cfg = load_stage_config(self.stage)
        threshold = cfg.get("promote_reward_threshold")
        if threshold is None:
            return False
        mean_r = sum(self.history) / len(self.history)
        if mean_r >= threshold:
            self.stage += 1
            self.history.clear()
            return True
        return False
