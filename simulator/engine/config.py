from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """
    All constants of the DatsSol game. Defaults match the game documentation
    and the real-round dump. Change via replace() for experiments.
    """
    # Plantation
    mhp: int = 50
    ts: int = 5
    cs: int = 5
    rs: int = 5
    se: int = 5
    be: int = 5
    ds: int = 10
    ar: int = 2
    sr: int = 3
    vr: int = 3

    # Limits
    base_plantation_limit: int = 30
    max_plantation_limit: int = 40

    # Upgrades
    upgrade_interval: int = 30
    max_upgrade_points: int = 15
    immunity_turns: int = 3

    # Beavers
    beaver_max_hp: int = 100
    beaver_regen: int = 5
    beaver_damage: int = 15
    beaver_attack_radius: int = 2

    # Cataclysms
    storm_damage: int = 2
    storm_forming_turns: int = 5
    storm_prob: float = 0.10
    storm_speed_min: int = 5
    storm_speed_max: int = 15
    storm_min_hp: int = 1
    earthquake_damage: int = 10
    earthquake_prob: float = 0.05

    # Cells
    cell_degradation_delay: int = 20   # 2026-04-18 balance: 40 -> 30 -> 20
    cell_degradation_per_turn: int = 10
    normal_cell_reward: int = 1000
    boosted_cell_reward: int = 1500
    boosted_mod: int = 7

    # Beavers - kill reward multiplier (2026-04-18 balance: 10x -> 20x)
    beaver_kill_multiplier: int = 20

    # Round
    round_turns: int = 600

    # Map (2026-04-18 balance: 378x378 -> 298x298 -> 278x278). Override via replace for curriculum.
    map_width: int = 278
    map_height: int = 278

    # Random map content at reset (used by DatssolEnv; 0 = disabled).
    initial_beavers: int = 0
    initial_mountain_density: float = 0.0   # fraction of cells as mountains (0.0 to ~0.1)
