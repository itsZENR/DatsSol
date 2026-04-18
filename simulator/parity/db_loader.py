from pathlib import Path
import pandas as pd
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import (
    Plantation, Construction, Beaver, CellState, UpgradeTier,
)
from simulator.engine.types import UpgradeName


def load_round_ids(dump_dir: Path) -> list[int]:
    rounds = pd.read_csv(dump_dir / "rounds.csv")
    return rounds["id"].tolist()


def load_state_at(dump_dir: Path, round_id: int, turn_no: int) -> EngineState:
    turns = pd.read_csv(dump_dir / "turns.csv")
    row = turns[(turns.round_id == round_id) & (turns.turn_no == turn_no)]
    if row.empty:
        raise ValueError(f"No turn {turn_no} in round {round_id}")
    turn_id = int(row.iloc[0]["id"])
    map_w = int(row.iloc[0]["map_width"])
    map_h = int(row.iloc[0]["map_height"])

    cfg = GameConfig(map_width=map_w, map_height=map_h)
    s = EngineState.new(cfg, seed=0)
    s.turn_no = turn_no

    plants = pd.read_csv(dump_dir / "plantations.csv")
    for _, r in plants[plants.turn_id == turn_id].iterrows():
        pid = str(r["plantation_id"])
        s.plantations[pid] = Plantation(
            id=pid, position=(int(r["x"]), int(r["y"])),
            is_main=bool(r["is_main"]), hp=int(r["hp"]),
            immunity_until_turn=int(r["immunity_until"]),
            built_on_turn=0,
        )
        if bool(r["is_main"]):
            s.hq_id = pid

    cells = pd.read_csv(dump_dir / "cells.csv")
    for _, r in cells[cells.turn_id == turn_id].iterrows():
        pos = (int(r["x"]), int(r["y"]))
        progress = int(r["progress"])
        s.cells[pos] = CellState(
            progress=progress,
            completed_on_turn=(turn_no if progress >= 100 else None),
        )

    cons = pd.read_csv(dump_dir / "constructions.csv")
    for _, r in cons[cons.turn_id == turn_id].iterrows():
        pos = (int(r["x"]), int(r["y"]))
        s.constructions[pos] = Construction(
            position=pos, progress=int(r["progress"]),
            last_progress_turn=turn_no,
            contributors_last_turn={},
        )

    beavers = pd.read_csv(dump_dir / "beavers.csv")
    for _, r in beavers[beavers.turn_id == turn_id].iterrows():
        bid = str(r["beaver_id"])
        s.beavers[bid] = Beaver(
            id=bid, position=(int(r["x"]), int(r["y"])),
            hp=int(r["hp"]), last_hit_by={}, total_damage_this_turn={},
        )

    mnt = pd.read_csv(dump_dir / "mountains.csv")
    round_id_val = int(row.iloc[0]["round_id"])
    mnt_round = mnt[mnt.round_id == round_id_val]
    s.mountains = frozenset(
        (int(r["x"]), int(r["y"])) for _, r in mnt_round.iterrows()
    )

    up = pd.read_csv(dump_dir / "upgrades.csv")
    up_row = up[up.turn_id == turn_id]
    if not up_row.empty:
        s.upgrades.points = int(up_row.iloc[0]["points"])
        s.upgrades.turns_until_points = int(up_row.iloc[0]["turns_until"])

    tiers = pd.read_csv(dump_dir / "upgrade_tiers.csv")
    for _, r in tiers[tiers.turn_id == turn_id].iterrows():
        try:
            name = UpgradeName(str(r["name"]))
        except ValueError:
            continue
        s.upgrades.tiers[name] = UpgradeTier(
            current=int(r["current"]),
            max=int(r["max_level"]) if r["max_level"] != -1 else 99,
        )

    return s
