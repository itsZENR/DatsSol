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


_round_cache: dict = {}


def _load_round_cache(dump_dir: Path, round_id: int) -> dict:
    """Load and cache ALL per-round data once, grouped by turn_id."""
    key = (str(dump_dir.resolve()), round_id)
    if key in _round_cache:
        return _round_cache[key]

    turns = pd.read_csv(dump_dir / "turns.csv")
    turns = turns[turns.round_id == round_id]
    turn_ids = set(turns["id"])
    if not turn_ids:
        raise ValueError(f"No turns for round {round_id}")

    def filter_by_turn(path):
        df = pd.read_csv(path)
        return df[df["turn_id"].isin(turn_ids)].copy()

    cache = {
        "turns": turns.set_index("id"),
        "plantations": filter_by_turn(dump_dir / "plantations.csv"),
        "cells": filter_by_turn(dump_dir / "cells.csv"),
        "constructions": filter_by_turn(dump_dir / "constructions.csv"),
        "beavers": filter_by_turn(dump_dir / "beavers.csv"),
        "upgrades": filter_by_turn(dump_dir / "upgrades.csv"),
        "upgrade_tiers": filter_by_turn(dump_dir / "upgrade_tiers.csv"),
    }
    # Mountains are round-scoped, not turn-scoped.
    mnt = pd.read_csv(dump_dir / "mountains.csv")
    cache["mountains"] = mnt[mnt.round_id == round_id].copy()

    # Group each turn-scoped frame by turn_id for O(1) lookup.
    for k in ("plantations", "cells", "constructions", "beavers", "upgrades", "upgrade_tiers"):
        cache[k] = {tid: grp for tid, grp in cache[k].groupby("turn_id")}

    _round_cache[key] = cache
    return cache


def load_state_at(dump_dir: Path, round_id: int, turn_no: int) -> EngineState:
    cache = _load_round_cache(dump_dir, round_id)
    turns = cache["turns"]
    row = turns[turns["turn_no"] == turn_no]
    if row.empty:
        raise ValueError(f"No turn {turn_no} in round {round_id}")
    turn_id = int(row.index[0])
    map_w = int(row.iloc[0]["map_width"])
    map_h = int(row.iloc[0]["map_height"])

    cfg = GameConfig(map_width=map_w, map_height=map_h)
    s = EngineState.new(cfg, seed=0)
    s.turn_no = turn_no

    plants = cache["plantations"].get(turn_id)
    if plants is not None:
        for _, r in plants.iterrows():
            pid = str(r["plantation_id"])
            s.plantations[pid] = Plantation(
                id=pid, position=(int(r["x"]), int(r["y"])),
                is_main=bool(r["is_main"]), hp=int(r["hp"]),
                immunity_until_turn=int(r["immunity_until"]),
                built_on_turn=0,
            )
            if bool(r["is_main"]):
                s.hq_id = pid

    cells = cache["cells"].get(turn_id)
    if cells is not None:
        for _, r in cells.iterrows():
            pos = (int(r["x"]), int(r["y"]))
            progress = int(r["progress"])
            s.cells[pos] = CellState(
                progress=progress,
                completed_on_turn=(turn_no if progress >= 100 else None),
            )

    cons = cache["constructions"].get(turn_id)
    if cons is not None:
        for _, r in cons.iterrows():
            pos = (int(r["x"]), int(r["y"]))
            s.constructions[pos] = Construction(
                position=pos, progress=int(r["progress"]),
                last_progress_turn=turn_no,
                contributors_last_turn={},
            )

    beavers = cache["beavers"].get(turn_id)
    if beavers is not None:
        for _, r in beavers.iterrows():
            bid = str(r["beaver_id"])
            s.beavers[bid] = Beaver(
                id=bid, position=(int(r["x"]), int(r["y"])),
                hp=int(r["hp"]), last_hit_by={}, total_damage_this_turn={},
            )

    mnt = cache["mountains"]
    s.mountains = frozenset(
        (int(r["x"]), int(r["y"])) for _, r in mnt.iterrows()
    )

    up = cache["upgrades"].get(turn_id)
    if up is not None and not up.empty:
        r = up.iloc[0]
        s.upgrades.points = int(r["points"])
        s.upgrades.turns_until_points = int(r["turns_until"])

    tiers = cache["upgrade_tiers"].get(turn_id)
    if tiers is not None:
        for _, r in tiers.iterrows():
            try:
                name = UpgradeName(str(r["name"]))
            except ValueError:
                continue
            s.upgrades.tiers[name] = UpgradeTier(
                current=int(r["current"]),
                max=int(r["max_level"]) if r["max_level"] != -1 else 99,
            )

    # Recompute connectivity: dump does not persist connected_ids, so we must
    # reconstruct it from plantation positions. Without this, all commands would
    # fail validation and never execute, causing divergence in construction completions.
    from simulator.engine.spatial import bfs_connected
    own_positions = {
        pid: p.position
        for pid, p in s.plantations.items()
        if pid not in s.enemy_ids
    }
    s.connected_ids = bfs_connected(s.hq_id, own_positions)

    return s
