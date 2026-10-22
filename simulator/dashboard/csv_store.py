"""CSV-backed read-only store. Loads dump/*.csv once, answers dashboard queries.

Schema reference: dump/DATABASE.md. Tables keyed by turn_id (int FK -> turns.id)
and round_id (int FK -> rounds.id).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


class CsvStore:
    def __init__(self, dump_dir: str | Path):
        d = Path(dump_dir)
        self.rounds = _read(d / "rounds.csv")
        self.turns = _read(d / "turns.csv")
        self.plantations = _read(d / "plantations.csv")
        self.enemies = _read(d / "enemies.csv")
        self.beavers = _read(d / "beavers.csv")
        self.cells = _read(d / "cells.csv")
        self.constructions = _read(d / "constructions.csv")
        self.meteo = _read(d / "meteo_events.csv")
        self.mountains = _read(d / "mountains.csv")
        self.upgrades = _read(d / "upgrades.csv")
        self.upgrade_tiers = _read(d / "upgrade_tiers.csv")
        self.score_log = _read(d / "score_log.csv")
        self.actions = _read(d / "plantation_actions.csv")
        self.strategy_config = _read(d / "strategy_config.csv")

        # Precompute (round_id, turn_no) -> turn_id for quick lookup.
        if not self.turns.empty:
            self._turn_id = {
                (int(r.round_id), int(r.turn_no)): int(r.id)
                for r in self.turns.itertuples()
            }
        else:
            self._turn_id = {}

    def turn_id(self, round_id: int, turn_no: int) -> Optional[int]:
        return self._turn_id.get((round_id, turn_no))

    def get_rounds(self) -> list[dict]:
        if self.rounds.empty:
            return []
        counts = (
            self.turns.groupby("round_id").size().to_dict()
            if not self.turns.empty else {}
        )
        out = []
        for r in self.rounds.itertuples():
            out.append({
                "id": int(r.id),
                "roundNum": int(r.round_num),
                "totalTurns": int(counts.get(int(r.id), 0)),
            })
        out.sort(key=lambda x: x["roundNum"])
        return out

    def get_score_history(self, round_id: int) -> list[dict]:
        if self.score_log.empty:
            return []
        df = self.score_log[self.score_log.round_id == round_id].sort_values("turn_no")
        return [
            {
                "turnNo": int(r.turn_no),
                "plantations": int(r.my_plantations),
                "scoreDelta": int(r.score_delta),
                "enemies": int(r.enemies_visible),
                "beavers": int(r.beavers_visible),
            }
            for r in df.itertuples()
        ]

    def get_mountains(self, round_id: int) -> list[dict]:
        if self.mountains.empty:
            return []
        df = self.mountains[self.mountains.round_id == round_id]
        return [{"x": int(r.x), "y": int(r.y)} for r in df.itertuples()]

    def get_turn_list(self, round_id: int) -> list[dict]:
        if self.turns.empty:
            return []
        df = self.turns[self.turns.round_id == round_id].sort_values("turn_no")
        return [{"turnNo": int(r.turn_no), "turnId": int(r.id)} for r in df.itertuples()]

    def get_turn_data(self, round_id: int, turn_no: int) -> Optional[dict]:
        tid = self.turn_id(round_id, turn_no)
        if tid is None:
            return None
        turn_row = self.turns[self.turns.id == tid].iloc[0]
        td = {
            "turnNo": int(turn_no),
            "mapWidth": int(turn_row.map_width),
            "mapHeight": int(turn_row.map_height),
            "plantations": [],
            "enemies": [],
            "beavers": [],
            "constructions": [],
            "cells": [],
            "meteo": [],
            "upgrades": [],
            "actions": [],
            "mountains": [],
        }

        if not self.plantations.empty:
            df = self.plantations[self.plantations.turn_id == tid]
            td["plantations"] = [
                {
                    "id": str(r.plantation_id),
                    "x": int(r.x), "y": int(r.y), "hp": int(r.hp),
                    "isMain": bool(_truthy(r.is_main)),
                    "isIsolated": bool(_truthy(r.is_isolated)),
                }
                for r in df.itertuples()
            ]

        if not self.enemies.empty:
            df = self.enemies[self.enemies.turn_id == tid]
            td["enemies"] = [
                {"id": str(r.enemy_id), "x": int(r.x), "y": int(r.y), "hp": int(r.hp)}
                for r in df.itertuples()
            ]

        if not self.beavers.empty:
            df = self.beavers[self.beavers.turn_id == tid]
            td["beavers"] = [
                {"id": str(r.beaver_id), "x": int(r.x), "y": int(r.y), "hp": int(r.hp)}
                for r in df.itertuples()
            ]

        if not self.constructions.empty:
            df = self.constructions[self.constructions.turn_id == tid]
            td["constructions"] = [
                {"x": int(r.x), "y": int(r.y), "progress": int(r.progress)}
                for r in df.itertuples()
            ]

        if not self.cells.empty:
            df = self.cells[self.cells.turn_id == tid]
            td["cells"] = [
                {
                    "x": int(r.x), "y": int(r.y),
                    "progress": int(r.progress),
                    "turnsUntilDeg": int(r.turns_until_deg),
                }
                for r in df.itertuples()
            ]

        if not self.meteo.empty:
            df = self.meteo[self.meteo.turn_id == tid]
            td["meteo"] = [_meteo_row(r) for r in df.itertuples()]

        if not self.upgrade_tiers.empty:
            df = self.upgrade_tiers[self.upgrade_tiers.turn_id == tid].sort_values("name")
            td["upgrades"] = [
                {"name": str(r.name), "current": int(r.current), "max": int(r.max_level)}
                for r in df.itertuples()
            ]

        if not self.actions.empty:
            df = self.actions[
                (self.actions.round_id == round_id) & (self.actions.turn_no == turn_no)
            ]
            td["actions"] = [_action_row(r) for r in df.itertuples()]

        td["mountains"] = self.get_mountains(round_id)
        return td

    def get_round_stats(self, round_id: int) -> dict:
        if self.score_log.empty:
            return _empty_stats()
        df = self.score_log[self.score_log.round_id == round_id]
        if df.empty:
            return _empty_stats()
        return {
            "totalTurns": int(len(df)),
            "totalScore": int(df.score_delta.sum()),
            "maxPlantations": int(df.my_plantations.max()),
            "totalEnemiesSeen": int(df.enemies_visible.max()),
            "totalBeaversSeen": int(df.beavers_visible.max()),
            "stormTurns": int(df.storm_active.apply(_truthy).sum()),
            "earthquakes": int(df.earthquake.apply(_truthy).sum()),
        }

    def get_accumulated(self, round_id: int, turn_no: int) -> dict:
        tids = set()
        if not self.turns.empty:
            mask = (self.turns.round_id == round_id) & (self.turns.turn_no <= turn_no)
            tids = set(int(x) for x in self.turns.loc[mask, "id"].tolist())
        if not tids:
            return {"cells": [], "enemies": [], "beavers": []}

        turn_no_map = {int(r.id): int(r.turn_no) for r in self.turns.itertuples()}

        cells = []
        if not self.cells.empty:
            df = self.cells[self.cells.turn_id.isin(tids)].copy()
            df["turn_no"] = df.turn_id.map(turn_no_map)
            df = df.sort_values("turn_no").drop_duplicates(["x", "y"], keep="last")
            cells = [
                {
                    "x": int(r.x), "y": int(r.y),
                    "progress": int(r.progress),
                    "turnsUntilDeg": int(r.turns_until_deg),
                }
                for r in df.itertuples()
            ]

        enemies = []
        if not self.enemies.empty:
            df = self.enemies[self.enemies.turn_id.isin(tids)].copy()
            df["turn_no"] = df.turn_id.map(turn_no_map)
            df = df.sort_values("turn_no").drop_duplicates("enemy_id", keep="last")
            enemies = [
                {"id": str(r.enemy_id), "x": int(r.x), "y": int(r.y), "hp": int(r.hp)}
                for r in df.itertuples()
            ]

        beavers = []
        if not self.beavers.empty:
            df = self.beavers[self.beavers.turn_id.isin(tids)].copy()
            df["turn_no"] = df.turn_id.map(turn_no_map)
            df = df.sort_values("turn_no").drop_duplicates("beaver_id", keep="last")
            beavers = [
                {"id": str(r.beaver_id), "x": int(r.x), "y": int(r.y), "hp": int(r.hp)}
                for r in df.itertuples()
            ]

        return {"cells": cells, "enemies": enemies, "beavers": beavers}

    def get_strategy_config(self) -> list[dict]:
        if self.strategy_config.empty:
            return []
        max_rid = int(self.strategy_config.round_id.max())
        df = self.strategy_config[self.strategy_config.round_id == max_rid].sort_values("param_name")
        return [
            {"name": str(r.param_name), "value": float(r.param_value)}
            for r in df.itertuples()
        ]


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ("t", "true", "1", "yes", "y")


def _meteo_row(r) -> dict:
    out = {"kind": str(r.kind)}
    out["turnsUntil"] = None if pd.isna(r.turns_until) else int(r.turns_until)
    if not pd.isna(r.storm_id):
        out["stormId"] = str(r.storm_id)
    if not pd.isna(r.forming):
        out["forming"] = _truthy(r.forming)
    if not pd.isna(r.pos_x):
        out["posX"] = int(r.pos_x)
    if not pd.isna(r.pos_y):
        out["posY"] = int(r.pos_y)
    if not pd.isna(r.next_pos_x):
        out["nextPosX"] = int(r.next_pos_x)
    if not pd.isna(r.next_pos_y):
        out["nextPosY"] = int(r.next_pos_y)
    if not pd.isna(r.radius):
        out["radius"] = int(r.radius)
    return out


def _action_row(r) -> dict:
    return {
        "authorX": int(r.author_x),
        "authorY": int(r.author_y),
        "targetX": int(r.target_x),
        "targetY": int(r.target_y),
        "action": str(r.action),
        "reason": "",
    }


def _empty_stats() -> dict:
    return {
        "totalTurns": 0, "totalScore": 0, "maxPlantations": 0,
        "totalEnemiesSeen": 0, "totalBeaversSeen": 0,
        "stormTurns": 0, "earthquakes": 0,
    }
