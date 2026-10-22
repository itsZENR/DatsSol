"""In-memory store for live arena polling.

Polls /api/arena in a background thread, converts each response to a
TurnData dict matching the Go dashboard shape, and exposes the same
query surface as CsvStore.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional


LIVE_ROUND_ID = 999999
ARENA_SNAPSHOT_PATH = Path("tmp/arena_snapshot.json")
ACTIONS_SNAPSHOT_PATH = Path("tmp/actions_snapshot.json")


def _arena_to_turn_data(resp: dict) -> dict:
    size = resp.get("size", [278, 278])
    turn_no = int(resp.get("turnNo", 0))

    plantations = []
    for p in resp.get("plantations", []):
        pos = p["position"]
        plantations.append({
            "id": str(p["id"]),
            "x": int(pos[0]), "y": int(pos[1]),
            "hp": int(p.get("hp", 0)),
            "isMain": bool(p.get("isMain", False)),
            "isIsolated": bool(p.get("isIsolated", False)),
        })

    enemies = []
    for e in resp.get("enemy", []):
        pos = e["position"]
        enemies.append({
            "id": str(e["id"]),
            "x": int(pos[0]), "y": int(pos[1]),
            "hp": int(e.get("hp", 0)),
        })

    beavers = []
    for b in resp.get("beavers", []):
        pos = b["position"]
        beavers.append({
            "id": str(b["id"]),
            "x": int(pos[0]), "y": int(pos[1]),
            "hp": int(b.get("hp", 0)),
        })

    cells = []
    for c in resp.get("cells", []):
        pos = c["position"]
        cells.append({
            "x": int(pos[0]), "y": int(pos[1]),
            "progress": int(c.get("terraformationProgress", 0)),
            "turnsUntilDeg": int(c.get("turnsUntilDegradation", 0)),
        })

    constructions = []
    for c in resp.get("construction", []):
        pos = c["position"]
        constructions.append({
            "x": int(pos[0]), "y": int(pos[1]),
            "progress": int(c.get("progress", 0)),
        })

    meteo = []
    for f in resp.get("meteoForecasts", []):
        m = {"kind": str(f.get("kind", ""))}
        if "turnsUntil" in f:
            m["turnsUntil"] = int(f["turnsUntil"])
        else:
            m["turnsUntil"] = None
        if f.get("id"):
            m["stormId"] = str(f["id"])
        if "forming" in f:
            m["forming"] = bool(f["forming"])
        if f.get("position"):
            m["posX"] = int(f["position"][0])
            m["posY"] = int(f["position"][1])
        if f.get("nextPosition"):
            m["nextPosX"] = int(f["nextPosition"][0])
            m["nextPosY"] = int(f["nextPosition"][1])
        if "radius" in f:
            m["radius"] = int(f["radius"])
        meteo.append(m)

    upgrades = []
    up = resp.get("plantationUpgrades", {})
    for t in up.get("tiers", []):
        upgrades.append({
            "name": str(t.get("name", "")),
            "current": int(t.get("current", 0)),
            "max": int(t.get("max", 0)),
        })

    mountains = [
        {"x": int(m[0]), "y": int(m[1])}
        for m in resp.get("mountains", [])
    ]

    return {
        "turnNo": turn_no,
        "mapWidth": int(size[0]),
        "mapHeight": int(size[1]),
        "plantations": plantations,
        "enemies": enemies,
        "beavers": beavers,
        "constructions": constructions,
        "cells": cells,
        "meteo": meteo,
        "upgrades": upgrades,
        "actions": [],
        "mountains": mountains,
    }


class LiveStore:
    """Polls arena and stores snapshots keyed by turn_no."""

    def __init__(self, base_url: str, token: str, interval: float = 1.0):
        self.base_url = base_url
        self.token = token
        self.interval = interval
        self._lock = threading.Lock()
        self._snapshots: dict[int, dict] = {}
        self._latest_turn: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _try_read_snapshot(self) -> Optional[dict]:
        try:
            with ARENA_SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def _try_read_actions(self) -> Optional[dict]:
        try:
            with ACTIONS_SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def _loop(self):
        last_arena_mtime: Optional[float] = None
        last_actions_mtime: Optional[float] = None
        while not self._stop.is_set():
            try:
                try:
                    arena_mtime = ARENA_SNAPSHOT_PATH.stat().st_mtime
                except FileNotFoundError:
                    self._stop.wait(self.interval)
                    continue
                actions_mtime = None
                try:
                    actions_mtime = ACTIONS_SNAPSHOT_PATH.stat().st_mtime
                except FileNotFoundError:
                    pass

                arena_changed = arena_mtime != last_arena_mtime
                actions_changed = actions_mtime != last_actions_mtime

                if not arena_changed and not actions_changed:
                    self._stop.wait(self.interval)
                    continue
                last_arena_mtime = arena_mtime
                last_actions_mtime = actions_mtime

                # Always pull actions — may apply to a past turn already stored.
                actions = self._try_read_actions()
                if actions:
                    with self._lock:
                        aturn = actions.get("turnNo")
                        if aturn in self._snapshots:
                            self._snapshots[aturn]["actions"] = actions.get("actions", [])

                if not arena_changed:
                    self._stop.wait(self.interval)
                    continue

                data = self._try_read_snapshot()
                if data is not None:
                    td = _arena_to_turn_data(data)
                    if actions and actions.get("turnNo") == td["turnNo"]:
                        td["actions"] = actions.get("actions", [])
                    with self._lock:
                        # Detect round restart: turnNo dropped below latest.
                        if (self._latest_turn is not None
                                and td["turnNo"] + 10 < self._latest_turn):
                            self._snapshots.clear()
                            self._latest_turn = None
                        self._snapshots[td["turnNo"]] = td
                        if self._latest_turn is None or td["turnNo"] >= self._latest_turn:
                            self._latest_turn = td["turnNo"]
            except Exception:
                pass
            self._stop.wait(self.interval)

    def has_data(self) -> bool:
        with self._lock:
            return self._latest_turn is not None

    def get_round_summary(self) -> Optional[dict]:
        with self._lock:
            if self._latest_turn is None:
                return None
            return {
                "id": LIVE_ROUND_ID,
                "roundNum": 99999,
                "totalTurns": len(self._snapshots),
            }

    def get_turn_list(self) -> list[dict]:
        with self._lock:
            return [
                {"turnNo": t, "turnId": t}
                for t in sorted(self._snapshots.keys())
            ]

    def get_turn_data(self, turn_no: int) -> Optional[dict]:
        with self._lock:
            return self._snapshots.get(turn_no)

    def get_mountains(self) -> list[dict]:
        with self._lock:
            if self._latest_turn is None:
                return []
            td = self._snapshots[self._latest_turn]
            return list(td.get("mountains", []))

    def get_score_history(self) -> list[dict]:
        with self._lock:
            items = sorted(self._snapshots.items())
        out = []
        for t, td in items:
            out.append({
                "turnNo": t,
                "plantations": len(td["plantations"]),
                "scoreDelta": len(td["plantations"]) * 50,
                "enemies": len(td["enemies"]),
                "beavers": len(td["beavers"]),
            })
        return out

    def get_round_stats(self) -> dict:
        with self._lock:
            items = list(self._snapshots.values())
        if not items:
            return {
                "totalTurns": 0, "totalScore": 0, "maxPlantations": 0,
                "totalEnemiesSeen": 0, "totalBeaversSeen": 0,
                "stormTurns": 0, "earthquakes": 0,
            }
        return {
            "totalTurns": len(items),
            "totalScore": sum(len(t["plantations"]) * 50 for t in items),
            "maxPlantations": max(len(t["plantations"]) for t in items),
            "totalEnemiesSeen": max(len(t["enemies"]) for t in items),
            "totalBeaversSeen": max(len(t["beavers"]) for t in items),
            "stormTurns": sum(
                1 for t in items
                if any(m["kind"] == "sandstorm" for m in t["meteo"])
            ),
            "earthquakes": sum(
                1 for t in items
                if any(m["kind"] == "earthquake" for m in t["meteo"])
            ),
        }

    def get_accumulated(self, turn_no: int) -> dict:
        with self._lock:
            items = [(t, td) for t, td in self._snapshots.items() if t <= turn_no]
        items.sort()
        cells: dict[tuple[int, int], dict] = {}
        enemies: dict[str, dict] = {}
        beavers: dict[str, dict] = {}
        for _, td in items:
            for c in td["cells"]:
                cells[(c["x"], c["y"])] = c
            for e in td["enemies"]:
                enemies[e["id"]] = e
            for b in td["beavers"]:
                beavers[b["id"]] = b
        return {
            "cells": list(cells.values()),
            "enemies": list(enemies.values()),
            "beavers": list(beavers.values()),
        }
