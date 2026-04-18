import requests
import logging
from models import (
    GameState, Plantation, Enemy, Cell, Construction, Beaver,
    UpgradeTier, PlantationUpgrades, MeteoForecast
)
from config import TOKEN, BASE_URL

log = logging.getLogger(__name__)


def _headers() -> dict:
    return {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}


def get_arena() -> GameState:
    r = requests.get(f"{BASE_URL}/api/arena", headers=_headers(), timeout=5)
    r.raise_for_status()
    d = r.json()

    plantations = [
        Plantation(
            id=p["id"],
            position=tuple(p["position"]),
            is_main=p.get("isMain", False),
            is_isolated=p.get("isIsolated", False),
            immunity_until_turn=p.get("immunityUntilTurn", 0),
            hp=p["hp"],
        )
        for p in d.get("plantations", [])
    ]

    enemies = [
        Enemy(id=e["id"], position=tuple(e["position"]), hp=e["hp"])
        for e in d.get("enemy", [])
    ]

    mountains = [tuple(m) for m in d.get("mountains", [])]

    cells = [
        Cell(
            position=tuple(c["position"]),
            terraformation_progress=c.get("terraformationProgress", 0),
            turns_until_degradation=c.get("turnsUntilDegradation", 0),
        )
        for c in d.get("cells", [])
    ]

    constructions = [
        Construction(position=tuple(c["position"]), progress=c.get("progress", 0))
        for c in d.get("construction", [])
    ]

    beavers = [
        Beaver(id=b["id"], position=tuple(b["position"]), hp=b["hp"])
        for b in d.get("beavers", [])
    ]

    tiers = [
        UpgradeTier(name=t["name"], current=t.get("current", 0), max=t.get("max", 0))
        for t in d.get("plantationUpgrades", {}).get("tiers", [])
    ]
    upgrades_raw = d.get("plantationUpgrades", {})
    upgrades = PlantationUpgrades(
        points=upgrades_raw.get("points", 0),
        interval_turns=upgrades_raw.get("intervalTurns", 30),
        turns_until_points=upgrades_raw.get("turnsUntilPoints", 30),
        max_points=upgrades_raw.get("maxPoints", 15),
        tiers=tiers,
    )

    forecasts = []
    for f in d.get("meteoForecasts", []):
        forecasts.append(MeteoForecast(
            kind=f.get("kind", ""),
            turns_until=f.get("turnsUntil", 0),
            id=f.get("id"),
            forming=f.get("forming"),
            position=tuple(f["position"]) if f.get("position") else None,
            next_position=tuple(f["nextPosition"]) if f.get("nextPosition") else None,
            radius=f.get("radius"),
        ))

    return GameState(
        turn_no=d.get("turnNo", 0),
        next_turn_in=d.get("nextTurnIn", 1.0),
        size=tuple(d.get("size", [500, 500])),
        action_range=d.get("actionRange", 2),
        plantations=plantations,
        enemies=enemies,
        mountains=mountains,
        cells=cells,
        constructions=constructions,
        beavers=beavers,
        upgrades=upgrades,
        meteo_forecasts=forecasts,
    )


def post_command(payload: dict) -> dict:
    r = requests.post(
        f"{BASE_URL}/api/command",
        json=payload,
        headers=_headers(),
        timeout=5,
    )
    r.raise_for_status()
    resp = r.json()
    if resp.get("errors"):
        log.warning("Server errors: %s", resp["errors"])
    return resp


def get_logs() -> list[dict]:
    r = requests.get(f"{BASE_URL}/api/logs", headers=_headers(), timeout=5)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return data
    log.warning("Logs error: %s", data)
    return []
