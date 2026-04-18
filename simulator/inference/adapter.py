from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import (
    Plantation, Construction, Beaver, Storm, CellState, UpgradeTier,
)
from simulator.engine.commands import CommandSet
from simulator.engine.types import UpgradeName


def api_to_state(resp: dict) -> EngineState:
    size = resp.get("size", [278, 278])
    cfg = GameConfig(map_width=int(size[0]), map_height=int(size[1]))
    s = EngineState.new(cfg, seed=0)
    s.turn_no = int(resp.get("turnNo", 0))

    for p in resp.get("plantations", []):
        pid = str(p["id"])
        s.plantations[pid] = Plantation(
            id=pid, position=(int(p["position"][0]), int(p["position"][1])),
            is_main=bool(p.get("isMain", False)),
            hp=int(p["hp"]),
            immunity_until_turn=int(p.get("immunityUntilTurn", 0)),
            built_on_turn=0,
        )
        if p.get("isMain"):
            s.hq_id = pid

    for e in resp.get("enemy", []):
        eid = str(e["id"])
        s.enemy_ids.add(eid)
        s.plantations[eid] = Plantation(
            id=eid, position=(int(e["position"][0]), int(e["position"][1])),
            is_main=False, hp=int(e["hp"]),
            immunity_until_turn=0, built_on_turn=0,
        )

    s.mountains = frozenset(
        (int(m[0]), int(m[1])) for m in resp.get("mountains", [])
    )

    for c in resp.get("cells", []):
        pos = (int(c["position"][0]), int(c["position"][1]))
        progress = int(c.get("terraformationProgress", 0))
        completed_on = None
        if progress >= 100:
            turns_until_deg = int(c.get("turnsUntilDegradation", cfg.cell_degradation_delay))
            completed_on = s.turn_no - (cfg.cell_degradation_delay - turns_until_deg)
        s.cells[pos] = CellState(progress=progress, completed_on_turn=completed_on)

    for c in resp.get("construction", []):
        pos = (int(c["position"][0]), int(c["position"][1]))
        s.constructions[pos] = Construction(
            position=pos, progress=int(c.get("progress", 0)),
            last_progress_turn=s.turn_no,
            contributors_last_turn={},
        )

    for b in resp.get("beavers", []):
        bid = str(b["id"])
        s.beavers[bid] = Beaver(
            id=bid, position=(int(b["position"][0]), int(b["position"][1])),
            hp=int(b["hp"]), last_hit_by={}, total_damage_this_turn={},
        )

    up = resp.get("plantationUpgrades", {})
    s.upgrades.points = int(up.get("points", 0))
    s.upgrades.turns_until_points = int(up.get("turnsUntilPoints", 30))
    for t in up.get("tiers", []):
        try:
            name = UpgradeName(str(t["name"]))
        except ValueError:
            continue
        s.upgrades.tiers[name] = UpgradeTier(
            current=int(t.get("current", 0)),
            max=int(t.get("max", 99)),
        )

    for f in resp.get("meteoForecasts", []):
        kind = str(f.get("kind", ""))
        turns_until = int(f.get("turnsUntil", 0))
        if kind == "earthquake":
            s.earthquake_turn = s.turn_no + turns_until
        elif kind == "sandstorm" and f.get("position"):
            s.storms.append(Storm(
                id=str(f.get("id", "")),
                position=(int(f["position"][0]), int(f["position"][1])),
                radius=int(f.get("radius", 30)),
                speed=0,
                direction=(1, 1),
                forming_left=turns_until if f.get("forming") else 0,
            ))

    return s


def state_to_payload(cs: CommandSet) -> dict:
    return cs.to_payload()
