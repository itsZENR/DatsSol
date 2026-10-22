import numpy as np
from simulator.engine.state import EngineState

MAX_PLANT = 40
MAX_ENEMY = 40
MAX_BEAVER = 30
MAX_STORM = 2
F_SELF = 13
F_ENEMY = 4
F_BEAVER = 4
F_STORM = 7
F_GLOBAL = 13
GRID_CHANNELS = 10
GRID_SIZE = 31  # ±15 around HQ


def _visible_positions(state: EngineState) -> set:
    """Chebyshev ≤ vr cells around every own plantation — matches server fog."""
    vr = state.effective_vr()
    out = set()
    for pid, p in state.plantations.items():
        if pid in state.enemy_ids:
            continue
        x, y = p.position
        for dx in range(-vr, vr + 1):
            for dy in range(-vr, vr + 1):
                out.add((x + dx, y + dy))
    return out


def encode_observation(
    state: EngineState,
    used_authors: set | None = None,
    vision_aware: bool = False,
) -> dict:
    """Encode state into obs dict.

    `used_authors` — ids of plantations already used as author in the current turn
    (from FlattenActionEnv's turn context). Policy can see which are free vs used.

    `vision_aware` — if True, hide entities/terrain outside Chebyshev vr of any
    own plantation. Matches what /api/arena gives to the live bot.
    """
    if used_authors is None:
        used_authors = set()
    visible = _visible_positions(state) if vision_aware else None
    hq = state.get_hq()
    hq_pos = hq.position if hq else (0, 0)
    mhp = state.effective_mhp()

    self_arr = np.zeros((MAX_PLANT, F_SELF), dtype=np.float32)
    self_mask = np.zeros(MAX_PLANT, dtype=bool)

    own_ids = sorted(
        pid for pid in state.plantations.keys() if pid not in state.enemy_ids
    )
    oldest_turn = min(
        (p.built_on_turn for pid, p in state.plantations.items()
         if pid not in state.enemy_ids),
        default=0,
    )

    for i, pid in enumerate(own_ids[:MAX_PLANT]):
        p = state.plantations[pid]
        dx = (p.position[0] - hq_pos[0]) / 50.0
        dy = (p.position[1] - hq_pos[1]) / 50.0
        # server never provides builtOnTurn → zero the age-dependent features
        # when vision_aware so training matches live inputs exactly.
        if vision_aware:
            age_feat = 0.0
            oldest_feat = 0.0
        else:
            age_feat = (state.turn_no - p.built_on_turn) / 100.0
            oldest_feat = float(p.built_on_turn == oldest_turn)
        self_arr[i] = [
            dx, dy,
            p.hp / max(1, mhp),
            float(p.is_main),
            float(pid not in state.connected_ids),
            max(0, p.immunity_until_turn - state.turn_no) / 3.0,
            float(state.is_boosted_cell(p.position)),
            (state.cells[p.position].progress / 100.0
             if p.position in state.cells else 0.0),
            age_feat,
            oldest_feat,
            0.0,  # n_connected_neighbors placeholder
            0.0,  # in_storm placeholder
            0.0 if pid in used_authors else 1.0,  # available_this_turn
        ]
        self_mask[i] = True

    enemy_arr = np.zeros((MAX_ENEMY, F_ENEMY), dtype=np.float32)
    enemy_mask = np.zeros(MAX_ENEMY, dtype=bool)
    enemy_ids = sorted(state.enemy_ids)
    i = 0
    for eid in enemy_ids:
        if i >= MAX_ENEMY:
            break
        e = state.plantations[eid]
        if visible is not None and e.position not in visible:
            continue
        dx = (e.position[0] - hq_pos[0]) / 50.0
        dy = (e.position[1] - hq_pos[1]) / 50.0
        enemy_arr[i] = [dx, dy, e.hp / 50.0, 0.0]
        enemy_mask[i] = True
        i += 1

    beaver_arr = np.zeros((MAX_BEAVER, F_BEAVER), dtype=np.float32)
    beaver_mask = np.zeros(MAX_BEAVER, dtype=bool)
    i = 0
    for bid, b in sorted(state.beavers.items()):
        if i >= MAX_BEAVER:
            break
        if visible is not None and b.position not in visible:
            continue
        dx = (b.position[0] - hq_pos[0]) / 50.0
        dy = (b.position[1] - hq_pos[1]) / 50.0
        beaver_arr[i] = [dx, dy, b.hp / 100.0, 0.0]
        beaver_mask[i] = True
        i += 1

    storm_arr = np.zeros((MAX_STORM, F_STORM), dtype=np.float32)
    storm_mask = np.zeros(MAX_STORM, dtype=bool)
    for i, s in enumerate(state.storms[:MAX_STORM]):
        dx = (s.position[0] - hq_pos[0]) / 50.0
        dy = (s.position[1] - hq_pos[1]) / 50.0
        storm_arr[i] = [dx, dy, 0.0, 0.0, s.radius / 50.0,
                        float(s.forming_left > 0), s.forming_left / 5.0]
        storm_mask[i] = True

    global_arr = np.zeros(F_GLOBAL, dtype=np.float32)
    global_arr[0] = state.turn_no / state.config.round_turns
    global_arr[1] = state.upgrades.points / state.config.max_upgrade_points
    global_arr[2] = state.effective_rs()
    global_arr[3] = state.config.se
    global_arr[4] = state.config.be
    global_arr[5] = mhp / 100.0
    global_arr[6] = state.config.ar
    global_arr[7] = state.effective_sr()
    global_arr[8] = (
        (state.earthquake_turn - state.turn_no) / 20.0
        if state.earthquake_turn is not None else 0.0
    )
    global_arr[9] = float(bool(state.storms))
    global_arr[10] = (hq.hp / max(1, mhp)) if hq else 0.0
    global_arr[11] = len([
        pid for pid in state.plantations if pid not in state.enemy_ids
    ]) / MAX_PLANT
    # /api/arena doesn't return score — hide it from vision_aware training too.
    global_arr[12] = 0.0 if vision_aware else (state.score / 1e5)

    grid = np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    half = GRID_SIZE // 2
    w, h = state.config.map_width, state.config.map_height
    own_positions = {p.position for pid, p in state.plantations.items() if pid not in state.enemy_ids}
    enemy_positions = {state.plantations[eid].position for eid in state.enemy_ids}
    beaver_positions = {b.position for b in state.beavers.values()}
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            x = hq_pos[0] + dx
            y = hq_pos[1] + dy
            gx = dx + half
            gy = dy + half
            if not (0 <= x < w and 0 <= y < h):
                continue
            seen = visible is None or (x, y) in visible
            if seen and (x, y) in state.mountains:
                grid[0, gx, gy] = 1.0
            if (x, y) in own_positions:
                grid[1, gx, gy] = 1.0
            if seen and (x, y) in state.constructions:
                grid[2, gx, gy] = state.constructions[(x, y)].progress / 50.0
            if seen and (x, y) in enemy_positions:
                grid[3, gx, gy] = 1.0
            if seen and (x, y) in beaver_positions:
                grid[4, gx, gy] = 1.0
            if seen and (x, y) in state.cells:
                grid[5, gx, gy] = state.cells[(x, y)].progress / 100.0
            if state.is_boosted_cell((x, y)):
                grid[6, gx, gy] = 1.0

    return {
        "self": self_arr,
        "self_mask": self_mask,
        "enemy": enemy_arr,
        "enemy_mask": enemy_mask,
        "beaver": beaver_arr,
        "beaver_mask": beaver_mask,
        "storm": storm_arr,
        "storm_mask": storm_mask,
        "global": global_arr,
        "grid": grid,
    }
