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


def encode_observation(state: EngineState, used_authors: set | None = None) -> dict:
    """Encode state into obs dict.

    `used_authors` — ids of plantations already used as author in the current turn
    (from FlattenActionEnv's turn context). Policy can see which are free vs used.
    """
    if used_authors is None:
        used_authors = set()
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
        self_arr[i] = [
            dx, dy,
            p.hp / max(1, mhp),
            float(p.is_main),
            float(pid not in state.connected_ids),
            max(0, p.immunity_until_turn - state.turn_no) / 3.0,
            float(state.is_boosted_cell(p.position)),
            (state.cells[p.position].progress / 100.0
             if p.position in state.cells else 0.0),
            (state.turn_no - p.built_on_turn) / 100.0,
            float(p.built_on_turn == oldest_turn),
            0.0,  # n_connected_neighbors placeholder
            0.0,  # in_storm placeholder
            0.0 if pid in used_authors else 1.0,  # available_this_turn
        ]
        self_mask[i] = True

    enemy_arr = np.zeros((MAX_ENEMY, F_ENEMY), dtype=np.float32)
    enemy_mask = np.zeros(MAX_ENEMY, dtype=bool)
    enemy_ids = sorted(state.enemy_ids)
    for i, eid in enumerate(enemy_ids[:MAX_ENEMY]):
        e = state.plantations[eid]
        dx = (e.position[0] - hq_pos[0]) / 50.0
        dy = (e.position[1] - hq_pos[1]) / 50.0
        enemy_arr[i] = [dx, dy, e.hp / 50.0, 0.0]
        enemy_mask[i] = True

    beaver_arr = np.zeros((MAX_BEAVER, F_BEAVER), dtype=np.float32)
    beaver_mask = np.zeros(MAX_BEAVER, dtype=bool)
    for i, (bid, b) in enumerate(sorted(state.beavers.items())[:MAX_BEAVER]):
        dx = (b.position[0] - hq_pos[0]) / 50.0
        dy = (b.position[1] - hq_pos[1]) / 50.0
        beaver_arr[i] = [dx, dy, b.hp / 100.0, 0.0]
        beaver_mask[i] = True

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
    global_arr[12] = state.score / 1e5

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
            if (x, y) in state.mountains:
                grid[0, gx, gy] = 1.0
            if (x, y) in own_positions:
                grid[1, gx, gy] = 1.0
            if (x, y) in state.constructions:
                grid[2, gx, gy] = state.constructions[(x, y)].progress / 50.0
            if (x, y) in enemy_positions:
                grid[3, gx, gy] = 1.0
            if (x, y) in beaver_positions:
                grid[4, gx, gy] = 1.0
            if (x, y) in state.cells:
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
