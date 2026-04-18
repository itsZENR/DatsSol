CREATE TABLE IF NOT EXISTS rounds (
    id          SERIAL PRIMARY KEY,
    round_num   INT NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at    TIMESTAMPTZ,
    total_turns INT DEFAULT 0,
    UNIQUE(round_num)
);

CREATE TABLE IF NOT EXISTS turns (
    id          SERIAL PRIMARY KEY,
    round_id    INT NOT NULL REFERENCES rounds(id),
    turn_no     INT NOT NULL,
    map_width   INT NOT NULL,
    map_height  INT NOT NULL,
    action_range INT NOT NULL,
    next_turn_in REAL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(round_id, turn_no)
);

CREATE TABLE IF NOT EXISTS plantations (
    id              SERIAL PRIMARY KEY,
    turn_id         INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    plantation_id   TEXT NOT NULL,
    x               INT NOT NULL,
    y               INT NOT NULL,
    hp              INT NOT NULL,
    is_main         BOOLEAN NOT NULL DEFAULT FALSE,
    is_isolated     BOOLEAN NOT NULL DEFAULT FALSE,
    immunity_until  INT DEFAULT 0,
    UNIQUE(turn_id, plantation_id)
);
CREATE INDEX IF NOT EXISTS idx_plantations_turn ON plantations(turn_id);

CREATE TABLE IF NOT EXISTS enemies (
    id              SERIAL PRIMARY KEY,
    turn_id         INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    enemy_id        TEXT NOT NULL,
    x               INT NOT NULL,
    y               INT NOT NULL,
    hp              INT NOT NULL,
    UNIQUE(turn_id, enemy_id)
);
CREATE INDEX IF NOT EXISTS idx_enemies_turn ON enemies(turn_id);

CREATE TABLE IF NOT EXISTS beavers (
    id          SERIAL PRIMARY KEY,
    turn_id     INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    beaver_id   TEXT NOT NULL,
    x           INT NOT NULL,
    y           INT NOT NULL,
    hp          INT NOT NULL,
    UNIQUE(turn_id, beaver_id)
);
CREATE INDEX IF NOT EXISTS idx_beavers_turn ON beavers(turn_id);

CREATE TABLE IF NOT EXISTS cells (
    id              SERIAL PRIMARY KEY,
    turn_id         INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    x               INT NOT NULL,
    y               INT NOT NULL,
    progress        INT NOT NULL,
    turns_until_deg INT NOT NULL,
    UNIQUE(turn_id, x, y)
);
CREATE INDEX IF NOT EXISTS idx_cells_turn ON cells(turn_id);

CREATE TABLE IF NOT EXISTS constructions (
    id          SERIAL PRIMARY KEY,
    turn_id     INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    x           INT NOT NULL,
    y           INT NOT NULL,
    progress    INT NOT NULL,
    UNIQUE(turn_id, x, y)
);
CREATE INDEX IF NOT EXISTS idx_constructions_turn ON constructions(turn_id);

CREATE TABLE IF NOT EXISTS mountains (
    id       SERIAL PRIMARY KEY,
    round_id INT NOT NULL REFERENCES rounds(id),
    x        INT NOT NULL,
    y        INT NOT NULL,
    UNIQUE(round_id, x, y)
);
CREATE INDEX IF NOT EXISTS idx_mountains_round ON mountains(round_id);

CREATE TABLE IF NOT EXISTS meteo_events (
    id              SERIAL PRIMARY KEY,
    turn_id         INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,
    turns_until     INT,
    storm_id        TEXT,
    forming         BOOLEAN,
    pos_x           INT,
    pos_y           INT,
    next_pos_x      INT,
    next_pos_y      INT,
    radius          INT
);
CREATE INDEX IF NOT EXISTS idx_meteo_turn ON meteo_events(turn_id);

CREATE TABLE IF NOT EXISTS upgrades (
    id          SERIAL PRIMARY KEY,
    turn_id     INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    points      INT NOT NULL,
    interval    INT NOT NULL,
    turns_until INT NOT NULL,
    max_points  INT NOT NULL,
    UNIQUE(turn_id)
);

CREATE TABLE IF NOT EXISTS upgrade_tiers (
    id          SERIAL PRIMARY KEY,
    turn_id     INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    current     INT NOT NULL,
    max_level   INT NOT NULL,
    UNIQUE(turn_id, name)
);
CREATE INDEX IF NOT EXISTS idx_upgrade_tiers_turn ON upgrade_tiers(turn_id);

CREATE TABLE IF NOT EXISTS score_log (
    id              SERIAL PRIMARY KEY,
    round_id        INT NOT NULL REFERENCES rounds(id),
    turn_no         INT NOT NULL,
    my_plantations  INT NOT NULL,
    score_delta     INT NOT NULL,
    enemies_visible INT NOT NULL,
    beavers_visible INT NOT NULL,
    storm_active    BOOLEAN NOT NULL DEFAULT FALSE,
    earthquake      BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(round_id, turn_no)
);
CREATE INDEX IF NOT EXISTS idx_score_log_round ON score_log(round_id, turn_no);

CREATE TABLE IF NOT EXISTS game_logs (
    id          SERIAL PRIMARY KEY,
    round_id    INT NOT NULL REFERENCES rounds(id),
    log_time    TEXT NOT NULL,
    message     TEXT NOT NULL,
    UNIQUE(round_id, log_time, message)
);
CREATE INDEX IF NOT EXISTS idx_game_logs_round ON game_logs(round_id);

CREATE TABLE IF NOT EXISTS strategy_config (
    id          SERIAL PRIMARY KEY,
    round_id    INT REFERENCES rounds(id),
    param_name  TEXT NOT NULL,
    param_value REAL NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(round_id, param_name)
);

CREATE TABLE IF NOT EXISTS strategy_log (
    id          SERIAL PRIMARY KEY,
    round_id    INT REFERENCES rounds(id),
    analysis    JSONB NOT NULL,
    changes     JSONB NOT NULL,
    top10_scores JSONB,
    our_score   INT,
    our_rank    INT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS external_stats (
    id          SERIAL PRIMARY KEY,
    realm_name  TEXT NOT NULL,
    team_name   TEXT NOT NULL,
    score       INT NOT NULL,
    terraforming_score INT,
    plantations_destroyed INT,
    beaver_lodges_destroyed INT,
    deaths      INT,
    kills       INT,
    fetched_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(realm_name, team_name)
);
