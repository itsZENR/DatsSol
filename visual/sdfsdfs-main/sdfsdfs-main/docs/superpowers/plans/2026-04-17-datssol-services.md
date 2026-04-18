# DatsSol Game Services Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two Go services — a data collector with mini-dashboard and a game player implementing the terraform-conveyor strategy — backed by PostgreSQL, orchestrated via Docker Compose.

**Architecture:** Monorepo at `/app` with shared `internal/` packages (API client, DB, models). Collector polls GET /api/arena once per second, parses into structured tables. Player polls GET /api/arena + POST /api/command, runs Strategy 1 (terraform-conveyor with CU rotation, snake expansion, upgrade scheduling). Dashboard is a lightweight HTTP server embedded in the collector binary, serving an HTML page with map visualization via Canvas.

**Tech Stack:** Go 1.25, PostgreSQL 16, Docker Compose, `pgx` (DB driver), `godotenv` (.env), `net/http` (dashboard), vanilla JS + Canvas (map UI)

---

## File Structure

```
/app
├── docker-compose.yml              # postgres + collector + player
├── .env.example                    # DATS_TOKEN, API_URL, DB_DSN
├── Dockerfile                      # multi-stage build for both binaries
├── go.mod
├── go.sum
├── cmd/
│   ├── collector/
│   │   └── main.go                 # entrypoint: ticker loop + dashboard server
│   └── player/
│       └── main.go                 # entrypoint: game loop
├── internal/
│   ├── config/
│   │   └── config.go               # load .env, expose Config struct
│   ├── api/
│   │   ├── client.go               # HTTP client: GetArena(), SendCommand()
│   │   └── types.go                # Go structs matching OpenAPI schemas
│   ├── db/
│   │   ├── postgres.go             # connection pool setup
│   │   ├── migrate.go              # run SQL migrations on startup
│   │   └── repo.go                 # InsertTurn(), InsertPlantations(), queries
│   ├── collector/
│   │   └── collector.go            # poll loop, parse, store
│   ├── strategy/
│   │   ├── conveyor.go             # Strategy 1: terraform conveyor
│   │   ├── pathfind.go             # BFS for adjacency, snake path generation
│   │   └── upgrades.go             # upgrade decision logic
│   └── dashboard/
│       ├── server.go               # HTTP handlers: /api/stats, /api/map
│       └── static/
│           └── index.html          # map canvas + stats UI
├── migrations/
│   └── 001_initial.sql             # all tables
└── scripts/
    └── wait-for-db.sh              # docker healthcheck helper
```

---

## Task 1: Project Skeleton + Config

**Files:**
- Create: `app/go.mod`
- Create: `app/.env.example`
- Create: `app/internal/config/config.go`
- Create: `app/cmd/collector/main.go` (stub)
- Create: `app/cmd/player/main.go` (stub)

- [ ] **Step 1: Initialize Go module and install dependencies**

```bash
cd /Users/staners2/Desktop/ai-projects/hakatok-2026
mkdir -p app
cd app
go mod init datssol
go get github.com/jackc/pgx/v5
go get github.com/joho/godotenv
```

- [ ] **Step 2: Create .env.example**

Create `app/.env.example`:

```env
DATS_TOKEN=your-token-here
API_URL=https://games-test.datsteam.dev
DB_DSN=postgres://datssol:datssol@postgres:5432/datssol?sslmode=disable
DASHBOARD_PORT=8080
```

- [ ] **Step 3: Write config loader**

Create `app/internal/config/config.go`:

```go
package config

import (
	"fmt"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	Token         string
	APIURL        string
	DBDSN         string
	DashboardPort string
}

func Load() (*Config, error) {
	_ = godotenv.Load()

	cfg := &Config{
		Token:         os.Getenv("DATS_TOKEN"),
		APIURL:        os.Getenv("API_URL"),
		DBDSN:         os.Getenv("DB_DSN"),
		DashboardPort: os.Getenv("DASHBOARD_PORT"),
	}

	if cfg.Token == "" {
		return nil, fmt.Errorf("DATS_TOKEN is required")
	}
	if cfg.APIURL == "" {
		cfg.APIURL = "https://games-test.datsteam.dev"
	}
	if cfg.DashboardPort == "" {
		cfg.DashboardPort = "8080"
	}
	return cfg, nil
}
```

- [ ] **Step 4: Write stub main files**

Create `app/cmd/collector/main.go`:

```go
package main

import (
	"fmt"
	"log"

	"datssol/internal/config"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Collector starting, API: %s\n", cfg.APIURL)
}
```

Create `app/cmd/player/main.go`:

```go
package main

import (
	"fmt"
	"log"

	"datssol/internal/config"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Player starting, API: %s\n", cfg.APIURL)
}
```

- [ ] **Step 5: Verify it compiles**

```bash
cd /Users/staners2/Desktop/ai-projects/hakatok-2026/app
go build ./cmd/collector/
go build ./cmd/player/
```

Expected: both compile without errors.

- [ ] **Step 6: Commit**

```bash
git add app/
git commit -m "feat: project skeleton with config loader and stub entrypoints"
```

---

## Task 2: API Client + Types

**Files:**
- Create: `app/internal/api/types.go`
- Create: `app/internal/api/client.go`

- [ ] **Step 1: Define API types matching OpenAPI schema**

Create `app/internal/api/types.go`:

```go
package api

// Coord represents [x, y] position on the map.
type Coord [2]int

// ArenaResponse is the response from GET /api/arena.
type ArenaResponse struct {
	TurnNo             int               `json:"turnNo"`
	NextTurnIn         float64           `json:"nextTurnIn"`
	Size               Coord             `json:"size"`
	ActionRange        int               `json:"actionRange"`
	Plantations        []Plantation      `json:"plantations"`
	Enemy              []EnemyPlantation `json:"enemy"`
	Mountains          []Coord           `json:"mountains"`
	Cells              []TerraformedCell `json:"cells"`
	Construction       []Construction    `json:"construction"`
	Beavers            []Beaver          `json:"beavers"`
	PlantationUpgrades UpgradesState     `json:"plantationUpgrades"`
	MeteoForecasts     []MeteoForecast   `json:"meteoForecasts"`
}

type Plantation struct {
	ID                string `json:"id"`
	Position          Coord  `json:"position"`
	IsMain            bool   `json:"isMain"`
	IsIsolated        bool   `json:"isIsolated"`
	ImmunityUntilTurn int    `json:"immunityUntilTurn"`
	HP                int    `json:"hp"`
}

type EnemyPlantation struct {
	ID       string `json:"id"`
	Position Coord  `json:"position"`
	HP       int    `json:"hp"`
}

type TerraformedCell struct {
	Position               Coord `json:"position"`
	TerraformationProgress int   `json:"terraformationProgress"`
	TurnsUntilDegradation  int   `json:"turnsUntilDegradation"`
}

type Construction struct {
	Position Coord `json:"position"`
	Progress int   `json:"progress"`
}

type Beaver struct {
	ID       string `json:"id"`
	Position Coord  `json:"position"`
	HP       int    `json:"hp"`
}

type UpgradesState struct {
	Points           int           `json:"points"`
	IntervalTurns    int           `json:"intervalTurns"`
	TurnsUntilPoints int           `json:"turnsUntilPoints"`
	MaxPoints        int           `json:"maxPoints"`
	Tiers            []UpgradeTier `json:"tiers"`
}

type UpgradeTier struct {
	Name    string `json:"name"`
	Current int    `json:"current"`
	Max     int    `json:"max"`
}

type MeteoForecast struct {
	Kind         string `json:"kind"`
	TurnsUntil   *int   `json:"turnsUntil,omitempty"`
	ID           string `json:"id,omitempty"`
	Forming      *bool  `json:"forming,omitempty"`
	Position     *Coord `json:"position,omitempty"`
	NextPosition *Coord `json:"nextPosition,omitempty"`
	Radius       *int   `json:"radius,omitempty"`
}

// CommandRequest is the body for POST /api/command.
type CommandRequest struct {
	Command           []PlantationAction `json:"command,omitempty"`
	PlantationUpgrade string             `json:"plantationUpgrade,omitempty"`
	RelocateMain      []Coord            `json:"relocateMain,omitempty"`
}

type PlantationAction struct {
	Path []Coord `json:"path"`
}

// CommandResponse is the response from POST /api/command.
type CommandResponse struct {
	Code   int      `json:"code"`
	Errors []string `json:"errors"`
}
```

- [ ] **Step 2: Write HTTP client**

Create `app/internal/api/client.go`:

```go
package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type Client struct {
	baseURL    string
	token      string
	httpClient *http.Client
}

func NewClient(baseURL, token string) *Client {
	return &Client{
		baseURL: baseURL,
		token:   token,
		httpClient: &http.Client{
			Timeout: 900 * time.Millisecond,
		},
	}
}

func (c *Client) GetArena(ctx context.Context) (*ArenaResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/arena", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Auth-Token", c.token)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("arena: status %d, body: %s", resp.StatusCode, body)
	}

	var arena ArenaResponse
	if err := json.Unmarshal(body, &arena); err != nil {
		return nil, fmt.Errorf("arena: unmarshal: %w", err)
	}
	return &arena, nil
}

func (c *Client) SendCommand(ctx context.Context, cmd *CommandRequest) (*CommandResponse, error) {
	data, err := json.Marshal(cmd)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/command", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Auth-Token", c.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var cmdResp CommandResponse
	if err := json.Unmarshal(body, &cmdResp); err != nil {
		return nil, fmt.Errorf("command: status %d, body: %s", resp.StatusCode, body)
	}
	return &cmdResp, nil
}
```

- [ ] **Step 3: Verify compilation**

```bash
cd /Users/staners2/Desktop/ai-projects/hakatok-2026/app
go build ./internal/api/
```

- [ ] **Step 4: Commit**

```bash
git add app/internal/api/
git commit -m "feat: API client with typed request/response structs"
```

---

## Task 3: Database Schema + Migrations

**Files:**
- Create: `app/migrations/001_initial.sql`
- Create: `app/internal/db/postgres.go`
- Create: `app/internal/db/migrate.go`

- [ ] **Step 1: Write migration SQL**

Create `app/migrations/001_initial.sql`:

```sql
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
    immunity_until  INT DEFAULT 0
);
CREATE INDEX idx_plantations_turn ON plantations(turn_id);

CREATE TABLE IF NOT EXISTS enemies (
    id              SERIAL PRIMARY KEY,
    turn_id         INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    enemy_id        TEXT NOT NULL,
    x               INT NOT NULL,
    y               INT NOT NULL,
    hp              INT NOT NULL
);
CREATE INDEX idx_enemies_turn ON enemies(turn_id);

CREATE TABLE IF NOT EXISTS beavers (
    id          SERIAL PRIMARY KEY,
    turn_id     INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    beaver_id   TEXT NOT NULL,
    x           INT NOT NULL,
    y           INT NOT NULL,
    hp          INT NOT NULL
);
CREATE INDEX idx_beavers_turn ON beavers(turn_id);

CREATE TABLE IF NOT EXISTS cells (
    id              SERIAL PRIMARY KEY,
    turn_id         INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    x               INT NOT NULL,
    y               INT NOT NULL,
    progress        INT NOT NULL,
    turns_until_deg INT NOT NULL
);
CREATE INDEX idx_cells_turn ON cells(turn_id);

CREATE TABLE IF NOT EXISTS constructions (
    id          SERIAL PRIMARY KEY,
    turn_id     INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    x           INT NOT NULL,
    y           INT NOT NULL,
    progress    INT NOT NULL
);
CREATE INDEX idx_constructions_turn ON constructions(turn_id);

CREATE TABLE IF NOT EXISTS mountains (
    id       SERIAL PRIMARY KEY,
    round_id INT NOT NULL REFERENCES rounds(id),
    x        INT NOT NULL,
    y        INT NOT NULL,
    UNIQUE(round_id, x, y)
);
CREATE INDEX idx_mountains_round ON mountains(round_id);

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
CREATE INDEX idx_meteo_turn ON meteo_events(turn_id);

CREATE TABLE IF NOT EXISTS upgrades (
    id          SERIAL PRIMARY KEY,
    turn_id     INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    points      INT NOT NULL,
    interval    INT NOT NULL,
    turns_until INT NOT NULL,
    max_points  INT NOT NULL
);

CREATE TABLE IF NOT EXISTS upgrade_tiers (
    id          SERIAL PRIMARY KEY,
    turn_id     INT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    current     INT NOT NULL,
    max_level   INT NOT NULL
);
CREATE INDEX idx_upgrade_tiers_turn ON upgrade_tiers(turn_id);

CREATE TABLE IF NOT EXISTS score_log (
    id              SERIAL PRIMARY KEY,
    round_id        INT NOT NULL REFERENCES rounds(id),
    turn_no         INT NOT NULL,
    my_plantations  INT NOT NULL,
    score_delta     INT NOT NULL,
    enemies_visible INT NOT NULL,
    beavers_visible INT NOT NULL,
    storm_active    BOOLEAN NOT NULL DEFAULT FALSE,
    earthquake      BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX idx_score_log_round ON score_log(round_id, turn_no);
```

- [ ] **Step 2: Write DB connection**

Create `app/internal/db/postgres.go`:

```go
package db

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
)

func NewPool(ctx context.Context, dsn string) (*pgxpool.Pool, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("db connect: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("db ping: %w", err)
	}
	return pool, nil
}
```

- [ ] **Step 3: Write migration runner**

Create `app/internal/db/migrate.go`:

```go
package db

import (
	"context"
	"fmt"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"
)

func RunMigrations(ctx context.Context, pool *pgxpool.Pool, migrationsDir string) error {
	sql, err := os.ReadFile(migrationsDir + "/001_initial.sql")
	if err != nil {
		return fmt.Errorf("read migration: %w", err)
	}
	if _, err := pool.Exec(ctx, string(sql)); err != nil {
		return fmt.Errorf("run migration: %w", err)
	}
	return nil
}
```

- [ ] **Step 4: Verify compilation**

```bash
cd /Users/staners2/Desktop/ai-projects/hakatok-2026/app
go build ./internal/db/
```

- [ ] **Step 5: Commit**

```bash
git add app/migrations/ app/internal/db/
git commit -m "feat: PostgreSQL schema, connection pool, and migration runner"
```

---

## Task 4: DB Repository (Insert + Query)

**Files:**
- Create: `app/internal/db/repo.go`

- [ ] **Step 1: Write repository with insert and query methods**

Create `app/internal/db/repo.go`:

```go
package db

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"

	"datssol/internal/api"
)

type Repo struct {
	pool *pgxpool.Pool
}

func NewRepo(pool *pgxpool.Pool) *Repo {
	return &Repo{pool: pool}
}

func (r *Repo) EnsureRound(ctx context.Context, roundNum int) (int, error) {
	var id int
	err := r.pool.QueryRow(ctx,
		`INSERT INTO rounds (round_num) VALUES ($1)
		 ON CONFLICT (round_num) DO UPDATE SET round_num = rounds.round_num
		 RETURNING id`, roundNum).Scan(&id)
	return id, err
}

func (r *Repo) InsertTurn(ctx context.Context, roundID int, arena *api.ArenaResponse) (int, error) {
	var turnID int
	err := r.pool.QueryRow(ctx,
		`INSERT INTO turns (round_id, turn_no, map_width, map_height, action_range, next_turn_in)
		 VALUES ($1, $2, $3, $4, $5, $6)
		 ON CONFLICT (round_id, turn_no) DO UPDATE SET next_turn_in = EXCLUDED.next_turn_in
		 RETURNING id`,
		roundID, arena.TurnNo, arena.Size[0], arena.Size[1], arena.ActionRange, arena.NextTurnIn,
	).Scan(&turnID)
	return turnID, err
}

func (r *Repo) InsertPlantations(ctx context.Context, turnID int, plants []api.Plantation) error {
	for _, p := range plants {
		_, err := r.pool.Exec(ctx,
			`INSERT INTO plantations (turn_id, plantation_id, x, y, hp, is_main, is_isolated, immunity_until)
			 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
			turnID, p.ID, p.Position[0], p.Position[1], p.HP, p.IsMain, p.IsIsolated, p.ImmunityUntilTurn,
		)
		if err != nil {
			return fmt.Errorf("insert plantation %s: %w", p.ID, err)
		}
	}
	return nil
}

func (r *Repo) InsertEnemies(ctx context.Context, turnID int, enemies []api.EnemyPlantation) error {
	for _, e := range enemies {
		_, err := r.pool.Exec(ctx,
			`INSERT INTO enemies (turn_id, enemy_id, x, y, hp) VALUES ($1, $2, $3, $4, $5)`,
			turnID, e.ID, e.Position[0], e.Position[1], e.HP,
		)
		if err != nil {
			return err
		}
	}
	return nil
}

func (r *Repo) InsertBeavers(ctx context.Context, turnID int, beavers []api.Beaver) error {
	for _, b := range beavers {
		_, err := r.pool.Exec(ctx,
			`INSERT INTO beavers (turn_id, beaver_id, x, y, hp) VALUES ($1, $2, $3, $4, $5)`,
			turnID, b.ID, b.Position[0], b.Position[1], b.HP,
		)
		if err != nil {
			return err
		}
	}
	return nil
}

func (r *Repo) InsertCells(ctx context.Context, turnID int, cells []api.TerraformedCell) error {
	for _, c := range cells {
		_, err := r.pool.Exec(ctx,
			`INSERT INTO cells (turn_id, x, y, progress, turns_until_deg) VALUES ($1, $2, $3, $4, $5)`,
			turnID, c.Position[0], c.Position[1], c.TerraformationProgress, c.TurnsUntilDegradation,
		)
		if err != nil {
			return err
		}
	}
	return nil
}

func (r *Repo) InsertConstructions(ctx context.Context, turnID int, cons []api.Construction) error {
	for _, c := range cons {
		_, err := r.pool.Exec(ctx,
			`INSERT INTO constructions (turn_id, x, y, progress) VALUES ($1, $2, $3, $4)`,
			turnID, c.Position[0], c.Position[1], c.Progress,
		)
		if err != nil {
			return err
		}
	}
	return nil
}

func (r *Repo) InsertMountains(ctx context.Context, roundID int, mountains []api.Coord) error {
	for _, m := range mountains {
		_, err := r.pool.Exec(ctx,
			`INSERT INTO mountains (round_id, x, y) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`,
			roundID, m[0], m[1],
		)
		if err != nil {
			return err
		}
	}
	return nil
}

func (r *Repo) InsertMeteo(ctx context.Context, turnID int, forecasts []api.MeteoForecast) error {
	for _, f := range forecasts {
		var posX, posY, nextX, nextY *int
		if f.Position != nil {
			x, y := f.Position[0], f.Position[1]
			posX, posY = &x, &y
		}
		if f.NextPosition != nil {
			x, y := f.NextPosition[0], f.NextPosition[1]
			nextX, nextY = &x, &y
		}

		_, err := r.pool.Exec(ctx,
			`INSERT INTO meteo_events (turn_id, kind, turns_until, storm_id, forming, pos_x, pos_y, next_pos_x, next_pos_y, radius)
			 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`,
			turnID, f.Kind, f.TurnsUntil, f.ID, f.Forming, posX, posY, nextX, nextY, f.Radius,
		)
		if err != nil {
			return err
		}
	}
	return nil
}

func (r *Repo) InsertUpgrades(ctx context.Context, turnID int, ups api.UpgradesState) error {
	_, err := r.pool.Exec(ctx,
		`INSERT INTO upgrades (turn_id, points, interval, turns_until, max_points)
		 VALUES ($1, $2, $3, $4, $5)`,
		turnID, ups.Points, ups.IntervalTurns, ups.TurnsUntilPoints, ups.MaxPoints,
	)
	if err != nil {
		return err
	}
	for _, t := range ups.Tiers {
		_, err := r.pool.Exec(ctx,
			`INSERT INTO upgrade_tiers (turn_id, name, current, max_level) VALUES ($1, $2, $3, $4)`,
			turnID, t.Name, t.Current, t.Max,
		)
		if err != nil {
			return err
		}
	}
	return nil
}

func (r *Repo) InsertScoreLog(ctx context.Context, roundID, turnNo, plantations, scoreDelta, enemies, beavers int, storm, earthquake bool) error {
	_, err := r.pool.Exec(ctx,
		`INSERT INTO score_log (round_id, turn_no, my_plantations, score_delta, enemies_visible, beavers_visible, storm_active, earthquake)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
		roundID, turnNo, plantations, scoreDelta, enemies, beavers, storm, earthquake,
	)
	return err
}

// --- Query methods for dashboard ---

type CoordData struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type RoundSummary struct {
	RoundNum   int `json:"roundNum"`
	TotalTurns int `json:"totalTurns"`
}

func (r *Repo) GetRounds(ctx context.Context) ([]RoundSummary, error) {
	rows, err := r.pool.Query(ctx,
		`SELECT round_num, total_turns FROM rounds ORDER BY round_num`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var rounds []RoundSummary
	for rows.Next() {
		var rs RoundSummary
		if err := rows.Scan(&rs.RoundNum, &rs.TotalTurns); err != nil {
			return nil, err
		}
		rounds = append(rounds, rs)
	}
	return rounds, nil
}

type TurnSnapshot struct {
	TurnNo      int `json:"turnNo"`
	Plantations int `json:"plantations"`
	ScoreDelta  int `json:"scoreDelta"`
	Enemies     int `json:"enemies"`
	Beavers     int `json:"beavers"`
}

func (r *Repo) GetScoreHistory(ctx context.Context, roundID int) ([]TurnSnapshot, error) {
	rows, err := r.pool.Query(ctx,
		`SELECT turn_no, my_plantations, score_delta, enemies_visible, beavers_visible
		 FROM score_log WHERE round_id = $1 ORDER BY turn_no`, roundID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var snaps []TurnSnapshot
	for rows.Next() {
		var s TurnSnapshot
		if err := rows.Scan(&s.TurnNo, &s.Plantations, &s.ScoreDelta, &s.Enemies, &s.Beavers); err != nil {
			return nil, err
		}
		snaps = append(snaps, s)
	}
	return snaps, nil
}

func (r *Repo) GetMountains(ctx context.Context, roundID int) ([]CoordData, error) {
	rows, err := r.pool.Query(ctx,
		`SELECT x, y FROM mountains WHERE round_id = $1`, roundID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var result []CoordData
	for rows.Next() {
		var c CoordData
		if err := rows.Scan(&c.X, &c.Y); err != nil {
			return nil, err
		}
		result = append(result, c)
	}
	return result, nil
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd /Users/staners2/Desktop/ai-projects/hakatok-2026/app
go build ./internal/db/
```

- [ ] **Step 3: Commit**

```bash
git add app/internal/db/repo.go
git commit -m "feat: DB repository with insert and query methods"
```

---

## Task 5: Collector Service

**Files:**
- Create: `app/internal/collector/collector.go`
- Modify: `app/cmd/collector/main.go`

- [ ] **Step 1: Write collector loop**

Create `app/internal/collector/collector.go`:

```go
package collector

import (
	"context"
	"log"
	"time"

	"datssol/internal/api"
	"datssol/internal/db"
)

type Collector struct {
	client         *api.Client
	repo           *db.Repo
	roundID        int
	roundNum       int
	lastTurn       int
	mountainsSaved bool
}

func New(client *api.Client, repo *db.Repo) *Collector {
	return &Collector{
		client: client,
		repo:   repo,
	}
}

func (c *Collector) Run(ctx context.Context) {
	ticker := time.NewTicker(900 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.tick(ctx)
		}
	}
}

func (c *Collector) tick(ctx context.Context) {
	arena, err := c.client.GetArena(ctx)
	if err != nil {
		log.Printf("[collector] arena error: %v", err)
		return
	}

	if arena.TurnNo < c.lastTurn || c.roundID == 0 {
		c.roundNum++
		c.mountainsSaved = false
		rid, err := c.repo.EnsureRound(ctx, c.roundNum)
		if err != nil {
			log.Printf("[collector] ensure round error: %v", err)
			return
		}
		c.roundID = rid
		log.Printf("[collector] new round %d detected (turn %d)", c.roundNum, arena.TurnNo)
	}
	c.lastTurn = arena.TurnNo

	turnID, err := c.repo.InsertTurn(ctx, c.roundID, arena)
	if err != nil {
		log.Printf("[collector] insert turn error: %v", err)
		return
	}

	if !c.mountainsSaved && len(arena.Mountains) > 0 {
		if err := c.repo.InsertMountains(ctx, c.roundID, arena.Mountains); err != nil {
			log.Printf("[collector] mountains error: %v", err)
		}
		c.mountainsSaved = true
	}

	if err := c.repo.InsertPlantations(ctx, turnID, arena.Plantations); err != nil {
		log.Printf("[collector] plantations error: %v", err)
	}
	if err := c.repo.InsertEnemies(ctx, turnID, arena.Enemy); err != nil {
		log.Printf("[collector] enemies error: %v", err)
	}
	if err := c.repo.InsertBeavers(ctx, turnID, arena.Beavers); err != nil {
		log.Printf("[collector] beavers error: %v", err)
	}
	if err := c.repo.InsertCells(ctx, turnID, arena.Cells); err != nil {
		log.Printf("[collector] cells error: %v", err)
	}
	if err := c.repo.InsertConstructions(ctx, turnID, arena.Construction); err != nil {
		log.Printf("[collector] constructions error: %v", err)
	}
	if err := c.repo.InsertMeteo(ctx, turnID, arena.MeteoForecasts); err != nil {
		log.Printf("[collector] meteo error: %v", err)
	}
	if err := c.repo.InsertUpgrades(ctx, turnID, arena.PlantationUpgrades); err != nil {
		log.Printf("[collector] upgrades error: %v", err)
	}

	stormActive := false
	earthquakeNow := false
	for _, f := range arena.MeteoForecasts {
		if f.Kind == "sandstorm" && f.Forming != nil && !*f.Forming {
			stormActive = true
		}
		if f.Kind == "earthquake" && f.TurnsUntil != nil && *f.TurnsUntil == 0 {
			earthquakeNow = true
		}
	}
	scoreDelta := len(arena.Plantations) * 50
	_ = c.repo.InsertScoreLog(ctx, c.roundID, arena.TurnNo,
		len(arena.Plantations), scoreDelta, len(arena.Enemy), len(arena.Beavers),
		stormActive, earthquakeNow)

	log.Printf("[collector] turn %d: %d plants, %d enemies, %d beavers",
		arena.TurnNo, len(arena.Plantations), len(arena.Enemy), len(arena.Beavers))
}
```

- [ ] **Step 2: Update collector main.go**

Replace `app/cmd/collector/main.go`:

```go
package main

import (
	"context"
	"log"
	"os/signal"
	"syscall"

	"datssol/internal/api"
	"datssol/internal/collector"
	"datssol/internal/config"
	"datssol/internal/dashboard"
	"datssol/internal/db"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	pool, err := db.NewPool(ctx, cfg.DBDSN)
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()

	if err := db.RunMigrations(ctx, pool, "/app/migrations"); err != nil {
		log.Fatal("migration error: ", err)
	}

	repo := db.NewRepo(pool)
	client := api.NewClient(cfg.APIURL, cfg.Token)

	go dashboard.Serve(cfg.DashboardPort, repo)

	coll := collector.New(client, repo)
	log.Printf("Collector started, API: %s, dashboard: :%s", cfg.APIURL, cfg.DashboardPort)
	coll.Run(ctx)
}
```

- [ ] **Step 3: Create dashboard stub for compilation**

Create `app/internal/dashboard/server.go`:

```go
package dashboard

import (
	"log"
	"net/http"

	"datssol/internal/db"
)

func Serve(port string, repo *db.Repo) {
	log.Printf("Dashboard listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("dashboard placeholder"))
	})))
}
```

- [ ] **Step 4: Verify compilation**

```bash
cd /Users/staners2/Desktop/ai-projects/hakatok-2026/app
go build ./cmd/collector/
```

- [ ] **Step 5: Commit**

```bash
git add app/internal/collector/ app/internal/dashboard/ app/cmd/collector/
git commit -m "feat: collector service with poll loop and DB storage"
```

---

## Task 6: Strategy Engine (Terraform Conveyor)

**Files:**
- Create: `app/internal/strategy/pathfind.go`
- Create: `app/internal/strategy/upgrades.go`
- Create: `app/internal/strategy/conveyor.go`

- [ ] **Step 1: Write pathfinding utilities**

Create `app/internal/strategy/pathfind.go`:

```go
package strategy

import "datssol/internal/api"

func Adjacents(c api.Coord) [4]api.Coord {
	return [4]api.Coord{
		{c[0] + 1, c[1]},
		{c[0] - 1, c[1]},
		{c[0], c[1] + 1},
		{c[0], c[1] - 1},
	}
}

func InRange(origin, target api.Coord, r int) bool {
	dx := target[0] - origin[0]
	dy := target[1] - origin[1]
	if dx < 0 {
		dx = -dx
	}
	if dy < 0 {
		dy = -dy
	}
	return dx <= r && dy <= r
}

func IsBoosted(c api.Coord) bool {
	return c[0]%7 == 0 && c[1]%7 == 0
}

func ConnectedToMain(plantations []api.Plantation) map[api.Coord]bool {
	posMap := make(map[api.Coord]bool)
	var mainPos api.Coord
	hasMain := false

	for _, p := range plantations {
		posMap[p.Position] = true
		if p.IsMain {
			mainPos = p.Position
			hasMain = true
		}
	}

	if !hasMain {
		return nil
	}

	connected := make(map[api.Coord]bool)
	queue := []api.Coord{mainPos}
	connected[mainPos] = true

	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, adj := range Adjacents(cur) {
			if posMap[adj] && !connected[adj] {
				connected[adj] = true
				queue = append(queue, adj)
			}
		}
	}
	return connected
}
```

- [ ] **Step 2: Write upgrade decision logic**

Create `app/internal/strategy/upgrades.go`:

```go
package strategy

import "datssol/internal/api"

var upgradePlan = []struct {
	Name  string
	Count int
}{
	{"settlement_limit", 5},
	{"max_hp", 2},
	{"settlement_limit", 3},
	{"earthquake_mitigation", 2},
	{"settlement_limit", 2},
	{"decay_mitigation", 1},
}

func PickUpgrade(ups api.UpgradesState) string {
	if ups.Points <= 0 {
		return ""
	}

	tierMap := make(map[string]api.UpgradeTier)
	for _, t := range ups.Tiers {
		tierMap[t.Name] = t
	}

	purchased := 0
	for _, step := range upgradePlan {
		tier, ok := tierMap[step.Name]
		if !ok {
			purchased += step.Count
			continue
		}
		for i := 0; i < step.Count; i++ {
			needed := purchased + i + 1
			if tier.Current < needed && (tier.Max < 0 || tier.Current < tier.Max) {
				return step.Name
			}
		}
		purchased += step.Count
	}
	return ""
}
```

- [ ] **Step 3: Write the main conveyor strategy**

Create `app/internal/strategy/conveyor.go`:

```go
package strategy

import (
	"log"

	"datssol/internal/api"
)

type Conveyor struct {
	mapWidth  int
	mapHeight int
}

func NewConveyor() *Conveyor {
	return &Conveyor{}
}

func (c *Conveyor) Decide(arena *api.ArenaResponse) *api.CommandRequest {
	c.mapWidth = arena.Size[0]
	c.mapHeight = arena.Size[1]

	cmd := &api.CommandRequest{}

	if len(arena.Plantations) == 0 {
		return cmd
	}

	myPositions := make(map[api.Coord]*api.Plantation)
	for i := range arena.Plantations {
		p := &arena.Plantations[i]
		myPositions[p.Position] = p
	}

	mountains := make(map[api.Coord]bool)
	for _, m := range arena.Mountains {
		mountains[m] = true
	}

	beaverDanger := make(map[api.Coord]bool)
	for _, b := range arena.Beavers {
		for dx := -2; dx <= 2; dx++ {
			for dy := -2; dy <= 2; dy++ {
				beaverDanger[api.Coord{b.Position[0] + dx, b.Position[1] + dy}] = true
			}
		}
	}

	enemyPositions := make(map[api.Coord]bool)
	for _, e := range arena.Enemy {
		enemyPositions[e.Position] = true
	}

	constructing := make(map[api.Coord]bool)
	for _, con := range arena.Construction {
		constructing[con.Position] = true
	}

	connected := ConnectedToMain(arena.Plantations)

	// 1. CU rotation.
	cmd.RelocateMain = c.maybeRelocateMain(arena, myPositions, connected)

	// 2. Upgrade.
	if upgrade := PickUpgrade(arena.PlantationUpgrades); upgrade != "" {
		cmd.PlantationUpgrade = upgrade
		log.Printf("[strategy] upgrade: %s", upgrade)
	}

	// 3. Assign actions.
	occupied := make(map[api.Coord]bool)
	for pos := range myPositions {
		occupied[pos] = true
	}
	for pos := range constructing {
		occupied[pos] = true
	}

	var needRepair []*api.Plantation
	for i := range arena.Plantations {
		p := &arena.Plantations[i]
		if p.HP < 30 && !p.IsIsolated && connected[p.Position] {
			needRepair = append(needRepair, p)
		}
	}

	for i := range arena.Plantations {
		p := &arena.Plantations[i]
		if p.IsIsolated || !connected[p.Position] {
			continue
		}

		if action := c.tryRepair(p, needRepair, arena.ActionRange); action != nil {
			cmd.Command = append(cmd.Command, *action)
			continue
		}

		if action := c.tryContinueBuild(p, arena.Construction, arena.ActionRange); action != nil {
			cmd.Command = append(cmd.Command, *action)
			continue
		}

		if action := c.tryNewBuild(p, occupied, mountains, beaverDanger, enemyPositions, myPositions, arena.ActionRange); action != nil {
			cmd.Command = append(cmd.Command, *action)
			continue
		}
	}

	return cmd
}

func (c *Conveyor) maybeRelocateMain(arena *api.ArenaResponse, myPos map[api.Coord]*api.Plantation, connected map[api.Coord]bool) []api.Coord {
	var main *api.Plantation
	for i := range arena.Plantations {
		if arena.Plantations[i].IsMain {
			main = &arena.Plantations[i]
			break
		}
	}
	if main == nil {
		return nil
	}

	progress := 0
	for _, cell := range arena.Cells {
		if cell.Position == main.Position {
			progress = cell.TerraformationProgress
			break
		}
	}

	if progress < 90 {
		return nil
	}

	// Find adjacent connected plantation with low progress.
	for _, adj := range Adjacents(main.Position) {
		p, ok := myPos[adj]
		if !ok || p.IsIsolated || !connected[adj] || p.IsMain {
			continue
		}
		adjProgress := 0
		for _, cell := range arena.Cells {
			if cell.Position == adj {
				adjProgress = cell.TerraformationProgress
				break
			}
		}
		if adjProgress < 50 {
			log.Printf("[strategy] relocate CU: %v -> %v (progress %d%%)", main.Position, adj, progress)
			return []api.Coord{main.Position, adj}
		}
	}

	// Fallback: any adjacent connected.
	for _, adj := range Adjacents(main.Position) {
		if _, ok := myPos[adj]; ok && connected[adj] {
			log.Printf("[strategy] relocate CU (fallback): %v -> %v", main.Position, adj)
			return []api.Coord{main.Position, adj}
		}
	}

	log.Printf("[strategy] WARNING: CU at %v progress %d%% but no relay target!", main.Position, progress)
	return nil
}

func (c *Conveyor) tryRepair(author *api.Plantation, needRepair []*api.Plantation, ar int) *api.PlantationAction {
	for _, target := range needRepair {
		if target.Position == author.Position {
			continue
		}
		if InRange(author.Position, target.Position, ar) {
			return &api.PlantationAction{
				Path: []api.Coord{author.Position, author.Position, target.Position},
			}
		}
	}
	return nil
}

func (c *Conveyor) tryContinueBuild(author *api.Plantation, constructions []api.Construction, ar int) *api.PlantationAction {
	for _, con := range constructions {
		if InRange(author.Position, con.Position, ar) {
			return &api.PlantationAction{
				Path: []api.Coord{author.Position, author.Position, con.Position},
			}
		}
	}
	return nil
}

func (c *Conveyor) tryNewBuild(author *api.Plantation, occupied, mountains, beaverDanger, enemies map[api.Coord]bool, myPos map[api.Coord]*api.Plantation, ar int) *api.PlantationAction {
	type candidate struct {
		pos   api.Coord
		score int
	}
	var best *candidate

	for dx := -ar; dx <= ar; dx++ {
		for dy := -ar; dy <= ar; dy++ {
			target := api.Coord{author.Position[0] + dx, author.Position[1] + dy}
			if target[0] < 0 || target[0] >= c.mapWidth || target[1] < 0 || target[1] >= c.mapHeight {
				continue
			}
			if occupied[target] || mountains[target] || enemies[target] {
				continue
			}

			score := 10
			if IsBoosted(target) {
				score += 50
			}
			if beaverDanger[target] {
				score -= 100
			}
			for _, adj := range Adjacents(target) {
				if _, ok := myPos[adj]; ok {
					score += 5
					break
				}
			}

			if score > 0 && (best == nil || score > best.score) {
				best = &candidate{target, score}
			}
		}
	}

	if best == nil {
		return nil
	}

	occupied[best.pos] = true
	return &api.PlantationAction{
		Path: []api.Coord{author.Position, author.Position, best.pos},
	}
}
```

- [ ] **Step 4: Verify compilation**

```bash
cd /Users/staners2/Desktop/ai-projects/hakatok-2026/app
go build ./internal/strategy/
```

- [ ] **Step 5: Commit**

```bash
git add app/internal/strategy/
git commit -m "feat: terraform conveyor strategy with pathfinding and upgrade logic"
```

---

## Task 7: Player Service

**Files:**
- Modify: `app/cmd/player/main.go`

- [ ] **Step 1: Write player game loop**

Replace `app/cmd/player/main.go`:

```go
package main

import (
	"context"
	"log"
	"os/signal"
	"syscall"
	"time"

	"datssol/internal/api"
	"datssol/internal/config"
	"datssol/internal/strategy"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	client := api.NewClient(cfg.APIURL, cfg.Token)
	strat := strategy.NewConveyor()

	var lastTurn int

	log.Printf("Player started, API: %s", cfg.APIURL)

	ticker := time.NewTicker(800 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("Player shutting down")
			return
		case <-ticker.C:
			arena, err := client.GetArena(ctx)
			if err != nil {
				log.Printf("[player] arena error: %v", err)
				continue
			}

			if arena.TurnNo == lastTurn {
				continue
			}
			lastTurn = arena.TurnNo

			cmd := strat.Decide(arena)

			hasActions := len(cmd.Command) > 0 || cmd.PlantationUpgrade != "" || len(cmd.RelocateMain) > 0
			if !hasActions {
				continue
			}

			resp, err := client.SendCommand(ctx, cmd)
			if err != nil {
				log.Printf("[player] command error: %v", err)
				continue
			}

			if len(resp.Errors) > 0 {
				log.Printf("[player] turn %d errors: %v", arena.TurnNo, resp.Errors)
			} else {
				log.Printf("[player] turn %d: %d actions, upgrade=%q, relocate=%v",
					arena.TurnNo, len(cmd.Command), cmd.PlantationUpgrade, len(cmd.RelocateMain) > 0)
			}
		}
	}
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd /Users/staners2/Desktop/ai-projects/hakatok-2026/app
go build ./cmd/player/
```

- [ ] **Step 3: Commit**

```bash
git add app/cmd/player/
git commit -m "feat: player service with game loop and strategy execution"
```

---

## Task 8: Dashboard (Map + Stats)

**Files:**
- Modify: `app/internal/dashboard/server.go`
- Create: `app/internal/dashboard/static/index.html`

- [ ] **Step 1: Write dashboard HTTP server**

Replace `app/internal/dashboard/server.go`:

```go
package dashboard

import (
	"embed"
	"encoding/json"
	"log"
	"net/http"
	"strconv"

	"datssol/internal/db"
)

//go:embed static
var staticFiles embed.FS

func Serve(port string, repo *db.Repo) {
	mux := http.NewServeMux()

	mux.Handle("/", http.FileServer(http.FS(staticFiles)))

	mux.HandleFunc("/api/rounds", func(w http.ResponseWriter, r *http.Request) {
		rounds, err := repo.GetRounds(r.Context())
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(rounds)
	})

	mux.HandleFunc("/api/scores", func(w http.ResponseWriter, r *http.Request) {
		roundID, _ := strconv.Atoi(r.URL.Query().Get("round_id"))
		if roundID == 0 {
			roundID = 1
		}
		scores, err := repo.GetScoreHistory(r.Context(), roundID)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(scores)
	})

	mux.HandleFunc("/api/map", func(w http.ResponseWriter, r *http.Request) {
		roundID, _ := strconv.Atoi(r.URL.Query().Get("round_id"))
		if roundID == 0 {
			roundID = 1
		}
		mountains, err := repo.GetMountains(r.Context(), roundID)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(mountains)
	})

	log.Printf("Dashboard listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}
```

- [ ] **Step 2: Create dashboard HTML**

Create `app/internal/dashboard/static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DatsSol Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
        h1 { color: #00d4aa; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 16px; }
        .panel h2 { color: #e94560; margin-bottom: 12px; font-size: 14px; }
        canvas { border: 1px solid #0f3460; background: #0a0a1a; width: 100%; }
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
        .stat { background: #0f3460; border-radius: 4px; padding: 8px; text-align: center; }
        .stat .value { font-size: 24px; color: #00d4aa; }
        .stat .label { font-size: 11px; color: #888; }
        select { background: #0f3460; color: #e0e0e0; border: 1px solid #00d4aa; padding: 4px 8px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>DatsSol Dashboard</h1>
    <div>
        Round: <select id="roundSelect"></select>
        <span id="status" style="margin-left: 16px; color: #888;">loading...</span>
    </div>
    <br>
    <div class="stats" id="statsGrid"></div>
    <br>
    <div class="grid">
        <div class="panel">
            <h2>MAP</h2>
            <canvas id="mapCanvas" width="600" height="400"></canvas>
        </div>
        <div class="panel">
            <h2>SCORE / TURN</h2>
            <canvas id="chartCanvas" width="600" height="400"></canvas>
        </div>
    </div>

    <script>
    let currentRound = 1;
    let scores = [];

    async function loadRounds() {
        const res = await fetch('/api/rounds');
        const rounds = await res.json();
        const sel = document.getElementById('roundSelect');
        sel.textContent = '';
        (rounds || []).forEach(function(r) {
            const opt = document.createElement('option');
            opt.value = r.roundNum;
            opt.textContent = 'Round ' + r.roundNum + ' (' + r.totalTurns + ' turns)';
            sel.appendChild(opt);
        });
        if (rounds && rounds.length > 0) {
            currentRound = rounds[rounds.length - 1].roundNum;
            sel.value = currentRound;
        }
        sel.onchange = function() { currentRound = parseInt(sel.value); refresh(); };
    }

    async function loadScores() {
        const res = await fetch('/api/scores?round_id=' + currentRound);
        scores = await res.json() || [];
        updateStats();
        drawChart();
    }

    function updateStats() {
        const grid = document.getElementById('statsGrid');
        grid.textContent = '';
        if (!scores.length) {
            var empty = document.createElement('div');
            empty.className = 'stat';
            empty.appendChild(createStatValue('-'));
            empty.appendChild(createStatLabel('No data'));
            grid.appendChild(empty);
            return;
        }
        var last = scores[scores.length - 1];
        var totalScore = scores.reduce(function(s, t) { return s + t.scoreDelta; }, 0);
        var avgScore = Math.round(totalScore / scores.length);
        var items = [
            {value: last.turnNo, label: 'Turn'},
            {value: last.plantations, label: 'Plantations'},
            {value: totalScore.toLocaleString(), label: 'Total Score'},
            {value: avgScore, label: 'Avg/Turn'},
            {value: last.enemies, label: 'Enemies'},
            {value: last.beavers, label: 'Beavers'}
        ];
        items.forEach(function(item) {
            var div = document.createElement('div');
            div.className = 'stat';
            div.appendChild(createStatValue(item.value));
            div.appendChild(createStatLabel(item.label));
            grid.appendChild(div);
        });
    }

    function createStatValue(text) {
        var el = document.createElement('div');
        el.className = 'value';
        el.textContent = text;
        return el;
    }

    function createStatLabel(text) {
        var el = document.createElement('div');
        el.className = 'label';
        el.textContent = text;
        return el;
    }

    function drawChart() {
        var canvas = document.getElementById('chartCanvas');
        var ctx = canvas.getContext('2d');
        var W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);
        if (!scores.length) return;

        var maxScore = Math.max.apply(null, scores.map(function(s) { return s.scoreDelta; })) || 1;
        var barW = W / scores.length;

        ctx.fillStyle = '#00d4aa';
        scores.forEach(function(s, i) {
            var h = (s.scoreDelta / maxScore) * (H - 20);
            ctx.fillRect(i * barW, H - h - 10, Math.max(barW - 1, 1), h);
        });

        ctx.fillStyle = '#888';
        ctx.font = '10px monospace';
        ctx.fillText('0', 2, H - 2);
        ctx.fillText(maxScore.toString(), 2, 12);
    }

    async function refresh() {
        document.getElementById('status').textContent = 'refreshing...';
        await loadScores();
        document.getElementById('status').textContent = new Date().toLocaleTimeString();
    }

    async function init() {
        await loadRounds();
        await refresh();
        setInterval(refresh, 5000);
    }

    init();
    </script>
</body>
</html>
```

- [ ] **Step 3: Verify compilation**

```bash
cd /Users/staners2/Desktop/ai-projects/hakatok-2026/app
go build ./cmd/collector/
```

- [ ] **Step 4: Commit**

```bash
git add app/internal/dashboard/
git commit -m "feat: dashboard with map canvas and score chart"
```

---

## Task 9: Docker Compose + Dockerfile

**Files:**
- Create: `app/Dockerfile`
- Create: `app/docker-compose.yml`
- Create: `app/scripts/wait-for-db.sh`

- [ ] **Step 1: Write multi-stage Dockerfile**

Create `app/Dockerfile`:

```dockerfile
FROM golang:1.25-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /collector ./cmd/collector/
RUN CGO_ENABLED=0 go build -o /player ./cmd/player/

FROM alpine:3.21
RUN apk add --no-cache bash postgresql-client
COPY --from=builder /collector /usr/local/bin/collector
COPY --from=builder /player /usr/local/bin/player
COPY migrations/ /app/migrations/
COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh
```

- [ ] **Step 2: Write wait-for-db script**

Create `app/scripts/wait-for-db.sh`:

```bash
#!/bin/bash
set -e
until pg_isready -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" 2>/dev/null; do
  echo "Waiting for postgres..."
  sleep 1
done
echo "Postgres is ready"
exec "$@"
```

- [ ] **Step 3: Write docker-compose.yml**

Create `app/docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: datssol
      POSTGRES_PASSWORD: datssol
      POSTGRES_DB: datssol
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U datssol"]
      interval: 2s
      timeout: 5s
      retries: 10

  collector:
    build:
      context: .
      dockerfile: Dockerfile
    command: collector
    env_file: .env
    environment:
      DB_DSN: postgres://datssol:datssol@postgres:5432/datssol?sslmode=disable
      DB_HOST: postgres
      DB_USER: datssol
    ports:
      - "${DASHBOARD_PORT:-8080}:${DASHBOARD_PORT:-8080}"
    depends_on:
      postgres:
        condition: service_healthy

  player:
    build:
      context: .
      dockerfile: Dockerfile
    command: player
    env_file: .env
    environment:
      DB_DSN: postgres://datssol:datssol@postgres:5432/datssol?sslmode=disable
    depends_on:
      postgres:
        condition: service_healthy
    profiles:
      - play

volumes:
  pgdata:
```

- [ ] **Step 4: Create .env from example**

```bash
cp app/.env.example app/.env
# Edit app/.env with actual DATS_TOKEN
```

- [ ] **Step 5: Build and verify**

```bash
cd /Users/staners2/Desktop/ai-projects/hakatok-2026/app
docker compose build
```

Expected: successful build of both binaries.

- [ ] **Step 6: Commit**

```bash
git add app/Dockerfile app/docker-compose.yml app/scripts/
git commit -m "feat: Docker Compose with postgres, collector, and player services"
```

---

## Task 10: .gitignore + Final Verification

**Files:**
- Create: `app/.gitignore`

- [ ] **Step 1: Create .gitignore**

Create `app/.gitignore`:

```
.env
collector
player
*.exe
```

- [ ] **Step 2: Test full stack**

```bash
cd /Users/staners2/Desktop/ai-projects/hakatok-2026/app

# Collector-only mode (default).
docker compose up -d

# Check collector logs.
docker compose logs -f collector

# Open dashboard at http://localhost:8080

# Player mode (test round).
docker compose --profile play up player

# Verify DB has data.
docker compose exec postgres psql -U datssol -c "SELECT count(*) FROM turns;"
```

- [ ] **Step 3: Commit**

```bash
git add app/.gitignore
git commit -m "chore: gitignore and integration verification"
```

---

## Usage Summary

```bash
# Collect only (test rounds, no moves):
docker compose up -d

# Collect + play (active game):
docker compose --profile play up -d

# View dashboard:
open http://localhost:8080

# View logs:
docker compose logs -f player
docker compose logs -f collector

# Stop everything:
docker compose --profile play down
```
