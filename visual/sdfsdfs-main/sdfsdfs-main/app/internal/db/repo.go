package db

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"

	"datssol/internal/api"
)

// Repo wraps a pgxpool.Pool and provides all insert/query operations.
type Repo struct {
	pool *pgxpool.Pool
}

// NewRepo creates a new Repo backed by the given connection pool.
func NewRepo(pool *pgxpool.Pool) *Repo {
	return &Repo{pool: pool}
}

// Pool returns the underlying connection pool.
func (r *Repo) Pool() *pgxpool.Pool {
	return r.pool
}

// ---------------------------------------------------------------------------
// Dashboard result types
// ---------------------------------------------------------------------------

// CoordData is a flat x/y coordinate for JSON serialisation.
type CoordData struct {
	X int `json:"x"`
	Y int `json:"y"`
}

// RoundSummary holds high-level information about one competition round.
type RoundSummary struct {
	ID         int `json:"id"`
	RoundNum   int `json:"roundNum"`
	TotalTurns int `json:"totalTurns"`
}

// TurnSnapshot is one row from the score_log table used by the dashboard.
type TurnSnapshot struct {
	TurnNo      int `json:"turnNo"`
	Plantations int `json:"plantations"`
	ScoreDelta  int `json:"scoreDelta"`
	Enemies     int `json:"enemies"`
	Beavers     int `json:"beavers"`
}

// ---------------------------------------------------------------------------
// Insert methods
// ---------------------------------------------------------------------------

// EnsureRound upserts a round row by round_num and returns the internal id.
func (r *Repo) EnsureRound(ctx context.Context, roundNum int) (int, error) {
	const q = `
		INSERT INTO rounds (round_num)
		VALUES ($1)
		ON CONFLICT (round_num) DO UPDATE SET round_num = EXCLUDED.round_num
		RETURNING id`
	var id int
	if err := r.pool.QueryRow(ctx, q, roundNum).Scan(&id); err != nil {
		return 0, fmt.Errorf("EnsureRound: %w", err)
	}
	return id, nil
}

// InsertTurn upserts a turn row and returns its internal id.
func (r *Repo) InsertTurn(ctx context.Context, roundID int, arena *api.ArenaResponse) (int, error) {
	const q = `
		INSERT INTO turns (round_id, turn_no, map_width, map_height, action_range, next_turn_in)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (round_id, turn_no) DO UPDATE
			SET map_width    = EXCLUDED.map_width,
			    map_height   = EXCLUDED.map_height,
			    action_range = EXCLUDED.action_range,
			    next_turn_in = EXCLUDED.next_turn_in
		RETURNING id`
	var id int
	err := r.pool.QueryRow(ctx, q,
		roundID,
		arena.TurnNo,
		arena.Size[0],
		arena.Size[1],
		arena.ActionRange,
		arena.NextTurnIn,
	).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("InsertTurn: %w", err)
	}
	return id, nil
}

// InsertPlantations bulk-inserts plantation snapshots for the given turn.
func (r *Repo) InsertPlantations(ctx context.Context, turnID int, plants []api.Plantation) error {
	for i := range plants {
		p := &plants[i]
		const q = `
			INSERT INTO plantations
				(turn_id, plantation_id, x, y, hp, is_main, is_isolated, immunity_until)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			ON CONFLICT (turn_id, plantation_id) DO UPDATE
				SET x              = EXCLUDED.x,
				    y              = EXCLUDED.y,
				    hp             = EXCLUDED.hp,
				    is_main        = EXCLUDED.is_main,
				    is_isolated    = EXCLUDED.is_isolated,
				    immunity_until = EXCLUDED.immunity_until`
		if _, err := r.pool.Exec(ctx, q,
			turnID,
			p.ID,
			p.Position[0],
			p.Position[1],
			p.HP,
			p.IsMain,
			p.IsIsolated,
			p.ImmunityUntilTurn,
		); err != nil {
			return fmt.Errorf("InsertPlantations[%d]: %w", i, err)
		}
	}
	return nil
}

// InsertEnemies bulk-inserts visible enemy plantation snapshots.
func (r *Repo) InsertEnemies(ctx context.Context, turnID int, enemies []api.EnemyPlantation) error {
	for i := range enemies {
		e := &enemies[i]
		const q = `
			INSERT INTO enemies (turn_id, enemy_id, x, y, hp)
			VALUES ($1, $2, $3, $4, $5)
			ON CONFLICT (turn_id, enemy_id) DO UPDATE
				SET x  = EXCLUDED.x,
				    y  = EXCLUDED.y,
				    hp = EXCLUDED.hp`
		if _, err := r.pool.Exec(ctx, q,
			turnID, e.ID, e.Position[0], e.Position[1], e.HP,
		); err != nil {
			return fmt.Errorf("InsertEnemies[%d]: %w", i, err)
		}
	}
	return nil
}

// InsertBeavers bulk-inserts beaver snapshots.
func (r *Repo) InsertBeavers(ctx context.Context, turnID int, beavers []api.Beaver) error {
	for i := range beavers {
		b := &beavers[i]
		const q = `
			INSERT INTO beavers (turn_id, beaver_id, x, y, hp)
			VALUES ($1, $2, $3, $4, $5)
			ON CONFLICT (turn_id, beaver_id) DO UPDATE
				SET x  = EXCLUDED.x,
				    y  = EXCLUDED.y,
				    hp = EXCLUDED.hp`
		if _, err := r.pool.Exec(ctx, q,
			turnID, b.ID, b.Position[0], b.Position[1], b.HP,
		); err != nil {
			return fmt.Errorf("InsertBeavers[%d]: %w", i, err)
		}
	}
	return nil
}

// InsertCells bulk-inserts terraformed cell snapshots.
func (r *Repo) InsertCells(ctx context.Context, turnID int, cells []api.TerraformedCell) error {
	for i := range cells {
		c := &cells[i]
		const q = `
			INSERT INTO cells (turn_id, x, y, progress, turns_until_deg)
			VALUES ($1, $2, $3, $4, $5)
			ON CONFLICT (turn_id, x, y) DO UPDATE
				SET progress      = EXCLUDED.progress,
				    turns_until_deg = EXCLUDED.turns_until_deg`
		if _, err := r.pool.Exec(ctx, q,
			turnID,
			c.Position[0],
			c.Position[1],
			c.TerraformationProgress,
			c.TurnsUntilDegradation,
		); err != nil {
			return fmt.Errorf("InsertCells[%d]: %w", i, err)
		}
	}
	return nil
}

// InsertConstructions bulk-inserts construction site snapshots.
func (r *Repo) InsertConstructions(ctx context.Context, turnID int, cons []api.Construction) error {
	for i := range cons {
		c := &cons[i]
		const q = `
			INSERT INTO constructions (turn_id, x, y, progress)
			VALUES ($1, $2, $3, $4)
			ON CONFLICT (turn_id, x, y) DO UPDATE
				SET progress = EXCLUDED.progress`
		if _, err := r.pool.Exec(ctx, q,
			turnID, c.Position[0], c.Position[1], c.Progress,
		); err != nil {
			return fmt.Errorf("InsertConstructions[%d]: %w", i, err)
		}
	}
	return nil
}

// InsertMountains replaces mountain coordinates for a round.
// Deletes old mountains and inserts current ones — handles earthquake landscape changes.
func (r *Repo) InsertMountains(ctx context.Context, roundID int, mountains []api.Coord) error {
	_, _ = r.pool.Exec(ctx, `DELETE FROM mountains WHERE round_id = $1`, roundID)
	for i, m := range mountains {
		const q = `
			INSERT INTO mountains (round_id, x, y)
			VALUES ($1, $2, $3)
			ON CONFLICT DO NOTHING`
		if _, err := r.pool.Exec(ctx, q, roundID, m[0], m[1]); err != nil {
			return fmt.Errorf("InsertMountains[%d]: %w", i, err)
		}
	}
	return nil
}

// InsertMeteo inserts meteo forecast rows for a turn.
// All pointer fields are handled as SQL NULLs when nil.
func (r *Repo) InsertMeteo(ctx context.Context, turnID int, forecasts []api.MeteoForecast) error {
	for i := range forecasts {
		f := &forecasts[i]

		// Expand nullable pointer fields into individual *int / *bool values.
		var posX, posY, nextX, nextY *int
		if f.Position != nil {
			x, y := f.Position[0], f.Position[1]
			posX, posY = &x, &y
		}
		if f.NextPosition != nil {
			x, y := f.NextPosition[0], f.NextPosition[1]
			nextX, nextY = &x, &y
		}

		const q = `
			INSERT INTO meteo_events
				(turn_id, kind, turns_until, storm_id, forming,
				 pos_x, pos_y, next_pos_x, next_pos_y, radius)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`
		if _, err := r.pool.Exec(ctx, q,
			turnID,
			f.Kind,
			f.TurnsUntil, // *int — pgx handles nil as NULL
			f.ID,
			f.Forming, // *bool — pgx handles nil as NULL
			posX, posY,
			nextX, nextY,
			f.Radius, // *int — pgx handles nil as NULL
		); err != nil {
			return fmt.Errorf("InsertMeteo[%d]: %w", i, err)
		}
	}
	return nil
}

// InsertUpgrades persists upgrade state for a turn, writing to both the
// upgrades and upgrade_tiers tables.
func (r *Repo) InsertUpgrades(ctx context.Context, turnID int, ups api.UpgradesState) error {
	const q = `
		INSERT INTO upgrades (turn_id, points, interval, turns_until, max_points)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (turn_id) DO UPDATE
			SET points     = EXCLUDED.points,
			    interval   = EXCLUDED.interval,
			    turns_until = EXCLUDED.turns_until,
			    max_points  = EXCLUDED.max_points`
	if _, err := r.pool.Exec(ctx, q,
		turnID, ups.Points, ups.IntervalTurns, ups.TurnsUntilPoints, ups.MaxPoints,
	); err != nil {
		return fmt.Errorf("InsertUpgrades: %w", err)
	}

	for i := range ups.Tiers {
		t := &ups.Tiers[i]
		const qt = `
			INSERT INTO upgrade_tiers (turn_id, name, current, max_level)
			VALUES ($1, $2, $3, $4)
			ON CONFLICT (turn_id, name) DO UPDATE
				SET current   = EXCLUDED.current,
				    max_level = EXCLUDED.max_level`
		if _, err := r.pool.Exec(ctx, qt, turnID, t.Name, t.Current, t.Max); err != nil {
			return fmt.Errorf("InsertUpgrades tier[%d]: %w", i, err)
		}
	}
	return nil
}

// InsertScoreLog appends one row to the score_log table.
func (r *Repo) InsertScoreLog(
	ctx context.Context,
	roundID, turnNo, plantations, scoreDelta, enemies, beavers int,
	storm, earthquake bool,
) error {
	const q = `
		INSERT INTO score_log
			(round_id, turn_no, my_plantations, score_delta,
			 enemies_visible, beavers_visible, storm_active, earthquake)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (round_id, turn_no) DO UPDATE
			SET my_plantations  = EXCLUDED.my_plantations,
			    score_delta      = EXCLUDED.score_delta,
			    enemies_visible  = EXCLUDED.enemies_visible,
			    beavers_visible  = EXCLUDED.beavers_visible,
			    storm_active     = EXCLUDED.storm_active,
			    earthquake       = EXCLUDED.earthquake`
	if _, err := r.pool.Exec(ctx, q,
		roundID, turnNo, plantations, scoreDelta, enemies, beavers, storm, earthquake,
	); err != nil {
		return fmt.Errorf("InsertScoreLog: %w", err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Query methods (dashboard)
// ---------------------------------------------------------------------------

// GetRounds returns a summary of all recorded rounds ordered by round_num.
func (r *Repo) GetRounds(ctx context.Context) ([]RoundSummary, error) {
	const q = `
		SELECT id, round_num,
		       COALESCE((SELECT COUNT(*) FROM turns WHERE round_id = rounds.id), 0) AS total_turns
		FROM rounds
		ORDER BY round_num`
	rows, err := r.pool.Query(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("GetRounds: %w", err)
	}
	defer rows.Close()

	var out []RoundSummary
	for rows.Next() {
		var s RoundSummary
		if err := rows.Scan(&s.ID, &s.RoundNum, &s.TotalTurns); err != nil {
			return nil, fmt.Errorf("GetRounds scan: %w", err)
		}
		out = append(out, s)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("GetRounds rows: %w", err)
	}
	return out, nil
}

// GetScoreHistory returns all score_log rows for a given round, ordered by turn.
func (r *Repo) GetScoreHistory(ctx context.Context, roundID int) ([]TurnSnapshot, error) {
	const q = `
		SELECT turn_no, my_plantations, score_delta, enemies_visible, beavers_visible
		FROM score_log
		WHERE round_id = $1
		ORDER BY turn_no`
	rows, err := r.pool.Query(ctx, q, roundID)
	if err != nil {
		return nil, fmt.Errorf("GetScoreHistory: %w", err)
	}
	defer rows.Close()

	var out []TurnSnapshot
	for rows.Next() {
		var s TurnSnapshot
		if err := rows.Scan(&s.TurnNo, &s.Plantations, &s.ScoreDelta, &s.Enemies, &s.Beavers); err != nil {
			return nil, fmt.Errorf("GetScoreHistory scan: %w", err)
		}
		out = append(out, s)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("GetScoreHistory rows: %w", err)
	}
	return out, nil
}

// GetMountains returns all mountain coordinates recorded for a round.
func (r *Repo) GetMountains(ctx context.Context, roundID int) ([]CoordData, error) {
	const q = `SELECT x, y FROM mountains WHERE round_id = $1 ORDER BY x, y`
	rows, err := r.pool.Query(ctx, q, roundID)
	if err != nil {
		return nil, fmt.Errorf("GetMountains: %w", err)
	}
	defer rows.Close()

	var out []CoordData
	for rows.Next() {
		var c CoordData
		if err := rows.Scan(&c.X, &c.Y); err != nil {
			return nil, fmt.Errorf("GetMountains scan: %w", err)
		}
		out = append(out, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("GetMountains rows: %w", err)
	}
	return out, nil
}

// TurnInfo is a lightweight turn record for the turn list slider.
type TurnInfo struct {
	TurnNo int `json:"turnNo"`
	TurnID int `json:"turnId"`
}

// PlantationData represents a plantation on the map.
type PlantationData struct {
	ID         string `json:"id"`
	X          int    `json:"x"`
	Y          int    `json:"y"`
	HP         int    `json:"hp"`
	IsMain     bool   `json:"isMain"`
	IsIsolated bool   `json:"isIsolated"`
}

// EnemyData represents an enemy plantation on the map.
type EnemyData struct {
	ID string `json:"id"`
	X  int    `json:"x"`
	Y  int    `json:"y"`
	HP int    `json:"hp"`
}

// BeaverData represents a beaver lair on the map.
type BeaverData struct {
	ID string `json:"id"`
	X  int    `json:"x"`
	Y  int    `json:"y"`
	HP int    `json:"hp"`
}

// ConstructionData represents an active construction.
type ConstructionData struct {
	X        int `json:"x"`
	Y        int `json:"y"`
	Progress int `json:"progress"`
}

// CellData represents a terraformed cell.
type CellData struct {
	X               int `json:"x"`
	Y               int `json:"y"`
	Progress        int `json:"progress"`
	TurnsUntilDeg   int `json:"turnsUntilDeg"`
}

// MeteoData represents a weather event.
type MeteoData struct {
	Kind      string `json:"kind"`
	TurnsUntil *int  `json:"turnsUntil"`
	StormID   string `json:"stormId,omitempty"`
	Forming   *bool  `json:"forming,omitempty"`
	PosX      *int   `json:"posX,omitempty"`
	PosY      *int   `json:"posY,omitempty"`
	NextPosX  *int   `json:"nextPosX,omitempty"`
	NextPosY  *int   `json:"nextPosY,omitempty"`
	Radius    *int   `json:"radius,omitempty"`
}

// UpgradeData is one upgrade tier for the dashboard.
type UpgradeData struct {
	Name    string `json:"name"`
	Current int    `json:"current"`
	Max     int    `json:"max"`
}

// TurnData is the full snapshot of a single turn for the dashboard map.
type TurnData struct {
	TurnNo        int                `json:"turnNo"`
	MapWidth      int                `json:"mapWidth"`
	MapHeight     int                `json:"mapHeight"`
	Plantations   []PlantationData   `json:"plantations"`
	Enemies       []EnemyData        `json:"enemies"`
	Beavers       []BeaverData       `json:"beavers"`
	Constructions []ConstructionData `json:"constructions"`
	Cells         []CellData         `json:"cells"`
	Meteo         []MeteoData        `json:"meteo"`
	Upgrades      []UpgradeData      `json:"upgrades"`
	Actions       []ActionData       `json:"actions"`
	Mountains     []CoordData        `json:"mountains"`
}

// GetTurnList returns all turn numbers for a given round.
func (r *Repo) GetTurnList(ctx context.Context, roundID int) ([]TurnInfo, error) {
	const q = `SELECT turn_no, id FROM turns WHERE round_id = $1 ORDER BY turn_no`
	rows, err := r.pool.Query(ctx, q, roundID)
	if err != nil {
		return nil, fmt.Errorf("GetTurnList: %w", err)
	}
	defer rows.Close()

	var out []TurnInfo
	for rows.Next() {
		var t TurnInfo
		if err := rows.Scan(&t.TurnNo, &t.TurnID); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// GetTurnData returns the full snapshot for a specific turn.
func (r *Repo) GetTurnData(ctx context.Context, roundID, turnNo int) (*TurnData, error) {
	// Get turn metadata.
	var turnID, mapW, mapH int
	err := r.pool.QueryRow(ctx,
		`SELECT id, map_width, map_height FROM turns WHERE round_id = $1 AND turn_no = $2`,
		roundID, turnNo).Scan(&turnID, &mapW, &mapH)
	if err != nil {
		return nil, nil // turn not found — not an error, just no data yet
	}

	td := &TurnData{TurnNo: turnNo, MapWidth: mapW, MapHeight: mapH}

	// Plantations
	{
		rows, err := r.pool.Query(ctx,
			`SELECT plantation_id, x, y, hp, is_main, is_isolated FROM plantations WHERE turn_id = $1`, turnID)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		for rows.Next() {
			var p PlantationData
			if err := rows.Scan(&p.ID, &p.X, &p.Y, &p.HP, &p.IsMain, &p.IsIsolated); err != nil {
				return nil, err
			}
			td.Plantations = append(td.Plantations, p)
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}

	// Enemies
	{
		rows, err := r.pool.Query(ctx,
			`SELECT enemy_id, x, y, hp FROM enemies WHERE turn_id = $1`, turnID)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		for rows.Next() {
			var e EnemyData
			if err := rows.Scan(&e.ID, &e.X, &e.Y, &e.HP); err != nil {
				return nil, err
			}
			td.Enemies = append(td.Enemies, e)
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}

	// Beavers
	{
		rows, err := r.pool.Query(ctx,
			`SELECT beaver_id, x, y, hp FROM beavers WHERE turn_id = $1`, turnID)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		for rows.Next() {
			var b BeaverData
			if err := rows.Scan(&b.ID, &b.X, &b.Y, &b.HP); err != nil {
				return nil, err
			}
			td.Beavers = append(td.Beavers, b)
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}

	// Constructions
	{
		rows, err := r.pool.Query(ctx,
			`SELECT x, y, progress FROM constructions WHERE turn_id = $1`, turnID)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		for rows.Next() {
			var c ConstructionData
			if err := rows.Scan(&c.X, &c.Y, &c.Progress); err != nil {
				return nil, err
			}
			td.Constructions = append(td.Constructions, c)
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}

	// Cells
	{
		rows, err := r.pool.Query(ctx,
			`SELECT x, y, progress, turns_until_deg FROM cells WHERE turn_id = $1`, turnID)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		for rows.Next() {
			var c CellData
			if err := rows.Scan(&c.X, &c.Y, &c.Progress, &c.TurnsUntilDeg); err != nil {
				return nil, err
			}
			td.Cells = append(td.Cells, c)
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}

	// Meteo
	{
		rows, err := r.pool.Query(ctx,
			`SELECT kind, turns_until, storm_id, forming, pos_x, pos_y, next_pos_x, next_pos_y, radius
			 FROM meteo_events WHERE turn_id = $1`, turnID)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		for rows.Next() {
			var m MeteoData
			var stormID *string
			if err := rows.Scan(&m.Kind, &m.TurnsUntil, &stormID, &m.Forming,
				&m.PosX, &m.PosY, &m.NextPosX, &m.NextPosY, &m.Radius); err != nil {
				return nil, err
			}
			if stormID != nil {
				m.StormID = *stormID
			}
			td.Meteo = append(td.Meteo, m)
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}

	// Upgrades
	{
		rows, err := r.pool.Query(ctx,
			`SELECT name, current, max_level FROM upgrade_tiers WHERE turn_id = $1 ORDER BY name`, turnID)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		for rows.Next() {
			var u UpgradeData
			if err := rows.Scan(&u.Name, &u.Current, &u.Max); err != nil {
				return nil, err
			}
			td.Upgrades = append(td.Upgrades, u)
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}

	// Actions
	{
		actions, err := r.GetActions(ctx, roundID, turnNo)
		if err == nil {
			td.Actions = actions
		}
	}

	// Mountains (from round — these are accumulated, but best we have)
	{
		mountains, err := r.GetMountains(ctx, roundID)
		if err == nil {
			td.Mountains = mountains
		}
	}

	return td, nil
}

// RoundStats holds aggregated statistics for a round.
type RoundStats struct {
	TotalTurns       int `json:"totalTurns"`
	TotalScore       int `json:"totalScore"`
	MaxPlantations   int `json:"maxPlantations"`
	TotalEnemiesSeen int `json:"totalEnemiesSeen"`
	TotalBeaversSeen int `json:"totalBeaversSeen"`
	StormTurns       int `json:"stormTurns"`
	Earthquakes      int `json:"earthquakes"`
}

// GetRoundStats returns aggregated statistics for a round.
func (r *Repo) GetRoundStats(ctx context.Context, roundID int) (*RoundStats, error) {
	var s RoundStats
	err := r.pool.QueryRow(ctx, `
		SELECT
			COUNT(*),
			COALESCE(SUM(score_delta), 0),
			COALESCE(MAX(my_plantations), 0),
			COALESCE(MAX(enemies_visible), 0),
			COALESCE(MAX(beavers_visible), 0),
			COALESCE(SUM(CASE WHEN storm_active THEN 1 ELSE 0 END), 0),
			COALESCE(SUM(CASE WHEN earthquake THEN 1 ELSE 0 END), 0)
		FROM score_log WHERE round_id = $1`, roundID).Scan(
		&s.TotalTurns, &s.TotalScore, &s.MaxPlantations,
		&s.TotalEnemiesSeen, &s.TotalBeaversSeen, &s.StormTurns, &s.Earthquakes)
	if err != nil {
		return nil, fmt.Errorf("GetRoundStats: %w", err)
	}
	return &s, nil
}

// InsertLogs batch-inserts game log messages.
func (r *Repo) InsertLogs(ctx context.Context, roundID int, logs []struct{ Time, Message string }) error {
	for _, l := range logs {
		_, err := r.pool.Exec(ctx,
			`INSERT INTO game_logs (round_id, log_time, message) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`,
			roundID, l.Time, l.Message)
		if err != nil {
			return err
		}
	}
	return nil
}

// GameLogEntry is a log row for the dashboard.
type GameLogEntry struct {
	Time    string `json:"time"`
	Message string `json:"message"`
}

// GetGameLogs returns logs for a round, latest first.
func (r *Repo) GetGameLogs(ctx context.Context, roundID int, limit int) ([]GameLogEntry, error) {
	if limit <= 0 {
		limit = 100
	}
	rows, err := r.pool.Query(ctx,
		`SELECT log_time, message FROM game_logs WHERE round_id = $1 ORDER BY id DESC LIMIT $2`,
		roundID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []GameLogEntry
	for rows.Next() {
		var e GameLogEntry
		if err := rows.Scan(&e.Time, &e.Message); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

// GetLatestRound returns the latest round_num and its id, or 0,0 if none.
func (r *Repo) GetLatestRound(ctx context.Context) (roundNum int, roundID int, err error) {
	err = r.pool.QueryRow(ctx,
		`SELECT round_num, id FROM rounds ORDER BY round_num DESC LIMIT 1`).Scan(&roundNum, &roundID)
	if err != nil {
		return 0, 0, nil // no rows = fresh start
	}
	return roundNum, roundID, nil
}

// GetLatestTurnNo returns the max turn_no for a given round, or 0.
func (r *Repo) GetLatestTurnNo(ctx context.Context, roundID int) (int, error) {
	var turnNo int
	err := r.pool.QueryRow(ctx,
		`SELECT COALESCE(MAX(turn_no), 0) FROM turns WHERE round_id = $1`, roundID).Scan(&turnNo)
	return turnNo, err
}

// AccumulatedData holds all unique objects ever seen in a round up to a given turn.
type AccumulatedData struct {
	Cells   []CellData   `json:"cells"`
	Enemies []EnemyData  `json:"enemies"`
	Beavers []BeaverData `json:"beavers"`
}

// GetAccumulatedData returns the latest known state of all cells/enemies/beavers
// ever seen in the round, up to the given turn.
func (r *Repo) GetAccumulatedData(ctx context.Context, roundID, turnNo int) (*AccumulatedData, error) {
	ad := &AccumulatedData{}

	// All cells: latest progress for each (x,y) seen up to turnNo.
	{
		rows, err := r.pool.Query(ctx, `
			SELECT DISTINCT ON (c.x, c.y) c.x, c.y, c.progress, c.turns_until_deg
			FROM cells c
			JOIN turns t ON t.id = c.turn_id
			WHERE t.round_id = $1 AND t.turn_no <= $2
			ORDER BY c.x, c.y, t.turn_no DESC`, roundID, turnNo)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		for rows.Next() {
			var c CellData
			if err := rows.Scan(&c.X, &c.Y, &c.Progress, &c.TurnsUntilDeg); err != nil {
				return nil, err
			}
			ad.Cells = append(ad.Cells, c)
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}

	// All enemies: latest known position for each enemy_id seen up to turnNo.
	{
		rows, err := r.pool.Query(ctx, `
			SELECT DISTINCT ON (e.enemy_id) e.enemy_id, e.x, e.y, e.hp
			FROM enemies e
			JOIN turns t ON t.id = e.turn_id
			WHERE t.round_id = $1 AND t.turn_no <= $2
			ORDER BY e.enemy_id, t.turn_no DESC`, roundID, turnNo)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		for rows.Next() {
			var e EnemyData
			if err := rows.Scan(&e.ID, &e.X, &e.Y, &e.HP); err != nil {
				return nil, err
			}
			ad.Enemies = append(ad.Enemies, e)
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}

	// All beavers: latest known position for each beaver_id.
	{
		rows, err := r.pool.Query(ctx, `
			SELECT DISTINCT ON (b.beaver_id) b.beaver_id, b.x, b.y, b.hp
			FROM beavers b
			JOIN turns t ON t.id = b.turn_id
			WHERE t.round_id = $1 AND t.turn_no <= $2
			ORDER BY b.beaver_id, t.turn_no DESC`, roundID, turnNo)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		for rows.Next() {
			var b BeaverData
			if err := rows.Scan(&b.ID, &b.X, &b.Y, &b.HP); err != nil {
				return nil, err
			}
			ad.Beavers = append(ad.Beavers, b)
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}

	return ad, nil
}

// StrategyParam is a single config parameter.
type StrategyParam struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
}

// StrategyLogEntry is one analysis result.
type StrategyLogEntry struct {
	RoundID  int             `json:"roundId"`
	Analysis json.RawMessage `json:"analysis"`
	Changes  json.RawMessage `json:"changes"`
	OurScore int             `json:"ourScore"`
}

// GetStrategyConfig returns current strategy parameters.
func (r *Repo) GetStrategyConfig(ctx context.Context) ([]StrategyParam, error) {
	rows, err := r.pool.Query(ctx,
		`SELECT param_name, param_value FROM strategy_config
		 WHERE round_id = (SELECT MAX(round_id) FROM strategy_config)
		 ORDER BY param_name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []StrategyParam
	for rows.Next() {
		var p StrategyParam
		if err := rows.Scan(&p.Name, &p.Value); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// GetStrategyLogs returns analysis history.
func (r *Repo) GetStrategyLogs(ctx context.Context, limit int) ([]StrategyLogEntry, error) {
	if limit <= 0 {
		limit = 10
	}
	rows, err := r.pool.Query(ctx,
		`SELECT round_id, analysis, changes, COALESCE(our_score, 0)
		 FROM strategy_log ORDER BY id DESC LIMIT $1`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []StrategyLogEntry
	for rows.Next() {
		var e StrategyLogEntry
		if err := rows.Scan(&e.RoundID, &e.Analysis, &e.Changes, &e.OurScore); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

// ActionRecord is one plantation action for saving.
type ActionRecord struct {
	AuthorX int
	AuthorY int
	TargetX int
	TargetY int
	Action  string
	Reason  string
}

// InsertActions saves plantation actions for a turn.
func (r *Repo) InsertActions(ctx context.Context, roundID, turnNo int, actions []ActionRecord) error {
	for _, a := range actions {
		_, err := r.pool.Exec(ctx,
			`INSERT INTO plantation_actions (round_id, turn_no, author_x, author_y, target_x, target_y, action, reason)
			 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			 ON CONFLICT (round_id, turn_no, author_x, author_y) DO UPDATE
			 SET target_x = EXCLUDED.target_x, target_y = EXCLUDED.target_y, action = EXCLUDED.action, reason = EXCLUDED.reason`,
			roundID, turnNo, a.AuthorX, a.AuthorY, a.TargetX, a.TargetY, a.Action, a.Reason)
		if err != nil {
			return err
		}
	}
	return nil
}

// ActionData is one action for the dashboard.
type ActionData struct {
	AuthorX int    `json:"authorX"`
	AuthorY int    `json:"authorY"`
	TargetX int    `json:"targetX"`
	TargetY int    `json:"targetY"`
	Action  string `json:"action"`
	Reason  string `json:"reason"`
}

// GetActions returns actions for a specific turn.
func (r *Repo) GetActions(ctx context.Context, roundID, turnNo int) ([]ActionData, error) {
	rows, err := r.pool.Query(ctx,
		`SELECT author_x, author_y, target_x, target_y, action, COALESCE(reason,'')
		 FROM plantation_actions WHERE round_id = $1 AND turn_no = $2`,
		roundID, turnNo)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ActionData
	for rows.Next() {
		var a ActionData
		if err := rows.Scan(&a.AuthorX, &a.AuthorY, &a.TargetX, &a.TargetY, &a.Action, &a.Reason); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}
