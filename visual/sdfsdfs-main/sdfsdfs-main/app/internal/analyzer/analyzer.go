package analyzer

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type Analyzer struct {
	pool     *pgxpool.Pool
	statsURL string // e.g. https://games-test.datsteam.dev/stats/{game-id}
}

func New(pool *pgxpool.Pool, statsURL string) *Analyzer {
	return &Analyzer{pool: pool, statsURL: statsURL}
}

// AnalyzeRound runs full analysis after a round ends.
func (a *Analyzer) AnalyzeRound(ctx context.Context, roundID int) error {
	log.Printf("[analyzer] analyzing round %d", roundID)

	// 1. Analyze our performance from DB.
	analysis, err := a.analyzeOurPerformance(ctx, roundID)
	if err != nil {
		return fmt.Errorf("analyze performance: %w", err)
	}

	// 2. Fetch external stats.
	top10, err := a.fetchExternalStats(ctx)
	if err != nil {
		log.Printf("[analyzer] external stats error (non-fatal): %v", err)
	}

	// 3. Calibrate parameters based on analysis.
	changes := a.calibrate(analysis)

	// 4. Save new config for next round.
	nextRoundID := roundID // will be updated when next round starts
	for param, value := range changes {
		_, err := a.pool.Exec(ctx,
			`INSERT INTO strategy_config (round_id, param_name, param_value)
			 VALUES ($1, $2, $3)
			 ON CONFLICT (round_id, param_name) DO UPDATE SET param_value = $3`,
			nextRoundID, param, value)
		if err != nil {
			log.Printf("[analyzer] save config error: %v", err)
		}
	}

	// 5. Save analysis log.
	analysisJSON, _ := json.Marshal(analysis)
	changesJSON, _ := json.Marshal(changes)
	top10JSON, _ := json.Marshal(top10)

	_, err = a.pool.Exec(ctx,
		`INSERT INTO strategy_log (round_id, analysis, changes, top10_scores, our_score)
		 VALUES ($1, $2, $3, $4, $5)`,
		roundID, analysisJSON, changesJSON, top10JSON, analysis.TotalScore)
	if err != nil {
		log.Printf("[analyzer] save log error: %v", err)
	}

	log.Printf("[analyzer] round %d: score=%d, deaths=%d, max_plants=%d, changes=%v",
		roundID, analysis.TotalScore, analysis.CUDeaths, analysis.MaxPlantations, changes)

	return nil
}

// ═══════════════════════════════════════════════════════════════
// OUR PERFORMANCE ANALYSIS
// ═══════════════════════════════════════════════════════════════

type RoundAnalysis struct {
	TotalScore          int     `json:"totalScore"`
	TotalTurns          int     `json:"totalTurns"`
	MaxPlantations      int     `json:"maxPlantations"`
	AvgPlantations      float64 `json:"avgPlantations"`
	CUDeaths            int     `json:"cuDeaths"`
	AvgLifeSpan         float64 `json:"avgLifeSpan"`
	MaxLifeSpan         int     `json:"maxLifeSpan"`
	BeaversKilled       int     `json:"beaversKilled"`
	BeaversSeen         int     `json:"beaversSeen"`
	EnemiesSeen         int     `json:"enemiesSeen"`
	ConstructionsBuilt  int     `json:"constructionsBuilt"`
	ConstructionsDegraded int   `json:"constructionsDegraded"`
	EarthquakeHits      int     `json:"earthquakeHits"`
	ScorePerTurn        float64 `json:"scorePerTurn"`
	WastedUpgradePoints int     `json:"wastedUpgradePoints"`
}

func (a *Analyzer) analyzeOurPerformance(ctx context.Context, roundID int) (*RoundAnalysis, error) {
	ra := &RoundAnalysis{}

	// Total score and turns.
	a.pool.QueryRow(ctx, `
		SELECT COALESCE(SUM(score_delta),0), COUNT(*), COALESCE(MAX(my_plantations),0),
		       COALESCE(AVG(my_plantations),0)
		FROM score_log WHERE round_id = $1`, roundID).Scan(
		&ra.TotalScore, &ra.TotalTurns, &ra.MaxPlantations, &ra.AvgPlantations)

	if ra.TotalTurns > 0 {
		ra.ScorePerTurn = float64(ra.TotalScore) / float64(ra.TotalTurns)
	}

	// CU deaths.
	a.pool.QueryRow(ctx, `
		WITH pc AS (
			SELECT t.turn_no,
				(SELECT count(*) FROM plantations p WHERE p.turn_id = t.id) as cnt
			FROM turns t WHERE t.round_id = $1
		),
		drops AS (
			SELECT turn_no, cnt, LAG(cnt) OVER (ORDER BY turn_no) as prev
			FROM pc
		)
		SELECT COUNT(*) FROM drops WHERE prev > 1 AND cnt <= 1`, roundID).Scan(&ra.CUDeaths)

	// Average and max life span.
	a.pool.QueryRow(ctx, `
		WITH pc AS (
			SELECT t.turn_no,
				(SELECT count(*) FROM plantations p WHERE p.turn_id = t.id) as cnt
			FROM turns t WHERE t.round_id = $1
		),
		drops AS (
			SELECT turn_no, cnt, LAG(cnt) OVER (ORDER BY turn_no) as prev
			FROM pc
		),
		deaths AS (
			SELECT turn_no as death,
				LAG(turn_no) OVER (ORDER BY turn_no) as prev_death
			FROM drops WHERE prev > 1 AND cnt <= 1
		)
		SELECT COALESCE(AVG(death - COALESCE(prev_death, 0)), 0),
		       COALESCE(MAX(death - COALESCE(prev_death, 0)), 0)
		FROM deaths`, roundID).Scan(&ra.AvgLifeSpan, &ra.MaxLifeSpan)

	// Beavers and enemies.
	a.pool.QueryRow(ctx, `
		SELECT COUNT(DISTINCT beaver_id) FROM beavers b
		JOIN turns t ON t.id = b.turn_id WHERE t.round_id = $1`, roundID).Scan(&ra.BeaversSeen)

	a.pool.QueryRow(ctx, `
		SELECT COUNT(DISTINCT enemy_id) FROM enemies e
		JOIN turns t ON t.id = e.turn_id WHERE t.round_id = $1`, roundID).Scan(&ra.EnemiesSeen)

	// Constructions.
	a.pool.QueryRow(ctx, `
		WITH ch AS (
			SELECT c.x, c.y, c.progress,
				LEAD(c.progress) OVER (PARTITION BY c.x, c.y ORDER BY t.turn_no) as next_p
			FROM constructions c JOIN turns t ON t.id=c.turn_id WHERE t.round_id=$1
		)
		SELECT COUNT(DISTINCT (x,y)) FILTER (WHERE progress >= 45 AND next_p IS NULL),
		       COUNT(*) FILTER (WHERE next_p IS NOT NULL AND next_p < progress)
		FROM ch`, roundID).Scan(&ra.ConstructionsBuilt, &ra.ConstructionsDegraded)

	// Earthquake hits.
	a.pool.QueryRow(ctx, `
		SELECT COUNT(*) FROM meteo_events m
		JOIN turns t ON t.id = m.turn_id
		WHERE t.round_id = $1 AND m.kind = 'earthquake' AND m.turns_until = 0`,
		roundID).Scan(&ra.EarthquakeHits)

	return ra, nil
}

// ═══════════════════════════════════════════════════════════════
// EXTERNAL STATS
// ═══════════════════════════════════════════════════════════════

type ExternalTeamScore struct {
	Team  string `json:"team"`
	Score int    `json:"score"`
}

func (a *Analyzer) fetchExternalStats(ctx context.Context) ([]ExternalTeamScore, error) {
	if a.statsURL == "" {
		return nil, nil
	}

	client := &http.Client{Timeout: 5 * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.statsURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	// Parse: {"realms": {"realm-name": {"players": [...]}}}
	var data struct {
		Realms map[string]struct {
			Players []struct {
				Player   string `json:"player"`
				Score    int    `json:"score"`
				Counters struct {
					TerraformingScore       int `json:"terraformingScore"`
					PlantationsDestroyed    int `json:"plantationsDestroyed"`
					BeaverLodgesDestroyed   int `json:"beaverLodgesDestroyed"`
					Deaths                  int `json:"deaths"`
					Kills                   int `json:"kills"`
				} `json:"counters"`
			} `json:"players"`
		} `json:"realms"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("parse stats: %w", err)
	}

	// Get the latest realm (highest number), take top 10.
	var latestRealm string
	for name := range data.Realms {
		if latestRealm == "" || name > latestRealm {
			latestRealm = name
		}
	}

	if latestRealm == "" {
		return nil, nil
	}

	realm := data.Realms[latestRealm]
	var top10 []ExternalTeamScore
	for i, p := range realm.Players {
		if i >= 10 {
			break
		}
		top10 = append(top10, ExternalTeamScore{Team: p.Player, Score: p.Score})

		// Save to DB for history.
		a.pool.Exec(ctx,
			`INSERT INTO external_stats (realm_name, team_name, score, deaths)
			 VALUES ($1, $2, $3, $4)
			 ON CONFLICT (realm_name, team_name) DO UPDATE
			 SET score = EXCLUDED.score, deaths = EXCLUDED.deaths, fetched_at = NOW()`,
			latestRealm, p.Player, p.Score, p.Counters.Deaths)
	}

	log.Printf("[analyzer] top-10 from %s: #1=%s(%d) #2=%s(%d) #3=%s(%d)",
		latestRealm,
		safeTeam(top10, 0), safeScore(top10, 0),
		safeTeam(top10, 1), safeScore(top10, 1),
		safeTeam(top10, 2), safeScore(top10, 2))

	return top10, nil
}

func safeTeam(s []ExternalTeamScore, i int) string {
	if i < len(s) { return s[i].Team }
	return "?"
}
func safeScore(s []ExternalTeamScore, i int) int {
	if i < len(s) { return s[i].Score }
	return 0
}

// ═══════════════════════════════════════════════════════════════
// CALIBRATION: adjust parameters based on analysis
// ═══════════════════════════════════════════════════════════════

func (a *Analyzer) calibrate(ra *RoundAnalysis) map[string]float64 {
	changes := make(map[string]float64)

	// Load previous config for comparison.
	// Start with defaults.
	cfg := map[string]float64{
		"cu_relocate_threshold":   85,
		"repair_hp_percent":       50,
		"beaver_min_plants":       2,
		"beaver_max_net_hp_loss":  3,
		"sabotage_hp_threshold":   30,
	}

	// Rule 1: CU relocate threshold. Never above 80 — higher = CU dies.
	if ra.CUDeaths > 10 {
		cfg["cu_relocate_threshold"] = 65
	} else if ra.CUDeaths > 5 {
		cfg["cu_relocate_threshold"] = 70
	} else {
		cfg["cu_relocate_threshold"] = 75 // max safe value
	}

	// Rule 2: If avg plantations < 1.5, we're dying too much → more aggressive repair.
	if ra.AvgPlantations < 1.5 {
		cfg["repair_hp_percent"] = 70 // repair earlier
	} else if ra.AvgPlantations > 3 {
		cfg["repair_hp_percent"] = 40 // can afford to be aggressive
	}

	// Rule 3: If we see beavers but never kill them, lower attack threshold.
	if ra.BeaversSeen > 0 && ra.BeaversKilled == 0 {
		cfg["beaver_max_net_hp_loss"] = 5 // more aggressive beaver attacks
		cfg["beaver_min_plants"] = 2
	}

	// Rule 4: If earthquakes hit 0 times, mark as low priority.
	if ra.EarthquakeHits == 0 {
		cfg["earthquake_priority"] = 0 // don't upgrade earthquake_mitigation early
	} else {
		cfg["earthquake_priority"] = 1
	}

	// Rule 5: If constructions degrade a lot, we're building wrong.
	if ra.ConstructionsDegraded > ra.ConstructionsBuilt {
		cfg["build_conservatively"] = 1 // fewer simultaneous constructions
	}

	// Rule 6: Score per turn analysis.
	if ra.ScorePerTurn < 60 {
		cfg["need_more_plants"] = 1 // score too low = not enough plantations
	}

	for k, v := range cfg {
		changes[k] = v
	}

	return changes
}

// ═══════════════════════════════════════════════════════════════
// CONFIG READER (used by strategy at runtime)
// ═══════════════════════════════════════════════════════════════

// GetConfig reads the latest strategy config from DB.
func GetConfig(ctx context.Context, pool *pgxpool.Pool) map[string]float64 {
	cfg := map[string]float64{
		"cu_relocate_threshold":  85,
		"repair_hp_percent":      50,
		"beaver_min_plants":      2,
		"beaver_max_net_hp_loss": 3,
		"sabotage_hp_threshold":  30,
		"earthquake_priority":    1,
		"build_conservatively":   0,
		"need_more_plants":       0,
	}

	rows, err := pool.Query(ctx,
		`SELECT param_name, param_value FROM strategy_config
		 WHERE round_id = (SELECT MAX(round_id) FROM strategy_config)`)
	if err != nil {
		return cfg
	}
	defer rows.Close()

	for rows.Next() {
		var name string
		var value float64
		if err := rows.Scan(&name, &value); err == nil {
			cfg[name] = value
		}
	}
	return cfg
}
