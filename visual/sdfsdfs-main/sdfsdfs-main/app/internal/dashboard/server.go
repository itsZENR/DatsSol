package dashboard

import (
	"embed"
	"encoding/json"
	"io/fs"
	"log"
	"net/http"
	"strconv"

	"datssol/internal/db"
)

//go:embed static
var staticFiles embed.FS

func Serve(port string, repo *db.Repo) {
	mux := http.NewServeMux()

	// Serve static files with no-cache headers
	subFS, _ := fs.Sub(staticFiles, "static")
	fileServer := http.FileServer(http.FS(subFS))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		fileServer.ServeHTTP(w, r)
	})

	mux.HandleFunc("/api/rounds", func(w http.ResponseWriter, r *http.Request) {
		rounds, err := repo.GetRounds(r.Context())
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		jsonResponse(w, rounds)
	})

	mux.HandleFunc("/api/scores", func(w http.ResponseWriter, r *http.Request) {
		roundID := intParam(r, "round_id", 1)
		scores, err := repo.GetScoreHistory(r.Context(), roundID)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		jsonResponse(w, scores)
	})

	mux.HandleFunc("/api/map", func(w http.ResponseWriter, r *http.Request) {
		roundID := intParam(r, "round_id", 1)
		mountains, err := repo.GetMountains(r.Context(), roundID)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		jsonResponse(w, mountains)
	})

	mux.HandleFunc("/api/turns", func(w http.ResponseWriter, r *http.Request) {
		roundID := intParam(r, "round_id", 1)
		turns, err := repo.GetTurnList(r.Context(), roundID)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		jsonResponse(w, turns)
	})

	mux.HandleFunc("/api/turn", func(w http.ResponseWriter, r *http.Request) {
		roundID := intParam(r, "round_id", 1)
		turnNo := intParam(r, "turn_no", 1)
		data, err := repo.GetTurnData(r.Context(), roundID, turnNo)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		jsonResponse(w, data)
	})

	mux.HandleFunc("/api/round_stats", func(w http.ResponseWriter, r *http.Request) {
		roundID := intParam(r, "round_id", 1)
		stats, err := repo.GetRoundStats(r.Context(), roundID)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		jsonResponse(w, stats)
	})

	mux.HandleFunc("/api/accumulated", func(w http.ResponseWriter, r *http.Request) {
		roundID := intParam(r, "round_id", 1)
		turnNo := intParam(r, "turn_no", 1)
		data, err := repo.GetAccumulatedData(r.Context(), roundID, turnNo)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		jsonResponse(w, data)
	})

	// Game logs from DB (collected periodically by collector)
	mux.HandleFunc("/api/game_logs", func(w http.ResponseWriter, r *http.Request) {
		roundID := intParam(r, "round_id", 0)
		if roundID == 0 {
			// Use latest round
			rounds, _ := repo.GetRounds(r.Context())
			if len(rounds) > 0 {
				roundID = rounds[len(rounds)-1].RoundNum
			} else {
				roundID = 1
			}
		}
		logs, err := repo.GetGameLogs(r.Context(), roundID, 100)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		jsonResponse(w, logs)
	})

	mux.HandleFunc("/api/strategy_config", func(w http.ResponseWriter, r *http.Request) {
		params, err := repo.GetStrategyConfig(r.Context())
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		jsonResponse(w, params)
	})

	mux.HandleFunc("/api/strategy_logs", func(w http.ResponseWriter, r *http.Request) {
		logs, err := repo.GetStrategyLogs(r.Context(), 10)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		jsonResponse(w, logs)
	})

	log.Printf("Dashboard listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}

func intParam(r *http.Request, key string, fallback int) int {
	v, err := strconv.Atoi(r.URL.Query().Get(key))
	if err != nil || v == 0 {
		return fallback
	}
	return v
}

func jsonResponse(w http.ResponseWriter, data any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}
