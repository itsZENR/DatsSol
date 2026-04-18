package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"

	"datssol/internal/analyzer"
	"datssol/internal/config"
	"datssol/internal/db"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	pool, err := db.NewPool(ctx, cfg.DBDSN)
	if err != nil {
		log.Fatal(err)
	}
	defer pool.Close()

	statsURL := os.Getenv("STATS_URL")
	az := analyzer.New(pool, statsURL)

	if len(os.Args) > 1 {
		roundID, _ := strconv.Atoi(os.Args[1])
		fmt.Printf("Analyzing round %d...\n", roundID)
		if err := az.AnalyzeRound(ctx, roundID); err != nil {
			log.Fatal(err)
		}
	} else {
		// Analyze all rounds that don't have a strategy_log entry.
		rows, err := pool.Query(ctx, `
			SELECT id FROM rounds
			WHERE id NOT IN (SELECT round_id FROM strategy_log)
			ORDER BY id`)
		if err != nil {
			log.Fatal(err)
		}
		defer rows.Close()
		for rows.Next() {
			var id int
			rows.Scan(&id)
			fmt.Printf("Analyzing round %d...\n", id)
			if err := az.AnalyzeRound(ctx, id); err != nil {
				log.Printf("round %d error: %v", id, err)
			}
		}
	}
	fmt.Println("Done.")
}
