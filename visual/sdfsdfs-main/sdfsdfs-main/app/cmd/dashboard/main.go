package main

import (
	"context"
	"log"

	"datssol/internal/config"
	"datssol/internal/dashboard"
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

	if err := db.RunMigrations(ctx, pool, "/app/migrations"); err != nil {
		log.Fatal("migration error: ", err)
	}

	repo := db.NewRepo(pool)
	log.Printf("Dashboard starting on :%s", cfg.DashboardPort)
	dashboard.Serve(cfg.DashboardPort, repo)
}
