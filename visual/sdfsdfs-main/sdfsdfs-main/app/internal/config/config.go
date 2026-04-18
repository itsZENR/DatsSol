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
