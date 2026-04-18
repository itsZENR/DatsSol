package strategy

import (
	"log"
	"math"

	"datssol/internal/api"
)

const forecastHorizon = 10

// Forecast predicts future state and sets strategic priorities.
type Forecast struct {
	// Per-plantation: estimated turn of death (0 = dies this turn).
	TurnsUntilDeath map[api.Coord]int
	DeathCause      map[api.Coord]string // "terraform", "hp", "beaver"

	// Cluster predictions.
	WaveDeathTurn int // relative turn when 30%+ plants die
	WaveDeathSize int
	PlantsDeadIn5  int
	PlantsDeadIn10 int

	// CU danger.
	CUTurnsLeft   int
	CUHasFreshAdj bool
	CUNeedsEscort bool

	// Pipeline.
	ActiveBuilds     int
	NeedReplacements int

	// Beaver threat: which beavers are damaging our plants.
	BeaverThreats []BeaverThreat

	// Territory analysis.
	TerritoryTooSmall bool // cluster boxed in, not enough room for plants
	ClusterWidth      int
	ClusterHeight     int

	// 10-turn plant count projection.
	ProjectedPlants [forecastHorizon]int
}

// BeaverThreat describes a beaver actively damaging our plants.
type BeaverThreat struct {
	Beaver          *api.Beaver
	PlantsInDanger  int // our plants in its AR=2
	TurnsToFirstKill int // when the first plant dies from this beaver
}

// RunForecast simulates 10 turns ahead, accounting for terraformation, beaver damage,
// storm damage, and earthquake probability.
func RunForecast(arena *api.ArenaResponse, myPos map[api.Coord]*api.Plantation, connected map[api.Coord]bool, cellProgress map[api.Coord]int) *Forecast {
	f := &Forecast{
		TurnsUntilDeath: make(map[api.Coord]int),
		DeathCause:      make(map[api.Coord]string),
	}

	// Build damage map: total damage per turn per position from all sources.
	dmgPerTurn := make(map[api.Coord]int)

	// Beaver damage.
	beaverDmg := 15 // base, TODO: account for mitigation upgrade
	for _, tier := range arena.PlantationUpgrades.Tiers {
		if tier.Name == "beaver_damage_mitigation" {
			beaverDmg = 15 - tier.Current*2
			break
		}
	}

	for i := range arena.Beavers {
		b := &arena.Beavers[i]
		plantsHit := 0
		minTurnsToKill := 999

		for pos, p := range myPos {
			if InRange(pos, b.Position, 2) {
				dmgPerTurn[pos] += beaverDmg
				plantsHit++
				turnsToKill := p.HP / beaverDmg
				if turnsToKill < minTurnsToKill {
					minTurnsToKill = turnsToKill
				}
			}
		}

		if plantsHit > 0 {
			f.BeaverThreats = append(f.BeaverThreats, BeaverThreat{
				Beaver:          b,
				PlantsInDanger:  plantsHit,
				TurnsToFirstKill: minTurnsToKill,
			})
		}
	}

	// Storm damage.
	for _, m := range arena.MeteoForecasts {
		if m.Kind != "sandstorm" || m.Position == nil || m.Radius == nil {
			continue
		}
		if m.Forming != nil && *m.Forming {
			continue // still forming, not active yet
		}
		stormX, stormY := (*m.Position)[0], (*m.Position)[1]
		radius := *m.Radius
		for pos := range myPos {
			dx := float64(pos[0] - stormX)
			dy := float64(pos[1] - stormY)
			dist := math.Sqrt(dx*dx + dy*dy)
			if dist <= float64(radius) {
				dmgPerTurn[pos] += 2 // sandstorm damage
			}
		}
	}

	// Earthquake: probabilistic. Average ~5% chance per turn, 10 damage.
	// Model as 0.5 expected damage per turn (10 * 0.05).
	// Too noisy to include per-plant, but factor into CU safety margin.

	// Simulate 10 turns for each plantation.
	type plantSim struct {
		hp       int
		progress int
		alive    bool
		deathT   int
		cause    string
	}
	sims := make(map[api.Coord]*plantSim)
	for pos, p := range myPos {
		sims[pos] = &plantSim{
			hp:       p.HP,
			progress: cellProgress[pos],
			alive:    true,
			deathT:   forecastHorizon,
			cause:    "",
		}
	}

	for t := 1; t <= forecastHorizon; t++ {
		aliveCount := 0
		for pos, s := range sims {
			if !s.alive {
				continue
			}

			// Terraformation.
			s.progress += 5
			if s.progress >= 100 {
				s.alive = false
				s.deathT = t
				s.cause = "terraform"
				continue
			}

			// Damage.
			s.hp -= dmgPerTurn[pos]
			if s.hp <= 0 {
				s.alive = false
				s.deathT = t
				if dmgPerTurn[pos] >= beaverDmg {
					s.cause = "beaver"
				} else {
					s.cause = "damage"
				}
				continue
			}

			aliveCount++
		}
		f.ProjectedPlants[t-1] = aliveCount
	}

	// Extract results from simulation.
	totalPlants := len(myPos)
	deathByTurn := make(map[int]int)

	for pos, s := range sims {
		f.TurnsUntilDeath[pos] = s.deathT
		f.DeathCause[pos] = s.cause

		if s.deathT <= 5 {
			f.PlantsDeadIn5++
		}
		if s.deathT <= 10 {
			f.PlantsDeadIn10++
		}
		deathByTurn[s.deathT]++
	}

	// CU analysis.
	for pos, p := range myPos {
		if !p.IsMain {
			continue
		}
		f.CUTurnsLeft = f.TurnsUntilDeath[pos]

		for _, adj := range Adjacents(pos) {
			if adjP := myPos[adj]; adjP != nil {
				adjProg := cellProgress[adj]
				if adjProg < 30 {
					f.CUHasFreshAdj = true
					break
				}
			}
		}
		// Also check constructions as future fresh neighbors.
		if !f.CUHasFreshAdj {
			for _, con := range arena.Construction {
				for _, adj := range Adjacents(pos) {
					if con.Position == adj {
						f.CUHasFreshAdj = true
						break
					}
				}
				if f.CUHasFreshAdj {
					break
				}
			}
		}

		if f.CUTurnsLeft <= 12 && !f.CUHasFreshAdj {
			f.CUNeedsEscort = true
		}
		break
	}

	// Wave death detection (sliding window of 3 turns).
	for t := 1; t <= forecastHorizon; t++ {
		waveSize := deathByTurn[t] + deathByTurn[t+1] + deathByTurn[t+2]
		if waveSize > f.WaveDeathSize {
			f.WaveDeathSize = waveSize
			f.WaveDeathTurn = t
		}
	}

	// Territory analysis: is the cluster boxed in?
	if totalPlants >= 5 {
		minX, maxX := 9999, -1
		minY, maxY := 9999, -1
		for pos := range myPos {
			if pos[0] < minX { minX = pos[0] }
			if pos[0] > maxX { maxX = pos[0] }
			if pos[1] < minY { minY = pos[1] }
			if pos[1] > maxY { maxY = pos[1] }
		}
		f.ClusterWidth = maxX - minX + 1
		f.ClusterHeight = maxY - minY + 1
		availableCells := f.ClusterWidth * f.ClusterHeight
		// Territory too small if plants occupy > 40% of bounding box.
		// This means cells are being recycled too fast.
		if totalPlants*100/max(availableCells, 1) > 40 {
			f.TerritoryTooSmall = true
			log.Printf("[forecast] TERRITORY SQUEEZE: %d plants in %dx%d box (%d cells, %d%% density). Need to break out!",
				totalPlants, f.ClusterWidth, f.ClusterHeight, availableCells, totalPlants*100/availableCells)
		}
	}

	// Pipeline.
	f.ActiveBuilds = len(arena.Construction)
	deficit := f.PlantsDeadIn10 - f.ActiveBuilds
	if deficit < 0 {
		deficit = 0
	}
	f.NeedReplacements = deficit

	// Logging.
	if f.CUNeedsEscort {
		log.Printf("[forecast] CU DANGER: dies in %d turns (%s), no fresh adj! Need escort.",
			f.CUTurnsLeft, sims[func() api.Coord {
				for pos, p := range myPos {
					if p.IsMain {
						return pos
					}
				}
				return api.Coord{}
			}()].cause)
	}

	for _, bt := range f.BeaverThreats {
		log.Printf("[forecast] BEAVER THREAT: B(%d,%d) hitting %d plants, first kill in %d turns",
			bt.Beaver.Position[0], bt.Beaver.Position[1], bt.PlantsInDanger, bt.TurnsToFirstKill)
	}

	if f.WaveDeathSize >= 3 && totalPlants > 0 && f.WaveDeathSize*100/totalPlants >= 30 {
		log.Printf("[forecast] WAVE DEATH in %d turns: %d/%d plants (%d%%)",
			f.WaveDeathTurn, f.WaveDeathSize, totalPlants, f.WaveDeathSize*100/totalPlants)
	}

	if totalPlants > 5 {
		log.Printf("[forecast] 10-turn projection: %d → [%d %d %d %d %d %d %d %d %d %d]",
			totalPlants,
			f.ProjectedPlants[0], f.ProjectedPlants[1], f.ProjectedPlants[2],
			f.ProjectedPlants[3], f.ProjectedPlants[4], f.ProjectedPlants[5],
			f.ProjectedPlants[6], f.ProjectedPlants[7], f.ProjectedPlants[8],
			f.ProjectedPlants[9])
	}

	return f
}
