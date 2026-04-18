package strategy

import (
	"log"
	"sort"

	"datssol/internal/api"
)

// beaverState tracks a beaver across turns for last-hit strategy.
type beaverState struct {
	LastHP          int // HP on previous turn
	HPDelta         int // HP change per turn (negative = being attacked)
	EnemiesNearby   int // enemy plantations in AR=2 of beaver
	TurnsTracked    int
	EnemyDamageRate int // estimated enemy damage per turn
	PredictedDeath  int // estimated turn when beaver dies (0 = unknown)
}

type Conveyor struct {
	mapWidth    int
	mapHeight   int
	beaverTrack map[string]*beaverState
	turnNo      int

	// Build target memory: stick with a target until it's done or impossible.
	lastBuildTarget    *api.Coord
	lastBuildTargetAge int

	// ActionReasons: author coord → reason string. Populated during Decide().
	ActionReasons map[api.Coord]string

	// Per-turn state for adaptive algorithms.
	turnConnected int // connected plants count this turn

	// Adaptive parameters (loaded from DB).
	cuRelocateThreshold  int     // terraformation % to trigger relocate (default 85)
	repairHPPercent      float64 // repair when HP < this % of MHP (default 50)
	beaverMinPlants      int     // min plantations to attack beaver (default 2)
	beaverMaxNetHPLoss   int     // max net HP loss/turn near beaver (default 3)
	sabotageHPThreshold  int     // attack enemies below this HP (default 30)
}

func NewConveyor() *Conveyor {
	return &Conveyor{
		beaverTrack:         make(map[string]*beaverState),
		cuRelocateThreshold: 85,
		repairHPPercent:     50,
		beaverMinPlants:     2,
		beaverMaxNetHPLoss:  3,
		sabotageHPThreshold: 30,
	}
}

// ApplyConfig updates adaptive parameters from analyzer.
func (c *Conveyor) ApplyConfig(cfg map[string]float64) {
	if v, ok := cfg["cu_relocate_threshold"]; ok {
		c.cuRelocateThreshold = int(v)
	}
	if v, ok := cfg["repair_hp_percent"]; ok {
		c.repairHPPercent = v
	}
	if v, ok := cfg["beaver_min_plants"]; ok {
		c.beaverMinPlants = int(v)
	}
	if v, ok := cfg["beaver_max_net_hp_loss"]; ok {
		c.beaverMaxNetHPLoss = int(v)
	}
	if v, ok := cfg["sabotage_hp_threshold"]; ok {
		c.sabotageHPThreshold = int(v)
	}
	log.Printf("[strategy] config: relocate=%d%%, repair=%0.f%%, beaver_min=%d, beaver_loss=%d, sabotage_hp=%d",
		c.cuRelocateThreshold, c.repairHPPercent, c.beaverMinPlants, c.beaverMaxNetHPLoss, c.sabotageHPThreshold)
}

func (c *Conveyor) Decide(arena *api.ArenaResponse) *api.CommandRequest {
	c.mapWidth = arena.Size[0]
	c.mapHeight = arena.Size[1]
	c.turnNo = arena.TurnNo
	c.ActionReasons = make(map[api.Coord]string)

	cmd := &api.CommandRequest{}

	if len(arena.Plantations) == 0 {
		return cmd
	}

	// Build lookup maps.
	myPositions := make(map[api.Coord]*api.Plantation)
	for i := range arena.Plantations {
		p := &arena.Plantations[i]
		myPositions[p.Position] = p
	}

	// Clear sticky build target if:
	// - plantation was built there (construction done)
	// - target hasn't produced a construction in 2 turns (server rejecting it)
	if c.lastBuildTarget != nil {
		pos := *c.lastBuildTarget
		if myPositions[pos] != nil {
			c.lastBuildTarget = nil
			c.lastBuildTargetAge = 0
		} else {
			// Check if construction exists at target.
			hasConstruction := false
			for _, con := range arena.Construction {
				if con.Position == pos {
					hasConstruction = true
					break
				}
			}
			if hasConstruction {
				c.lastBuildTargetAge = 0 // construction exists, keep going
			} else {
				c.lastBuildTargetAge++
				if c.lastBuildTargetAge >= 2 {
					// Tried for 2 turns, no construction appeared — target is invalid.
					log.Printf("[strategy] abandoning build target %v after %d failed turns", pos, c.lastBuildTargetAge)
					c.lastBuildTarget = nil
					c.lastBuildTargetAge = 0
				}
			}
		}
	}

	mountains := make(map[api.Coord]bool)
	for _, m := range arena.Mountains {
		mountains[m] = true
	}

	beaverMap := make(map[api.Coord]*api.Beaver)
	// Beaver proximity: Chebyshev distance to nearest beaver.
	// Used for gradient penalty in build scoring.
	beaverPositions := make([]api.Coord, 0, len(arena.Beavers))
	for i := range arena.Beavers {
		b := &arena.Beavers[i]
		beaverMap[b.Position] = b
		beaverPositions = append(beaverPositions, b.Position)
	}

	enemyMap := make(map[api.Coord]*api.EnemyPlantation)
	for i := range arena.Enemy {
		e := &arena.Enemy[i]
		enemyMap[e.Position] = e
	}

	constructing := make(map[api.Coord]bool)
	for _, con := range arena.Construction {
		constructing[con.Position] = true
	}

	connected := ConnectedToMain(arena.Plantations)

	// Get upgrade levels.
	beaverMitigation := getTierLevel(arena.PlantationUpgrades, "beaver_damage_mitigation")
	repairPower := 5 + getTierLevel(arena.PlantationUpgrades, "repair_power")
	mhp := 50 + getTierLevel(arena.PlantationUpgrades, "max_hp")*10

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

	// Build cell progress and degradation lookups.
	cellProgress := make(map[api.Coord]int)
	cellDegTurns := make(map[api.Coord]int) // turns until degradation starts
	for _, cell := range arena.Cells {
		cellProgress[cell.Position] = cell.TerraformationProgress
		cellDegTurns[cell.Position] = cell.TurnsUntilDegradation
	}

	// Calculate incoming damage per position.
	incomingDamage := c.calcIncomingDamage(arena, beaverMitigation)

	// Evaluate which plantations actually need repair (economic model).
	repairCandidates := c.evaluateRepairNeed(arena, connected, cellProgress, incomingDamage, repairPower, mhp)

	// Track assignments.
	assignedRepairs := make(map[api.Coord]bool)
	assignedBuilds := make(map[api.Coord]int) // builders per construction
	newBuildsThisTurn := 0
	newBeaverBuildsThisTurn := 0

	connectedPlants := make([]*api.Plantation, 0)
	var isolatedPositions []api.Coord
	for i := range arena.Plantations {
		p := &arena.Plantations[i]
		if p.IsIsolated {
			isolatedPositions = append(isolatedPositions, p.Position)
		}
		if !p.IsIsolated && connected[p.Position] {
			connectedPlants = append(connectedPlants, p)
		}
	}

	c.turnConnected = len(connectedPlants)

	// Run forecast — predict future events and set priorities.
	forecast := RunForecast(arena, myPositions, connected, cellProgress)

	// Evaluate beaver attacks (economic model + forecast).
	beaverPlans := c.evaluateBeaverAttacks(arena, myPositions, connected, beaverMitigation, repairPower, mhp, arena.ActionRange, forecast)
	assignedBeaverAttack := make(map[api.Coord]int)  // beaver pos → assigned attackers
	assignedBeaverRepair := make(map[api.Coord]int)  // beaver pos → assigned repairers

	// Count adjacent plantations + constructions near CU to detect urgency.
	var cuPlant *api.Plantation
	cuAdjCount := 0
	for i := range arena.Plantations {
		if arena.Plantations[i].IsMain {
			cuPlant = &arena.Plantations[i]
			break
		}
	}
	if cuPlant != nil {
		for _, adj := range Adjacents(cuPlant.Position) {
			if myPositions[adj] != nil {
				cuAdjCount++
			}
		}
		for _, con := range arena.Construction {
			for _, adj := range Adjacents(cuPlant.Position) {
				if con.Position == adj {
					cuAdjCount++
					break
				}
			}
		}
	}
	// P0 only when CU has ZERO adjacent plants/constructions — truly alone.
	// With 1+ neighbors, CU participates in normal P2/P3 flow to avoid splitting builders.
	cuNeedsBuild := cuAdjCount == 0 || (forecast.CUNeedsEscort && cuAdjCount < 2)

	for _, p := range connectedPlants {
		// Priority 0: CU survival — if CU has < 2 adjacent plantations/constructions,
		// CU must build nearby before anything else (except critical self-repair).
		if cuNeedsBuild && p.IsMain {
			if action := c.trySmartRepair(p, repairCandidates, assignedRepairs, arena.ActionRange, true); action != nil {
				cmd.Command = append(cmd.Command, *action)
				c.ActionReasons[p.Position] = "П0: ЦУ крит. ремонт (умирает)"
				continue
			}
			if action := c.tryContinueBuild(p, arena.Construction, assignedBuilds, arena.ActionRange); action != nil {
				cmd.Command = append(cmd.Command, *action)
				c.ActionReasons[p.Position] = "П0: ЦУ продолжает стройку (соседей: " + itoa(cuAdjCount) + ")"
				continue
			}
			plantLimit := 30 + getTierLevel(arena.PlantationUpgrades, "settlement_limit")
			totalFuturePlants := len(arena.Plantations) + len(arena.Construction) + newBuildsThisTurn
			if totalFuturePlants < plantLimit {
				if action := c.tryNewBuild(p, occupied, mountains, beaverPositions, enemyMap, myPositions, cellProgress, cellDegTurns, isolatedPositions, arena.ActionRange, nil); action != nil {
					cmd.Command = append(cmd.Command, *action)
					// Don't increment newBuildsThisTurn — P0 is survival, not expansion.
					// Other plants should still be allowed to start new builds.
					c.ActionReasons[p.Position] = "П0: ЦУ срочная стройка (соседей: " + itoa(cuAdjCount) + ")"
					log.Printf("[strategy] CU urgent build: only %d adjacent, starting build", cuAdjCount)
					continue
				}
			}
		}

		// Priority 1: Critical repair (ЦУ dying, chain link breaking).
		if action := c.trySmartRepair(p, repairCandidates, assignedRepairs, arena.ActionRange, true); action != nil {
			cmd.Command = append(cmd.Command, *action)
			c.ActionReasons[p.Position] = "П1: крит. ремонт (цепочка/ЦУ под угрозой)"
			continue
		}

		// Priority 1.5: Beaver attack.
		if action := c.tryBeaverAttackAction(p, beaverPlans, assignedBeaverAttack, assignedBeaverRepair, arena.ActionRange); action != nil {
			cmd.Command = append(cmd.Command, *action)
			c.ActionReasons[p.Position] = "П1.5: атака/ремонт бобра"
			continue
		}
		// Beaver build: only use the CLOSEST plant to the beaver.
		// Other plants continue with normal priorities (P2/P3).
		// Allow up to 3 beaver builders to rush constructions near beaver simultaneously.
		maxBeaverBuilders := 3
		if len(beaverPlans) > 0 && beaverPlans[0].needBuildCloser && !p.IsMain && newBeaverBuildsThisTurn < maxBeaverBuilders {
			bpos := beaverPlans[0].beaver.Position
			{
				plantLimit := 30 + getTierLevel(arena.PlantationUpgrades, "settlement_limit")
				totalFuturePlants := len(arena.Plantations) + len(arena.Construction) + newBuildsThisTurn
				if totalFuturePlants < plantLimit {
					if action := c.tryNewBuild(p, occupied, mountains, beaverPositions, enemyMap, myPositions, cellProgress, cellDegTurns, isolatedPositions, arena.ActionRange, &bpos); action != nil {
						cmd.Command = append(cmd.Command, *action)
						newBuildsThisTurn++
						newBeaverBuildsThisTurn++
						target := action.Path[2]
						c.ActionReasons[p.Position] = "П1.5: стройка к бобру B(" + itoa(bpos[0]) + "," + itoa(bpos[1]) + ") → (" + itoa(target[0]) + "," + itoa(target[1]) + ")"
						log.Printf("[beaver-build] plant (%d,%d) → (%d,%d) toward B(%d,%d) dist=%d",
							p.Position[0], p.Position[1], target[0], target[1], bpos[0], bpos[1],
							abs(target[0]-bpos[0])+abs(target[1]-bpos[1]))
						continue
					}
				}
			}
		}

		// Priority 2: Continue existing construction.
		if action := c.tryContinueBuild(p, arena.Construction, assignedBuilds, arena.ActionRange); action != nil {
			cmd.Command = append(cmd.Command, *action)
			target := action.Path[2]
			c.ActionReasons[p.Position] = "П2: продолжаю стройку " + itoa(target[0]) + "," + itoa(target[1])
			continue
		}

		// Priority 3: Start new build.
		// Adaptive: if no active constructions to continue, allow more new starts
		// so plants don't sit idle. With active constructions, be conservative.
		busyPlants := len(cmd.Command)
		freePlants := len(connectedPlants) - busyPlants
		activeConstructions := len(arena.Construction)
		maxNewBuilds := 1
		if freePlants > 0 {
			if activeConstructions == 0 {
				// No constructions — start as many as we can staff.
				maxNewBuilds = freePlants
			} else if len(connectedPlants) <= 4 {
				// Early growth phase: always allow new builds even with active constructions.
				// Need parallel constructions to outpace terraformation deaths.
				maxNewBuilds = freePlants
			} else {
				// Stable phase: be conservative with new starts.
				minBuildersPerConstruction := 3
				if len(beaverPlans) > 0 && beaverPlans[0].needBuildCloser {
					minBuildersPerConstruction = 4
				}
				maxNewBuilds = freePlants / minBuildersPerConstruction
			}
			if maxNewBuilds < 1 {
				maxNewBuilds = 1
			}
		}
		// Forecast: if many plants dying soon, boost construction rate.
		if forecast.NeedReplacements > maxNewBuilds {
			maxNewBuilds = forecast.NeedReplacements
			if maxNewBuilds > freePlants/2 {
				maxNewBuilds = freePlants / 2
			}
			if maxNewBuilds < 1 {
				maxNewBuilds = 1
			}
		}
		plantLimit := 30 + getTierLevel(arena.PlantationUpgrades, "settlement_limit")
		totalFuturePlants := len(arena.Plantations) + len(arena.Construction) + newBuildsThisTurn
		if newBuildsThisTurn < maxNewBuilds && totalFuturePlants < plantLimit {
			// If there's a beaver attack plan that needs building closer, pass target.
			var buildTargetBeaver *api.Coord
			for _, bp := range beaverPlans {
				if bp.needBuildCloser {
					pos := bp.beaver.Position
					buildTargetBeaver = &pos
					break
				}
			}
			if action := c.tryNewBuild(p, occupied, mountains, beaverPositions, enemyMap, myPositions, cellProgress, cellDegTurns, isolatedPositions, arena.ActionRange, buildTargetBeaver); action != nil {
				cmd.Command = append(cmd.Command, *action)
				newBuildsThisTurn++
				target := action.Path[2]
				c.ActionReasons[p.Position] = "П3: новая стройка " + itoa(target[0]) + "," + itoa(target[1]) + " (свободных=" + itoa(freePlants) + " макс=" + itoa(maxNewBuilds) + ")"
				continue
			}
		}

		// No non-critical repair — expansion gives more points than healing.
		// Critical repair (P1) already handles CU and chain-link survival.
	}

	return cmd
}

func getTierLevel(ups api.UpgradesState, name string) int {
	for _, t := range ups.Tiers {
		if t.Name == name {
			return t.Current
		}
	}
	return 0
}

// ═══════════════════════════════════════════════════════════════
// SMART BEAVER ATTACK (economic model)
// ═══════════════════════════════════════════════════════════════

// beaverAttackPlan describes a profitable beaver attack.
type beaverAttackPlan struct {
	beaver        *api.Beaver
	attackers     int  // how many plants should attack
	repairers     int  // how many plants should repair attackers
	turnsToKill   int  // estimated turns to kill
	pointsGained  int  // points from killing beaver
	pointsLost    int  // points lost from diverting plants from terraforming
	netProfit     int  // pointsGained - pointsLost
	needBuildCloser bool // true = no plants in AR yet, need to build toward beaver
}

// evaluateBeaverAttacks finds beavers that are profitable to attack.
// If plants aren't in AR yet but are close, returns a plan with needBuildCloser=true.
func (c *Conveyor) evaluateBeaverAttacks(arena *api.ArenaResponse, myPos map[api.Coord]*api.Plantation, connected map[api.Coord]bool, beaverMit, repairPower, mhp, ar int, forecast *Forecast) []beaverAttackPlan {
	if len(arena.Beavers) == 0 {
		return nil
	}

	connectedCount := 0
	for _, p := range arena.Plantations {
		if !p.IsIsolated && connected[p.Position] {
			connectedCount++
		}
	}

	// Beaver kill = 20,000 points. Even with 3 plants it's profitable.
	// Always be ready to attack — beavers block expansion and kill clusters.
	minPlantsForAttack := 3
	if connectedCount < minPlantsForAttack {
		return nil
	}

	beaverDmg := 15 - beaverMit*2
	beaverRegen := 5
	attackPower := 5
	scorePerPlantPerTurn := 50

	var plans []beaverAttackPlan

	for i := range arena.Beavers {
		b := &arena.Beavers[i]

		// Count plants in attack range (AR) and nearby (AR+3).
		inRange := 0
		nearby := 0 // within AR+3 but not in AR
		for _, p := range arena.Plantations {
			if p.IsIsolated || !connected[p.Position] {
				continue
			}
			if InRange(p.Position, b.Position, ar) {
				inRange++
			} else if InRange(p.Position, b.Position, ar+3) {
				nearby++
			}
		}

		if inRange == 0 && nearby == 0 {
			continue // too far, no chance
		}

		// Calculate beaver kill points (20x multiplier).
		beaverKillPoints := 20000
		if IsBoosted(b.Position) {
			beaverKillPoints = 30000
		}

		// If plants are nearby but not in range → plan "build closer".
		if inRange < 2 && nearby >= 2 {
			// Estimate: need ~10 turns to build 2 plants in AR, then 10 turns to kill.
			buildTurns := 10
			numAttackers := 3
			numRepairers := 1
			totalCommitted := numAttackers + numRepairers
			turnsToKill := (b.HP + (numAttackers*attackPower - beaverRegen) - 1) / (numAttackers*attackPower - beaverRegen)
			totalTurns := buildTurns + turnsToKill
			opportunityCost := totalCommitted * scorePerPlantPerTurn * totalTurns
			netProfit := beaverKillPoints - opportunityCost

			maxCommit := connectedCount * 3 / 5
			if maxCommit < 3 {
				maxCommit = 3
			}

			if netProfit > 0 && totalCommitted <= maxCommit && totalTurns <= 30 {
				plans = append(plans, beaverAttackPlan{
					beaver:          b,
					attackers:       numAttackers,
					repairers:       numRepairers,
					turnsToKill:     turnsToKill,
					pointsGained:    beaverKillPoints,
					pointsLost:      opportunityCost,
					netProfit:       netProfit,
					needBuildCloser: true,
				})
			}
			continue
		}

		if inRange < 2 {
			continue
		}

		// Plants already in range — find optimal attacker count.
		for numAttackers := 2; numAttackers <= inRange && numAttackers <= 5; numAttackers++ {
			netDmgPerTurn := numAttackers*attackPower - beaverRegen
			if netDmgPerTurn <= 0 {
				continue
			}

			turnsToKill := (b.HP + netDmgPerTurn - 1) / netDmgPerTurn

			needRepairPerTurn := beaverDmg - mhp/turnsToKill
			if needRepairPerTurn < 0 {
				needRepairPerTurn = 0
			}
			numRepairers := 0
			if needRepairPerTurn > 0 {
				totalRepairNeeded := needRepairPerTurn * numAttackers
				numRepairers = (totalRepairNeeded + repairPower - 1) / repairPower
			}

			totalCommitted := numAttackers + numRepairers

			maxCommit := connectedCount * 3 / 5
			if maxCommit < 3 {
				maxCommit = 3
			}
			if totalCommitted > maxCommit {
				continue
			}

			// Survival check: attackers don't need to survive the ENTIRE fight.
			// They deal damage while alive, then get replaced by new builds.
			// Just verify at least 1 attacker survives long enough to deal meaningful damage.
			totalRepair := numRepairers * repairPower
			hpLossPerTurn := beaverDmg - totalRepair/max(numAttackers, 1)
			attackerSurvivalTurns := mhp / max(hpLossPerTurn, 1)
			if hpLossPerTurn > 0 && attackerSurvivalTurns < 3 && numRepairers == 0 {
				// Dies too fast without repair — need at least 1 repairer.
				continue
			}

			opportunityCost := totalCommitted * scorePerPlantPerTurn * turnsToKill
			netProfit := beaverKillPoints - opportunityCost

			if netProfit <= 0 || turnsToKill > 30 {
				continue
			}

			plans = append(plans, beaverAttackPlan{
				beaver:       b,
				attackers:    numAttackers,
				repairers:    numRepairers,
				turnsToKill:  turnsToKill,
				pointsGained: beaverKillPoints,
				pointsLost:   opportunityCost,
				netProfit:    netProfit,
			})
		}
	}

	// Sort by net profit descending.
	sort.Slice(plans, func(i, j int) bool {
		return plans[i].netProfit > plans[j].netProfit
	})

	// Only attack 1 beaver at a time.
	if len(plans) > 1 {
		plans = plans[:1]
	}

	for _, p := range plans {
		if p.needBuildCloser {
			log.Printf("[beaver-attack] PLAN: build toward B(%d,%d) hp=%d, profit=%d (need %d att + %d rep)",
				p.beaver.Position[0], p.beaver.Position[1], p.beaver.HP,
				p.netProfit, p.attackers, p.repairers)
		} else {
			log.Printf("[beaver-attack] ATTACK: B(%d,%d) hp=%d: %d att + %d rep, kill in %d turns, profit=%d",
				p.beaver.Position[0], p.beaver.Position[1], p.beaver.HP,
				p.attackers, p.repairers, p.turnsToKill, p.netProfit)
		}
	}

	return plans
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// tryBeaverAttackAction assigns a plant to attack or repair for a profitable beaver plan.
func (c *Conveyor) tryBeaverAttackAction(author *api.Plantation, plans []beaverAttackPlan, assignedAttack, assignedRepair map[api.Coord]int, ar int) *api.PlantationAction {
	for _, plan := range plans {
		bPos := plan.beaver.Position

		// Don't assign CU to beaver combat.
		if author.IsMain {
			return nil
		}

		if !InRange(author.Position, bPos, ar) {
			continue
		}

		// Assign as attacker first, then repairer.
		if assignedAttack[bPos] < plan.attackers {
			assignedAttack[bPos]++
			return &api.PlantationAction{
				Path: []api.Coord{author.Position, author.Position, bPos},
			}
		}

		// Assign as repairer: repair the nearest attacker in beaver's AR=2 zone.
		if assignedRepair[bPos] < plan.repairers {
			// Find an ally in beaver's damage zone to repair.
			// Look for plants within AR=2 of beaver (they're taking damage).
			// We can't easily know which ones are attackers, so repair any nearby ally.
			for dx := -2; dx <= 2; dx++ {
				for dy := -2; dy <= 2; dy++ {
					repairTarget := api.Coord{bPos[0] + dx, bPos[1] + dy}
					if repairTarget == author.Position {
						continue
					}
					if InRange(author.Position, repairTarget, ar) {
						assignedRepair[bPos]++
						return &api.PlantationAction{
							Path: []api.Coord{author.Position, author.Position, repairTarget},
						}
					}
				}
			}
		}
	}
	return nil
}

func (c *Conveyor) countEnemiesNearBeavers(arena *api.ArenaResponse) {
	for i := range arena.Beavers {
		b := &arena.Beavers[i]
		st := c.beaverTrack[b.ID]
		if st == nil {
			continue
		}
		count := 0
		for _, e := range arena.Enemy {
			if InRange(e.Position, b.Position, 2) { // beaver AR=2
				count++
			}
		}
		st.EnemiesNearby = count
	}
}

type beaverDecision struct {
	beaver      *api.Beaver
	action      string // "WAIT", "ATTACK", "LAST_HIT", "SKIP"
	reason      string
	neededPlants int   // how many of our plantations should attack
}

func (c *Conveyor) decideBeaverActions(arena *api.ArenaResponse, myPos map[api.Coord]*api.Plantation, connected map[api.Coord]bool, beaverMit, repairPow, mhp int) []beaverDecision {
	beaverDmg := 15 - beaverMit*2

	var decisions []beaverDecision

	for i := range arena.Beavers {
		b := &arena.Beavers[i]
		st := c.beaverTrack[b.ID]
		if st == nil {
			continue
		}

		// Count our plantations that can attack this beaver (in our AR).
		ourInRange := 0
		// Count our plantations that can repair attackers (nearby but outside beaver AR=2).
		repairSupport := 0
		for _, p := range arena.Plantations {
			if p.IsIsolated || !connected[p.Position] {
				continue
			}
			if InRange(p.Position, b.Position, arena.ActionRange) {
				ourInRange++
			}
			// Count plants within AR=2 of attacker positions but outside beaver AR=2.
			bCheb := abs(p.Position[0]-b.Position[0])
			if abs(p.Position[1]-b.Position[1]) > bCheb {
				bCheb = abs(p.Position[1] - b.Position[1])
			}
			if bCheb > 2 && bCheb <= 5 {
				repairSupport++ // can repair attackers but won't be hit by beaver
			}
		}

		if ourInRange == 0 {
			continue
		}

		// Can we sustain? Need repair > beaver damage for each attacker.
		// With repairSupport repairers at RS each: total repair = repairSupport * repairPow
		canSustain := repairSupport*repairPow >= beaverDmg

		// Estimate enemy attacking force from HP delta.
		enemyPlants := 0
		if st.EnemyDamageRate > 0 {
			enemyPlants = st.EnemyDamageRate / 5 // each enemy plantation does BE=5
			if enemyPlants == 0 {
				enemyPlants = 1
			}
		}

		d := beaverDecision{beaver: b}

		// Case 1: Nobody else is attacking beaver.
		if enemyPlants == 0 && st.HPDelta >= 0 {
			// Net damage per turn = ourInRange * 5 (attack power) - 5 (beaver regen).
			// Only attack if we can actually kill the beaver in reasonable time.
			netDmgPerTurn := ourInRange*5 - 5
			turnsToKill := 0
			if netDmgPerTurn > 0 {
				turnsToKill = (b.HP + netDmgPerTurn - 1) / netDmgPerTurn
			}

			if !canSustain {
				d.action = "SKIP"
				d.reason = "not enough repair support (" + itoa(repairSupport) + " repairers, need " + itoa((beaverDmg+repairPow-1)/repairPow) + ")"
			} else if netDmgPerTurn <= 0 {
				d.action = "SKIP"
				d.reason = "net damage " + itoa(netDmgPerTurn) + "/turn, beaver regens faster than we attack"
			} else if turnsToKill > 15 {
				d.action = "SKIP"
				d.reason = "would take " + itoa(turnsToKill) + " turns to kill, not worth it"
			} else if ourInRange >= c.beaverMinPlants {
				d.action = "ATTACK"
				d.reason = "attack with " + itoa(ourInRange) + " in range, kill in ~" + itoa(turnsToKill) + " turns"
				d.neededPlants = ourInRange
			} else {
				d.action = "SKIP"
				d.reason = "need " + itoa(c.beaverMinPlants) + "+ plants in range, have " + itoa(ourInRange)
			}
			decisions = append(decisions, d)
			continue
		}

		// Case 2: Enemy is attacking beaver → LAST-HIT strategy.
		if enemyPlants > 0 {
			// We need MORE plantations than enemy to win last-hit.
			neededToWin := enemyPlants + 1

			if ourInRange >= neededToWin {
				// We have enough — but WHEN to strike?
				// If beaver will die soon (< 3 turns), attack NOW.
				turnsUntilDeath := 0
				if st.EnemyDamageRate > 5 {
					netEnemyDmg := st.EnemyDamageRate - 5
					turnsUntilDeath = b.HP / netEnemyDmg
				}

				if turnsUntilDeath <= 3 && turnsUntilDeath > 0 {
					d.action = "LAST_HIT"
					d.reason = "beaver dies in ~" + itoa(turnsUntilDeath) + " turns, we have " + itoa(ourInRange) + " vs enemy " + itoa(enemyPlants)
					d.neededPlants = ourInRange // all in!
				} else if turnsUntilDeath > 3 {
					d.action = "WAIT"
					d.reason = "beaver dies in ~" + itoa(turnsUntilDeath) + " turns, waiting"
				} else {
					d.action = "WAIT"
					d.reason = "enemy attacking but death not imminent"
				}
			} else {
				// Not enough plants yet — build more near beaver.
				d.action = "WAIT"
				d.reason = "need " + itoa(neededToWin) + " plants, have " + itoa(ourInRange)
			}
			decisions = append(decisions, d)
			continue
		}

		// Default: skip.
		d.action = "SKIP"
		d.reason = "no clear opportunity"
		decisions = append(decisions, d)
	}

	// Log decisions.
	for _, d := range decisions {
		if d.action != "SKIP" {
			log.Printf("[beaver] %s at %v HP=%d: %s (%s)",
				d.action, d.beaver.Position, d.beaver.HP, d.reason, d.beaver.ID)
		}
	}

	return decisions
}

func (c *Conveyor) tryBeaverAction(author *api.Plantation, decisions []beaverDecision, assigned map[api.Coord]int, ar int) *api.PlantationAction {
	for _, d := range decisions {
		if d.action != "ATTACK" && d.action != "LAST_HIT" {
			continue
		}
		if !InRange(author.Position, d.beaver.Position, ar) {
			continue
		}
		if assigned[d.beaver.Position] >= d.neededPlants {
			continue
		}
		assigned[d.beaver.Position]++
		return &api.PlantationAction{
			Path: []api.Coord{author.Position, author.Position, d.beaver.Position},
		}
	}
	return nil
}

// ═══════════════════════════════════════════════════════════════
// ENEMY EVALUATION & SABOTAGE
// ═══════════════════════════════════════════════════════════════

type enemyTarget struct {
	enemy    *api.EnemyPlantation
	priority int
	reason   string
}

func (c *Conveyor) evaluateEnemies(arena *api.ArenaResponse, myPos map[api.Coord]*api.Plantation, connected map[api.Coord]bool) []enemyTarget {
	if len(arena.Enemy) == 0 {
		return nil
	}

	var targets []enemyTarget

	for i := range arena.Enemy {
		e := &arena.Enemy[i]
		priority := 0
		reason := ""

		// Check if any of our plants can reach.
		canReach := false
		for _, p := range arena.Plantations {
			if p.IsIsolated || !connected[p.Position] {
				continue
			}
			if InRange(p.Position, e.Position, arena.ActionRange) {
				canReach = true
				break
			}
		}
		if !canReach {
			continue
		}

		// Weak target: low HP = easy kill = points.
		threshold := c.sabotageHPThreshold
		if e.HP <= threshold/3 {
			priority += 150
			reason = "almost dead"
		} else if e.HP <= threshold*2/3 {
			priority += 80
			reason = "weakened"
		} else if e.HP <= threshold {
			priority += 30
			reason = "hittable"
		}
		// Skip healthy enemies (HP > 40) — not worth the investment.
		if priority == 0 {
			continue
		}

		// Enemy near beaver? Sabotaging them helps our beaver strategy.
		for _, b := range arena.Beavers {
			if InRange(e.Position, b.Position, 3) {
				priority += 100
				reason += " + near beaver"
				break
			}
		}

		targets = append(targets, enemyTarget{e, priority, reason})
	}

	sort.Slice(targets, func(i, j int) bool {
		return targets[i].priority > targets[j].priority
	})

	// Log notable targets.
	for _, t := range targets {
		if t.priority >= 80 {
			log.Printf("[enemy] target at %v HP=%d priority=%d (%s)", t.enemy.Position, t.enemy.HP, t.priority, t.reason)
		}
	}

	return targets
}

func (c *Conveyor) trySabotageEnemy(author *api.Plantation, targets []enemyTarget, ar int) *api.PlantationAction {
	for _, t := range targets {
		if !InRange(author.Position, t.enemy.Position, ar) {
			continue
		}
		return &api.PlantationAction{
			Path: []api.Coord{author.Position, author.Position, t.enemy.Position},
		}
	}
	return nil
}

// ═══════════════════════════════════════════════════════════════
// CU ROTATION
// ═══════════════════════════════════════════════════════════════

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

	// Build cell progress lookup.
	cellProg := make(map[api.Coord]int)
	for _, cell := range arena.Cells {
		cellProg[cell.Position] = cell.TerraformationProgress
	}
	progress := cellProg[main.Position]

	// Compute centroid of all plantations for directional preference.
	centX, centY := 0, 0
	for pos := range myPos {
		centX += pos[0]
		centY += pos[1]
	}
	if len(myPos) > 0 {
		centX /= len(myPos)
		centY /= len(myPos)
	}

	// Collect adjacent connected plantations as relay candidates.
	type relayCandidate struct {
		pos      api.Coord
		progress int
		score    int
	}
	var candidates []relayCandidate
	for _, adj := range Adjacents(main.Position) {
		p, ok := myPos[adj]
		if !ok || p.IsIsolated || !connected[adj] || p.IsMain {
			continue
		}
		_ = p
		candidates = append(candidates, relayCandidate{
			pos:      adj,
			progress: cellProg[adj],
		})
	}

	if len(candidates) == 0 {
		if progress >= 50 {
			log.Printf("[strategy] WARNING: CU at %v progress %d%% but no adjacent plantation!", main.Position, progress)
		}
		return nil
	}

	// === "Running CU" strategy (B+C hybrid) ===
	//
	// Opportunistic: relocate early (progress >= 30%) if there's a significantly
	// fresher neighbor (at least 20% less progress).
	// Direction: prefer candidates closer to the cluster centroid.
	// Emergency: always relocate at progress >= 80% to any better neighbor.

	minThreshold := 30 // opportunistic threshold
	freshGap := 20     // target must be at least this much fresher
	emergencyThreshold := 80

	if progress < minThreshold {
		return nil // too early, no point checking
	}

	// Score candidates: lower progress = better, closer to centroid = better.
	for i := range candidates {
		cc := &candidates[i]
		// Freshness score: how much better than current position.
		cc.score = progress - cc.progress // higher = fresher relative to CU
		// Centroid proximity bonus: prefer moving toward cluster center.
		curDistToCentroid := abs(main.Position[0]-centX) + abs(main.Position[1]-centY)
		candDistToCentroid := abs(cc.pos[0]-centX) + abs(cc.pos[1]-centY)
		if candDistToCentroid < curDistToCentroid {
			cc.score += 10 // bonus for moving toward center
		}
	}

	// Find best candidate.
	best := candidates[0]
	for _, cc := range candidates[1:] {
		if cc.score > best.score || (cc.score == best.score && cc.progress < best.progress) {
			best = cc
		}
	}

	// Decision: relocate if conditions are met.
	if progress >= emergencyThreshold {
		// Emergency: relocate to anything better.
		if best.progress >= progress {
			log.Printf("[strategy] skip emergency relocate: CU at %v (%d%%), best %v (%d%%) is not better",
				main.Position, progress, best.pos, best.progress)
			return nil
		}
		log.Printf("[strategy] EMERGENCY relocate CU: %v -> %v (CU %d%%, target %d%%)", main.Position, best.pos, progress, best.progress)
		return []api.Coord{main.Position, best.pos}
	}

	// Opportunistic: relocate only if target is significantly fresher.
	if best.progress > progress-freshGap {
		return nil // not fresh enough to justify early move
	}

	log.Printf("[strategy] relocate CU: %v -> %v (CU %d%%, target %d%%, gap %d%%)",
		main.Position, best.pos, progress, best.progress, progress-best.progress)
	return []api.Coord{main.Position, best.pos}
}

// ═══════════════════════════════════════════════════════════════
// SMART REPAIR (economic model)
// ═══════════════════════════════════════════════════════════════

type repairCandidate struct {
	plant    *api.Plantation
	priority int  // higher = more urgent
	critical bool // true = will die before terraformation completes
}

// calcIncomingDamage estimates damage per turn for each cell position.
func (c *Conveyor) calcIncomingDamage(arena *api.ArenaResponse, beaverMit int) map[api.Coord]int {
	dmg := make(map[api.Coord]int)
	beaverDmg := 15 - beaverMit*2

	// Beaver damage: 15 HP/turn to all in AR=2.
	for _, b := range arena.Beavers {
		for dx := -2; dx <= 2; dx++ {
			for dy := -2; dy <= 2; dy++ {
				pos := api.Coord{b.Position[0] + dx, b.Position[1] + dy}
				dmg[pos] += beaverDmg
			}
		}
	}

	// Enemy sabotage: assume SE=5 per visible enemy in AR=2.
	for _, e := range arena.Enemy {
		for _, p := range arena.Plantations {
			if InRange(e.Position, p.Position, arena.ActionRange) {
				dmg[p.Position] += 5
				break // 1 enemy attacks 1 plantation
			}
		}
	}

	// Storm damage: 2 HP/turn if active storm.
	for _, m := range arena.MeteoForecasts {
		if m.Kind == "sandstorm" && m.Forming != nil && !*m.Forming && m.Position != nil && m.Radius != nil {
			r2 := (*m.Radius) * (*m.Radius)
			for _, p := range arena.Plantations {
				dx := p.Position[0] - m.Position[0]
				dy := p.Position[1] - m.Position[1]
				if dx*dx+dy*dy <= r2 {
					dmg[p.Position] += 2
				}
			}
		}
	}

	return dmg
}

// evaluateRepairNeed decides which plantations need repair and how urgently.
func (c *Conveyor) evaluateRepairNeed(arena *api.ArenaResponse, connected map[api.Coord]bool, cellProgress map[api.Coord]int, incomingDamage map[api.Coord]int, repairPower, mhp int) []repairCandidate {
	var candidates []repairCandidate

	for i := range arena.Plantations {
		p := &arena.Plantations[i]
		if p.IsIsolated || !connected[p.Position] {
			continue
		}

		progress := cellProgress[p.Position]
		dmgPerTurn := incomingDamage[p.Position]

		// How many turns until terraformation completes?
		turnsToComplete := 0
		if progress < 100 {
			turnsToComplete = (100 - progress + 4) / 5 // ceil(remaining / TS)
		}

		// If already at 100% or close — this plantation will disappear soon, don't heal.
		if turnsToComplete <= 1 {
			continue
		}

		// How many turns until this plantation dies?
		turnsToLive := 999
		if dmgPerTurn > 0 {
			turnsToLive = p.HP / dmgPerTurn
		}

		// Will it survive until terraformation completes?
		if turnsToLive >= turnsToComplete && dmgPerTurn == 0 {
			continue // No damage, will survive — don't waste repair.
		}

		if turnsToLive >= turnsToComplete && dmgPerTurn > 0 {
			// Taking damage but will survive. Low priority repair.
			// But check: can incoming damage increase? (enemy could build more)
			predictedDmg := dmgPerTurn + 5 // assume enemy might add 1 more attacker
			predictedLife := p.HP / predictedDmg
			if predictedLife >= turnsToComplete {
				continue // Even with escalation, probably fine.
			}
			// Might die with escalation — low priority heal.
			candidates = append(candidates, repairCandidate{
				plant:    p,
				priority: 10,
				critical: false,
			})
			continue
		}

		// Will die before completing terraformation. Worth healing?
		// Cost: how many repair-turns needed?
		hpDeficit := dmgPerTurn*turnsToComplete - p.HP
		repairTurnsNeeded := (hpDeficit + repairPower - 1) / repairPower

		// If repair costs more turns than just building a new plantation (10 turns),
		// building new is cheaper.
		if repairTurnsNeeded > 10 {
			continue // Not worth healing — build new instead.
		}

		priority := 50

		// ЦУ: always critical — losing it = losing everything.
		if p.IsMain {
			priority = 200
		}

		// Chain link: if this plantation is the only connection for others,
		// losing it isolates them.
		adjConnected := 0
		for _, adj := range Adjacents(p.Position) {
			if connected[adj] {
				adjConnected++
			}
		}
		if adjConnected >= 2 {
			priority += 30 // chain link — important
		}

		// Boosted cell: worth more points.
		if IsBoosted(p.Position) {
			priority += 20
		}

		// Lower HP = more urgent.
		if p.HP < 15 {
			priority += 40
		} else if p.HP < 30 {
			priority += 20
		}

		// Higher incoming damage = more urgent.
		priority += dmgPerTurn * 3

		candidates = append(candidates, repairCandidate{
			plant:    p,
			priority: priority,
			critical: turnsToLive < 5 || p.IsMain,
		})
	}

	// Sort by priority descending.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].priority > candidates[j].priority
	})

	return candidates
}

// trySmartRepair assigns repair based on economic analysis.
// criticalOnly=true → only repair critical plantations (ЦУ, about to die).
// criticalOnly=false → repair any candidate.
func (c *Conveyor) trySmartRepair(author *api.Plantation, candidates []repairCandidate, assigned map[api.Coord]bool, ar int, criticalOnly bool) *api.PlantationAction {
	for _, rc := range candidates {
		if criticalOnly && !rc.critical {
			continue
		}
		if rc.plant.Position == author.Position || assigned[rc.plant.Position] {
			continue
		}
		if InRange(author.Position, rc.plant.Position, ar) {
			assigned[rc.plant.Position] = true
			return &api.PlantationAction{
				Path: []api.Coord{author.Position, author.Position, rc.plant.Position},
			}
		}
	}
	return nil
}

// ═══════════════════════════════════════════════════════════════
// BUILD
// ═══════════════════════════════════════════════════════════════

func (c *Conveyor) tryContinueBuild(author *api.Plantation, constructions []api.Construction, assigned map[api.Coord]int, ar int) *api.PlantationAction {
	// Prioritize construction closest to COMPLETION (highest progress).
	// Adaptive cap: distribute builders evenly — total connected plants / active constructions.
	// This ensures each construction gets enough builders without starving others.
	maxPerConstruction := 999
	if len(constructions) > 0 {
		// Estimate: we want each construction to get roughly equal share.
		// But minimum 3 to ensure progress outpaces degradation.
		maxPerConstruction = (c.turnConnected + len(constructions) - 1) / len(constructions)
		if maxPerConstruction < 3 {
			maxPerConstruction = 3
		}
	}

	// Filter constructions: only those we can realistically finish.
	// Estimate: how many builders can reach this construction?
	// If turnsToFinish > 10 with available builders, skip it.
	type conRanked struct {
		con           *api.Construction
		progress      int
		turnsToFinish int
	}
	var ranked []conRanked
	for i := range constructions {
		con := &constructions[i]
		if !InRange(author.Position, con.Position, ar) {
			continue
		}
		remaining := 50 - con.Progress
		if remaining <= 0 {
			continue
		}

		// Estimate builders: at least 1 (this author) + already assigned.
		expectedBuilders := assigned[con.Position] + 1
		if expectedBuilders < 1 {
			expectedBuilders = 1
		}
		progressPerTurn := expectedBuilders * 5
		turnsToFinish := (remaining + progressPerTurn - 1) / progressPerTurn

		// Skip only truly hopeless: would take >15 turns even with multiple builders.
		// Estimate with 3 builders (typical team size after P2 assignment).
		turnsWithTeam := (remaining + 14) / 15 // 3 builders × 5 progress
		if c.turnConnected > 5 && turnsWithTeam > 5 && con.Progress <= 5 {
			continue
		}

		ranked = append(ranked, conRanked{con, con.Progress, turnsToFinish})
	}

	// Sort by: priority = progress (finish closest-to-done first).
	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].progress > ranked[j].progress
	})
	// No hard limit — let all viable constructions receive builders.
	// maxPerConstruction already prevents over-assignment.

	var best *api.Construction
	bestScore := -1
	for _, r := range ranked {
		if assigned[r.con.Position] >= maxPerConstruction {
			continue
		}
		if r.progress > bestScore {
			bestScore = r.progress
			best = r.con
		}
	}
	if best == nil {
		return nil
	}
	assigned[best.Position]++
	return &api.PlantationAction{
		Path: []api.Coord{author.Position, author.Position, best.Position},
	}
}

func (c *Conveyor) tryNewBuild(author *api.Plantation, occupied, mountains map[api.Coord]bool, beaverPositions []api.Coord, enemies map[api.Coord]*api.EnemyPlantation, myPos map[api.Coord]*api.Plantation, cellProgress map[api.Coord]int, cellDegTurns map[api.Coord]int, isolatedPositions []api.Coord, ar int, targetBeaver *api.Coord) *api.PlantationAction {
	type candidate struct {
		pos   api.Coord
		score int
	}
	var best *candidate

	// Find CU position and centroid of all plantations.
	var cuPos api.Coord
	centroidX, centroidY := 0, 0
	for pos, p := range myPos {
		if p.IsMain {
			cuPos = pos
		}
		centroidX += pos[0]
		centroidY += pos[1]
	}
	if len(myPos) > 0 {
		centroidX /= len(myPos)
		centroidY /= len(myPos)
	}

	// Build a set of construction positions — we allow building adjacent to
	// active constructions (they'll become plantations by the time we finish).
	constructing := make(map[api.Coord]bool)
	for pos := range occupied {
		if myPos[pos] == nil {
			constructing[pos] = true
		}
	}

	for dx := -ar; dx <= ar; dx++ {
		for dy := -ar; dy <= ar; dy++ {
			target := api.Coord{author.Position[0] + dx, author.Position[1] + dy}
			if target[0] < 0 || target[0] >= c.mapWidth || target[1] < 0 || target[1] >= c.mapHeight {
				continue
			}
			if occupied[target] || mountains[target] {
				continue
			}
			if _, isEnemy := enemies[target]; isEnemy {
				continue
			}
			prog := cellProgress[target]
			if prog >= 100 {
				continue
			}
			// Skip cells that will die too fast after construction.
			// Exception: when building toward a target beaver, allow higher progress cells
			// as stepping stones (they only need to survive long enough to connect).
			turnsAliveAfterBuild := (100 - prog) / 5
			minAlive := 6
			if targetBeaver != nil {
				tbDist := abs(target[0]-(*targetBeaver)[0])
				if abs(target[1]-(*targetBeaver)[1]) > tbDist {
					tbDist = abs(target[1] - (*targetBeaver)[1])
				}
				if tbDist <= 4 {
					minAlive = 2 // stepping stone toward beaver — only needs to live briefly
				}
			}
			if prog > 0 && turnsAliveAfterBuild < minAlive {
				continue
			}

			// Count adjacent plantations AND constructions.
			// Constructions count as future neighbors — by the time we finish building,
			// the adjacent construction will likely be a plantation.
			adjPlants := 0
			adjConstructions := 0
			hasLivingNeighbor := false
			adjToCU := false
			for _, adj := range Adjacents(target) {
				if p, ok := myPos[adj]; ok {
					adjPlants++
					adjProg := cellProgress[adj]
					turnsToLive := (100 - adjProg) / 5
					if p.IsMain {
						adjToCU = true
						if adjProg < 50 {
							hasLivingNeighbor = true
						}
					} else if turnsToLive >= 10 {
						hasLivingNeighbor = true
					}
				} else if constructing[adj] {
					adjConstructions++
					hasLivingNeighbor = true // construction = future plantation
				}
			}
			adjTotal := adjPlants + adjConstructions
			if adjTotal == 0 {
				continue
			}
			if !hasLivingNeighbor && len(myPos) > 1 {
				continue
			}

			// === SCORING: build toward cluster center, stay connected ===
			score := 100

			// CLUSTER DENSITY: distance to centroid of all plantations.
			// Build toward the center of mass, not toward CU edge.
			distToCentroid := abs(target[0]-centroidX) + abs(target[1]-centroidY)
			score += 300 - distToCentroid*30

			// CU connectivity: still important but secondary to cluster.
			distToCU := abs(target[0]-cuPos[0]) + abs(target[1]-cuPos[1])
			score += 200 - distToCU*20

			// Adjacent to CU = bonus for CU survival.
			if adjToCU {
				score += 150
			}

			// NEIGHBOR DENSITY: strongly favor cells surrounded by existing plants.
			// More neighbors = filling gaps in existing cluster, not expanding frontier.
			score += adjPlants * 100
			score += adjConstructions * 30

			// CELL PROGRESS PENALTY: prefer fresh cells.
			if prog > 0 {
				score -= prog * 2
			}

			// Boosted cell bonus (7×7 grid).
			if IsBoosted(target) {
				score += 100
			}
			nearestBoostedDist := nearestBoostedDistance(target)
			if nearestBoostedDist <= 3 {
				score += 40
			}

			// MAP EDGE PENALTY: building near edges limits future expansion directions.
			// Lose ~25% expansion room per adjacent edge.
			edgeDist := target[0] // distance to left edge
			if c.mapWidth-1-target[0] < edgeDist {
				edgeDist = c.mapWidth - 1 - target[0] // distance to right edge
			}
			if target[1] < edgeDist {
				edgeDist = target[1] // distance to top edge
			}
			if c.mapHeight-1-target[1] < edgeDist {
				edgeDist = c.mapHeight - 1 - target[1] // distance to bottom edge
			}
			if edgeDist < 10 {
				score -= (10 - edgeDist) * 30 // 0→-300, 3→-210, 5→-150, 9→-30
			}

			// BEAVER GRADIENT PENALTY: closer to beaver = much worse.
			// Exception: if we're building toward a target beaver for attack,
			// skip penalty for that beaver and ADD bonus for getting closer.
			for _, bp := range beaverPositions {
				// Skip penalty for target beaver — we WANT to build toward it.
				if targetBeaver != nil && bp == *targetBeaver {
					bDist := abs(target[0]-bp[0])
					if abs(target[1]-bp[1]) > bDist {
						bDist = abs(target[1] - bp[1])
					}
					// Bonus for getting close to beaver. Prefer dist=3 (just outside
				// damage zone) over dist=2 (inside — construction may be destroyed).
					if bDist == 3 {
						score += 600 // sweet spot: outside damage, 1 build to AR
					} else if bDist == 2 {
						score += 400 // inside AR but risky (beaver damages plants here)
					} else if bDist == 4 {
						score += 300 // 2 builds to reach AR
					}
					continue
				}

				bDist := abs(target[0]-bp[0])
				if abs(target[1]-bp[1]) > bDist {
					bDist = abs(target[1] - bp[1])
				}
				if bDist <= 2 {
					score -= 1000 // inside damage zone — never build
				} else if bDist <= 4 {
					score -= (5 - bDist) * 200 // 3→-400, 4→-200
				}
				// dist 5+: no penalty. Outer zone was causing corridor traps
				// when two beavers squeezed the cluster from both sides.
			}

			if score > 0 && (best == nil || score > best.score) {
				best = &candidate{target, score}
			}
		}
	}

	// Sticky build target: if we were building somewhere last turn and it's still valid,
	// keep building there. This prevents alternating between two equal-score targets.
	// Skip sticky logic when building toward a beaver — beaver target takes priority.
	if c.lastBuildTarget != nil && best != nil && targetBeaver == nil {
		lastPos := *c.lastBuildTarget
		if lastPos != best.pos {
			// Check if last target is still a valid candidate.
			if !occupied[lastPos] && !mountains[lastPos] && InRange(author.Position, lastPos, ar) {
				prog := cellProgress[lastPos]
				if prog < 100 {
					turnsAlive := (100 - prog) / 5
					if prog == 0 || turnsAlive >= 6 {
						// Last target is still valid — stick with it.
						best = &candidate{lastPos, best.score + 1}
					}
				}
			}
		}
	}

	if best == nil {
		if targetBeaver != nil {
			log.Printf("[strategy] tryNewBuild BEAVER FAIL: author %v, targetBeaver %v, ar=%d, plants=%d — no valid candidates found",
				author.Position, *targetBeaver, ar, len(myPos))
		}
		c.lastBuildTarget = nil
		return nil
	}

	// Don't save beaver build targets as sticky — they're one-off directional builds.
	if targetBeaver == nil {
		c.lastBuildTarget = &best.pos
	}
	occupied[best.pos] = true
	return &api.PlantationAction{
		Path: []api.Coord{author.Position, author.Position, best.pos},
	}
}

// ═══════════════════════════════════════════════════════════════
// UTILS
// ═══════════════════════════════════════════════════════════════

func nearestBoostedDistance(c api.Coord) int {
	// Nearest x%7==0 and y%7==0
	rx := c[0] % 7
	if rx > 3 { rx = 7 - rx }
	ry := c[1] % 7
	if ry > 3 { ry = 7 - ry }
	return rx + ry
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	s := ""
	neg := n < 0
	if neg {
		n = -n
	}
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	if neg {
		s = "-" + s
	}
	return s
}
