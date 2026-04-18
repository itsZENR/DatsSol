package api

type Coord [2]int

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

type CommandRequest struct {
	Command           []PlantationAction `json:"command,omitempty"`
	PlantationUpgrade string             `json:"plantationUpgrade,omitempty"`
	RelocateMain      []Coord            `json:"relocateMain,omitempty"`
}

type PlantationAction struct {
	Path []Coord `json:"path"`
}

type CommandResponse struct {
	Code   int      `json:"code"`
	Errors []string `json:"errors"`
}

type LogMessage struct {
	Time    string `json:"time"`
	Message string `json:"message"`
}
