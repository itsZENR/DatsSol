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
