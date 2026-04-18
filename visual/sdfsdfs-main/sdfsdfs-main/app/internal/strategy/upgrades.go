package strategy

import "datssol/internal/api"

/*
Upgrade plan v10 — SR + vision + decay, then settlement limit.

Point:  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
Turn:  30  60  90 120 150 180 210 240 270 300 330 360 390 420 450

  signal_range:     1   2   3
  vision_range:                 4
  decay_mitig:                      5   6   7
  settlement_limit:                         8   9  10  11  12  13
  max_hp:                                                       14  15
*/

var upgradeSequence = []string{
	"signal_range",          // 1:  SR 3→4
	"signal_range",          // 2:  SR 4→5
	"signal_range",          // 3:  SR 5→6
	"vision_range",          // 4:  VR 3→5 (+2 per level)
	"decay_mitigation",      // 5:  DS 10→8
	"decay_mitigation",      // 6:  DS 8→6
	"decay_mitigation",      // 7:  DS 6→4
	"settlement_limit",      // 8:  limit 30→31
	"settlement_limit",      // 9:  limit 31→32
	"settlement_limit",      // 10: limit 32→33
	"settlement_limit",      // 11: limit 33→34
	"settlement_limit",      // 12: limit 34→35
	"settlement_limit",      // 13: limit 35→36
	"max_hp",                // 14: MHP 50→60
	"max_hp",                // 15: MHP 60→70
}

func PickUpgrade(ups api.UpgradesState) string {
	if ups.Points <= 0 {
		return ""
	}

	tierMap := make(map[string]api.UpgradeTier)
	for _, t := range ups.Tiers {
		tierMap[t.Name] = t
	}

	totalPurchased := 0
	for _, t := range ups.Tiers {
		totalPurchased += t.Current
	}

	for idx := totalPurchased; idx < len(upgradeSequence); idx++ {
		name := upgradeSequence[idx]
		tier, ok := tierMap[name]
		if ok && (tier.Max < 0 || tier.Current < tier.Max) {
			return name
		}
	}

	return ""
}
