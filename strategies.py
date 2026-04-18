"""
Strategy pool for DatsSol bot.

Each strategy implements:
  priority(state, bot_state) -> float   (higher = more urgent)
  decide(state, bot_state) -> TurnDecision

BotState carries persistent info across turns (beaver HP history, etc.).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from models import GameState, TurnDecision, Command, Plantation, Pos
from game_logic import (
    find_best_relay, build_command, find_expansion_targets,
    find_repair_targets, find_sabotage_targets, find_beaver_targets,
    assess_enemy_strength, in_range, earthquake_incoming, is_storm_approaching,
    chebyshev,
)
from config import (
    DEFAULT_AR, DEFAULT_SR, DEFAULT_CS, DEFAULT_BE, DEFAULT_RS, DEFAULT_SE,
    DEFAULT_MHP, HQ_REPAIR_THRESHOLD, REPAIR_THRESHOLD,
    BEAVER_SNIPE_HP, MIN_PLANTATIONS_FOR_AGGRESSION,
    UPGRADE_SEQUENCE_SPEC,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upgrade sequence parser
# ---------------------------------------------------------------------------

def parse_upgrade_sequence(spec: str) -> list[str]:
    """
    Parse upgrade sequence spec into ordered list of upgrade names.
    Format per line: "upgrade_name:pattern"
    Pattern chars: digit 1-9 or a-f (hex) = global step, '*' = skip.
    Returns list indexed by step order (0-based).
    """
    steps: dict[int, str] = {}
    for line in spec.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, pattern = line.split(":", 1)
        name = name.strip()
        for i, ch in enumerate(pattern):
            if ch == "*":
                continue
            if ch.isdigit():
                step = int(ch)
            elif ch in "abcdef":
                step = 10 + "abcdef".index(ch)
            else:
                continue
            if step in steps:
                log.warning("Duplicate step %d in upgrade spec", step)
            steps[step] = name
    return [steps[k] for k in sorted(steps.keys())]


UPGRADE_ORDER: list[str] = parse_upgrade_sequence(UPGRADE_SEQUENCE_SPEC)


# ---------------------------------------------------------------------------
# Persistent bot state
# ---------------------------------------------------------------------------

@dataclass
class BeaverHistory:
    last_hp: int = 100
    hp_drop_rate: float = 0.0   # observed HP drop per turn (from external attackers)
    our_damage_last_turn: int = 0
    attacking: bool = False


@dataclass
class BotState:
    turn: int = 0
    mhp: int = DEFAULT_MHP
    ar: int = DEFAULT_AR
    sr: int = DEFAULT_SR
    cs: int = DEFAULT_CS
    rs: int = DEFAULT_RS
    se: int = DEFAULT_SE
    be: int = DEFAULT_BE
    plantation_limit: int = 30
    upgrades_applied: list[str] = field(default_factory=list)
    beaver_history: dict[str, BeaverHistory] = field(default_factory=dict)
    enemy_cluster_history: list[dict] = field(default_factory=list)

    def update_from_state(self, state: GameState):
        self.turn = state.turn_no
        self.ar = state.action_range or DEFAULT_AR

    def update_beaver_history(self, state: GameState):
        seen_ids = {b.id for b in state.beavers}
        for bid in list(self.beaver_history.keys()):
            if bid not in seen_ids:
                del self.beaver_history[bid]
        for b in state.beavers:
            if b.id not in self.beaver_history:
                self.beaver_history[b.id] = BeaverHistory(last_hp=b.hp)
            else:
                hist = self.beaver_history[b.id]
                drop = hist.last_hp - b.hp + 5  # +5 accounts for regen
                # EMA of drop rate
                hist.hp_drop_rate = 0.7 * hist.hp_drop_rate + 0.3 * max(0, drop)
                hist.last_hp = b.hp

    def get_next_upgrade(self, state: GameState) -> Optional[str]:
        """Return next upgrade to buy, or None if nothing available."""
        if state.upgrades.points <= 0:
            return None
        applied_counts: dict[str, int] = {}
        for name in self.upgrades_applied:
            applied_counts[name] = applied_counts.get(name, 0) + 1

        # Follow the configured sequence
        for name in UPGRADE_ORDER:
            tier = state.get_upgrade_tier(name)
            if tier is None:
                continue
            bought = applied_counts.get(name, 0)
            if bought < tier.max and bought >= tier.current:
                # This is the next level we should buy
                if tier.current < tier.max:
                    return name

        # Fallback: buy anything available
        for tier in state.upgrades.tiers:
            if tier.current < tier.max:
                return tier.name
        return None


# ---------------------------------------------------------------------------
# Base strategy class
# ---------------------------------------------------------------------------

class Strategy:
    name: str = "base"

    def priority(self, state: GameState, bot: BotState) -> float:
        return 0.0

    def decide(self, state: GameState, bot: BotState) -> TurnDecision:
        return TurnDecision()


# ---------------------------------------------------------------------------
# Strategy: Upgrade
# ---------------------------------------------------------------------------

class UpgradeStrategy(Strategy):
    name = "upgrade"

    def priority(self, state: GameState, bot: BotState) -> float:
        if state.upgrades.points > 0 and bot.get_next_upgrade(state):
            return 100.0
        return 0.0

    def decide(self, state: GameState, bot: BotState) -> TurnDecision:
        name = bot.get_next_upgrade(state)
        if not name:
            return TurnDecision()
        bot.upgrades_applied.append(name)
        log.info("Buying upgrade: %s", name)
        return TurnDecision(upgrade=name)


# ---------------------------------------------------------------------------
# Strategy: Repair HQ
# ---------------------------------------------------------------------------

class RepairHQStrategy(Strategy):
    name = "repair_hq"

    def priority(self, state: GameState, bot: BotState) -> float:
        hq = state.main_plantation
        if not hq:
            return 0.0
        if hq.hp < bot.mhp * HQ_REPAIR_THRESHOLD:
            # Urgency increases as HP drops
            return 90.0 + (1.0 - hq.hp / bot.mhp) * 50.0
        return 0.0

    def decide(self, state: GameState, bot: BotState) -> TurnDecision:
        hq = state.main_plantation
        if not hq:
            return TurnDecision()

        relay_usage: dict[str, int] = {}
        commands: list[Command] = []

        for p in state.connected_plantations:
            if p.id == hq.id:
                continue
            relay = find_best_relay(state, p, tuple(hq.position), relay_usage, bot.ar, bot.sr)
            if relay:
                cmd = build_command(p, relay, tuple(hq.position), relay_usage)
                commands.append(cmd)
                if len(commands) >= 3:  # enough repairers
                    break

        if commands:
            log.info("Repairing HQ (hp=%d)", hq.hp)
        return TurnDecision(commands=commands)


# ---------------------------------------------------------------------------
# Strategy: Repair network
# ---------------------------------------------------------------------------

class RepairStrategy(Strategy):
    name = "repair"

    def priority(self, state: GameState, bot: BotState) -> float:
        damaged = [
            p for p in state.connected_plantations
            if not p.is_main and p.hp < bot.mhp * REPAIR_THRESHOLD
        ]
        if damaged:
            return 70.0
        return 0.0

    def decide(self, state: GameState, bot: BotState) -> TurnDecision:
        relay_usage: dict[str, int] = {}
        commands: list[Command] = []
        assigned_targets: set[str] = set()

        # For each connected plantation, find a damaged neighbour
        for p in state.connected_plantations:
            targets = find_repair_targets(state, p, bot.mhp, REPAIR_THRESHOLD, bot.ar, bot.sr)
            for target in targets:
                if target.id in assigned_targets:
                    continue
                relay = find_best_relay(state, p, tuple(target.position), relay_usage, bot.ar, bot.sr)
                if relay:
                    cmd = build_command(p, relay, tuple(target.position), relay_usage)
                    commands.append(cmd)
                    assigned_targets.add(target.id)
                    break  # each plantation issues one command

        return TurnDecision(commands=commands)


# ---------------------------------------------------------------------------
# Strategy: Expand (build new plantations)
# ---------------------------------------------------------------------------

class ExpandStrategy(Strategy):
    name = "expand"

    def priority(self, state: GameState, bot: BotState) -> float:
        n = len(state.plantations) + len(state.constructions)
        if n >= bot.plantation_limit:
            return 0.0
        # More urgent early game, lower priority as we approach limit
        fill_ratio = n / max(1, bot.plantation_limit)
        return 60.0 * (1.0 - fill_ratio * 0.5)

    def decide(self, state: GameState, bot: BotState) -> TurnDecision:
        relay_usage: dict[str, int] = {}
        commands: list[Command] = []
        claimed_targets: set[Pos] = set()

        n_active = len(state.plantations) + len(state.constructions)
        slots_free = bot.plantation_limit - n_active

        if slots_free <= 0:
            return TurnDecision()

        # Collect all candidate build targets with scores
        all_candidates: list[tuple[float, Pos, Plantation]] = []
        for p in state.connected_plantations:
            for target_pos, score in find_expansion_targets(state, p, bot.ar):
                if target_pos not in claimed_targets and target_pos not in state.construction_set:
                    all_candidates.append((score, target_pos, p))

        # Sort by score, pick top unique targets
        all_candidates.sort(key=lambda x: -x[0])
        builds_this_turn = 0

        for score, target_pos, builder in all_candidates:
            if builds_this_turn >= slots_free:
                break
            if target_pos in claimed_targets:
                continue
            relay = find_best_relay(state, builder, target_pos, relay_usage, bot.ar, bot.sr)
            if relay:
                cmd = build_command(builder, relay, target_pos, relay_usage)
                commands.append(cmd)
                claimed_targets.add(target_pos)
                builds_this_turn += 1

        if commands:
            log.info("Building %d new plantations", len(commands))
        return TurnDecision(commands=commands)


# ---------------------------------------------------------------------------
# Strategy: Beaver attack (snipe / last hit)
# ---------------------------------------------------------------------------

class BeaverSnipeStrategy(Strategy):
    """
    Wait until we can deal more damage than any enemy in the killing turn.
    Track observed HP drop to estimate enemy attack strength.
    Commit only when our total BE > enemy total BE near the beaver.
    """
    name = "beaver_snipe"

    def priority(self, state: GameState, bot: BotState) -> float:
        if not state.beavers:
            return 0.0
        if len(state.connected_plantations) < MIN_PLANTATIONS_FOR_AGGRESSION:
            return 0.0

        for b in state.beavers:
            hist = bot.beaver_history.get(b.id)
            if not hist:
                continue
            our_reach = self._count_our_reach(b, state, bot)
            if our_reach == 0:
                continue
            our_dmg = our_reach * bot.be
            # If we can kill it and deliver more damage than competitors
            if b.hp <= our_dmg and our_dmg >= hist.hp_drop_rate:
                return 50.0
        return 0.0

    def decide(self, state: GameState, bot: BotState) -> TurnDecision:
        relay_usage: dict[str, int] = {}
        commands: list[Command] = []

        for b in state.beavers:
            hist = bot.beaver_history.get(b.id)
            if not hist:
                continue
            our_reach_plants = self._get_our_reach_plants(b, state, bot)
            if not our_reach_plants:
                continue
            our_dmg = len(our_reach_plants) * bot.be
            if b.hp > our_dmg or our_dmg < hist.hp_drop_rate:
                continue

            b_pos = tuple(b.position)
            for p in our_reach_plants:
                relay = find_best_relay(state, p, b_pos, relay_usage, bot.ar, bot.sr)
                if relay:
                    cmd = build_command(p, relay, b_pos, relay_usage)
                    commands.append(cmd)
            log.info("SNIPE: attacking beaver %s (hp=%d) with %d plantations", b.id, b.hp, len(commands))
            break  # focus one beaver per turn

        return TurnDecision(commands=commands)

    def _count_our_reach(self, b, state: GameState, bot: BotState) -> int:
        return len(self._get_our_reach_plants(b, state, bot))

    def _get_our_reach_plants(self, b, state: GameState, bot: BotState) -> list[Plantation]:
        b_pos = tuple(b.position)
        result = []
        for p in state.connected_plantations:
            p_pos = tuple(p.position)
            if in_range(p_pos, b_pos, bot.ar):
                result.append(p)
                continue
            for relay in state.connected_plantations:
                r_pos = tuple(relay.position)
                if in_range(p_pos, r_pos, bot.sr) and in_range(r_pos, b_pos, bot.ar):
                    result.append(p)
                    break
        return result


class BeaverAttackStrategy(Strategy):
    """
    Attack beavers that are blocking expansion or have low HP.
    Lower priority than snipe (we don't care about the last hit here).
    Used to soften beavers that block our path.
    """
    name = "beaver_attack"

    def priority(self, state: GameState, bot: BotState) -> float:
        if not state.beavers:
            return 0.0
        # Only attack beavers that threaten our plantations
        for b in state.beavers:
            for p in state.connected_plantations:
                if in_range(tuple(b.position), tuple(p.position), 2):
                    return 30.0
        return 0.0

    def decide(self, state: GameState, bot: BotState) -> TurnDecision:
        relay_usage: dict[str, int] = {}
        commands: list[Command] = []

        # Prioritize beavers attacking our plantations
        threatening_beavers = []
        for b in state.beavers:
            for p in state.connected_plantations:
                if in_range(tuple(b.position), tuple(p.position), 2):
                    threatening_beavers.append(b)
                    break

        threatening_beavers.sort(key=lambda b: b.hp)

        for p in state.connected_plantations:
            targets = find_beaver_targets(state, p, bot.ar, bot.sr)
            # Prefer threatening beavers
            threat_ids = {b.id for b in threatening_beavers}
            targets.sort(key=lambda b: (0 if b.id in threat_ids else 1, b.hp))
            if targets:
                b = targets[0]
                b_pos = tuple(b.position)
                relay = find_best_relay(state, p, b_pos, relay_usage, bot.ar, bot.sr)
                if relay:
                    cmd = build_command(p, relay, b_pos, relay_usage)
                    commands.append(cmd)

        return TurnDecision(commands=commands)


# ---------------------------------------------------------------------------
# Strategy: Sabotage weak enemies
# ---------------------------------------------------------------------------

class SabotageStrategy(Strategy):
    name = "sabotage"

    def priority(self, state: GameState, bot: BotState) -> float:
        if len(state.connected_plantations) < MIN_PLANTATIONS_FOR_AGGRESSION:
            return 0.0
        if not state.enemies:
            return 0.0
        # Check if any enemy is within reach
        for p in state.connected_plantations:
            if find_sabotage_targets(state, p, bot.ar, bot.sr):
                return 40.0
        return 0.0

    def decide(self, state: GameState, bot: BotState) -> TurnDecision:
        relay_usage: dict[str, int] = {}
        commands: list[Command] = []
        enemy_targeted: set[str] = set()

        # Find weakest reachable enemy across all our plantations
        all_targets: list[tuple[int, object, Plantation]] = []
        for p in state.connected_plantations:
            for e in find_sabotage_targets(state, p, bot.ar, bot.sr):
                all_targets.append((e.hp, e, p))

        all_targets.sort(key=lambda x: x[0])

        for hp, enemy, attacker in all_targets:
            if enemy.id in enemy_targeted:
                continue
            relay = find_best_relay(state, attacker, tuple(enemy.position), relay_usage, bot.ar, bot.sr)
            if relay:
                cmd = build_command(attacker, relay, tuple(enemy.position), relay_usage)
                commands.append(cmd)
                enemy_targeted.add(enemy.id)
                if len(commands) >= 5:
                    break

        if commands:
            log.info("Sabotaging %d enemy plantations", len(commands))
        return TurnDecision(commands=commands)


# ---------------------------------------------------------------------------
# Strategy: Relocate HQ away from threats
# ---------------------------------------------------------------------------

class RelocateHQStrategy(Strategy):
    """Move HQ to safer adjacent plantation when under heavy threat."""
    name = "relocate_hq"

    def priority(self, state: GameState, bot: BotState) -> float:
        hq = state.main_plantation
        if not hq:
            return 0.0
        hq_pos = tuple(hq.position)

        # Check if any enemy or beaver can reach HQ directly
        threatened = False
        for e in state.enemies:
            if in_range(tuple(e.position), hq_pos, bot.ar):
                threatened = True
                break
        for b in state.beavers:
            if in_range(tuple(b.position), hq_pos, 2):
                threatened = True
                break

        if threatened and hq.hp < bot.mhp * 0.5:
            return 95.0
        return 0.0

    def decide(self, state: GameState, bot: BotState) -> TurnDecision:
        hq = state.main_plantation
        if not hq:
            return TurnDecision()
        hq_pos = tuple(hq.position)

        # Find adjacent connected plantation with highest HP
        candidates = []
        for p in state.connected_plantations:
            if p.id == hq.id:
                continue
            p_pos = tuple(p.position)
            # Must be orthogonally adjacent to HQ
            if abs(p_pos[0] - hq_pos[0]) + abs(p_pos[1] - hq_pos[1]) == 1:
                candidates.append(p)

        if not candidates:
            return TurnDecision()

        # Move to healthiest adjacent plantation
        target = max(candidates, key=lambda p: p.hp)
        log.info("Relocating HQ from %s to %s", hq_pos, target.position)
        return TurnDecision(relocate_main=[hq_pos, tuple(target.position)])


# ---------------------------------------------------------------------------
# Strategy pool and selector
# ---------------------------------------------------------------------------

ALL_STRATEGIES: list[Strategy] = [
    UpgradeStrategy(),
    RelocateHQStrategy(),
    RepairHQStrategy(),
    RepairStrategy(),
    ExpandStrategy(),
    BeaverSnipeStrategy(),
    SabotageStrategy(),
    BeaverAttackStrategy(),
]


def select_and_merge(state: GameState, bot: BotState) -> TurnDecision:
    """
    Run all strategies, collect decisions from those above priority threshold,
    merge commands into a single TurnDecision.
    Only one upgrade can be applied per turn.
    """
    bot.update_from_state(state)
    bot.update_beaver_history(state)

    scored = [(s.priority(state, bot), s) for s in ALL_STRATEGIES]
    scored.sort(key=lambda x: -x[0])

    merged = TurnDecision()
    relay_usage: dict[str, int] = {}

    for priority, strategy in scored:
        if priority <= 0:
            continue
        decision = strategy.decide(state, bot)

        # Apply upgrade from highest-priority strategy that has one
        if decision.upgrade and not merged.upgrade:
            merged.upgrade = decision.upgrade

        # Apply relocate from first strategy that requests it
        if decision.relocate_main and not merged.relocate_main:
            merged.relocate_main = decision.relocate_main

        # Merge commands
        for cmd in decision.commands:
            # Check if this command is a duplicate author
            author_already = any(
                c.author == cmd.author for c in merged.commands
            )
            if not author_already:
                merged.commands.append(cmd)

    return merged
