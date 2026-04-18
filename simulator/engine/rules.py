"""
Engine tick() — 15 phase handlers for a full game turn.
Order of phases follows the main game documentation; details calibrated via dump parity.
"""
import copy
from simulator.engine.state import EngineState
from simulator.engine.commands import CommandSet
from simulator.engine.events import UpgradeBought
from simulator.engine.types import UpgradeName


def tick(state: EngineState, cmds: CommandSet) -> EngineState:
    """Apply commands and advance state by one turn."""
    s = copy.deepcopy(state)
    s.events_this_turn = []

    _apply_upgrade_points(s)           # 1
    _apply_upgrade(s, cmds)            # 2
    _resolve_repair(s, cmds)           # 3
    _resolve_build(s, cmds)            # 4
    _resolve_sabotage(s, cmds)         # 5
    _resolve_beaver_attack(s, cmds)    # 6
    _resolve_relocate(s, cmds)         # 7
    _resolve_beaver_damage(s)          # 8
    _resolve_degradation(s)            # 9
    _resolve_meteo(s)                  # 10
    _resolve_terraform_and_score(s)    # 11
    _resolve_respawn(s)                # 12
    _spawn_new_meteo(s)                # 13
    _cleanup_dead(s)                   # 14
    _recompute_connectivity(s)         # 15

    s.turn_no += 1
    return s


# ---------- Phase handlers (stubs become real in later tasks) ----------

def _apply_upgrade_points(s: EngineState) -> None:
    if s.upgrades.points >= s.config.max_upgrade_points:
        return
    s.upgrades.turns_until_points -= 1
    if s.turn_no > 0 and s.upgrades.turns_until_points <= 0:
        s.upgrades.points = min(
            s.upgrades.points + 1, s.config.max_upgrade_points
        )
        s.upgrades.turns_until_points = s.config.upgrade_interval


def _apply_upgrade(s: EngineState, cmds: CommandSet) -> None:
    if cmds.upgrade is None or s.upgrades.points <= 0:
        return
    tier = s.upgrades.tiers.get(cmds.upgrade)
    if tier is None or tier.current >= tier.max:
        return
    tier.current += 1
    s.upgrades.points -= 1
    s.events_this_turn.append(
        UpgradeBought(name=cmds.upgrade.value, new_level=tier.current)
    )


def _resolve_repair(s: EngineState, cmds: CommandSet) -> None:
    from simulator.engine.validation import is_valid_command
    from simulator.engine.effects import RelayUsage, apply_relay_penalty

    pos_to_id = {p.position: pid for pid, p in s.plantations.items()}
    mhp = s.effective_mhp()
    rs = s.effective_rs()
    usage = RelayUsage()

    for cmd in cmds.commands:
        if not is_valid_command(s, cmd):
            continue
        target_id = pos_to_id.get(cmd.target)
        if target_id is None:
            continue  # not a repair — other phases handle this
        if target_id == pos_to_id.get(cmd.author):
            continue  # cannot repair self
        target = s.plantations[target_id]
        relay_id = pos_to_id[cmd.relay]
        heal = apply_relay_penalty(rs, relay_id, usage)
        target.hp = min(mhp, target.hp + heal)
def _resolve_build(s: EngineState, cmds: CommandSet) -> None: ...
def _resolve_sabotage(s: EngineState, cmds: CommandSet) -> None: ...
def _resolve_beaver_attack(s: EngineState, cmds: CommandSet) -> None: ...
def _resolve_relocate(s: EngineState, cmds: CommandSet) -> None: ...
def _resolve_beaver_damage(s: EngineState) -> None: ...
def _resolve_degradation(s: EngineState) -> None: ...
def _resolve_meteo(s: EngineState) -> None: ...
def _resolve_terraform_and_score(s: EngineState) -> None: ...
def _resolve_respawn(s: EngineState) -> None: ...
def _spawn_new_meteo(s: EngineState) -> None: ...
def _cleanup_dead(s: EngineState) -> None: ...
def _recompute_connectivity(s: EngineState) -> None: ...
