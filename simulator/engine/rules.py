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
        if target_id not in s.connected_ids:
            continue  # cannot repair enemy
        if target_id == pos_to_id.get(cmd.author):
            continue  # cannot repair self
        target = s.plantations[target_id]
        relay_id = pos_to_id[cmd.relay]
        heal = apply_relay_penalty(rs, relay_id, usage)
        target.hp = min(mhp, target.hp + heal)


def _resolve_build(s: EngineState, cmds: CommandSet) -> None:
    from simulator.engine.validation import is_valid_command
    from simulator.engine.effects import RelayUsage, apply_relay_penalty
    from simulator.engine.entities import Construction, Plantation
    from simulator.engine.events import ConstructionCompleted

    pos_to_id = {p.position: pid for pid, p in s.plantations.items()}
    usage = RelayUsage()
    cs_base = s.config.cs

    for cmd in cmds.commands:
        if not is_valid_command(s, cmd):
            continue
        target = cmd.target
        # Disallow build on: mountain, own plantation, beaver, off-map
        if target in s.mountains:
            continue
        if target in pos_to_id:
            continue
        if not (0 <= target[0] < s.config.map_width and 0 <= target[1] < s.config.map_height):
            continue
        if any(b.position == target for b in s.beavers.values()):
            continue
        relay_id = pos_to_id.get(cmd.relay)
        if relay_id is None:
            continue
        cs_eff = apply_relay_penalty(cs_base, relay_id, usage)
        # Overflow check: new construction (first progress tick) counts against limit
        is_new_construction = target not in s.constructions
        if is_new_construction:
            current_total = len(s.plantations) + sum(
                1 for c in s.constructions.values() if c.progress > 0
            )
            if current_total >= s.effective_plantation_limit():
                _remove_oldest_plantation(s)
        if target not in s.constructions:
            s.constructions[target] = Construction(
                position=target, progress=0,
                last_progress_turn=s.turn_no,
                contributors_last_turn={},
            )
        con = s.constructions[target]
        con.progress += cs_eff
        con.last_progress_turn = s.turn_no
        author_id = pos_to_id.get(cmd.author, "?")
        con.contributors_last_turn[author_id] = (
            con.contributors_last_turn.get(author_id, 0) + cs_eff
        )

    # Finish constructions that reached 50
    done_positions = [pos for pos, c in s.constructions.items() if c.progress >= 50]
    for pos in done_positions:
        del s.constructions[pos]
        new_id = s.new_plant_id()
        s.plantations[new_id] = Plantation(
            id=new_id, position=pos, is_main=False,
            hp=s.effective_mhp(),
            immunity_until_turn=s.turn_no + s.config.immunity_turns + 1,
            built_on_turn=s.turn_no,
        )
        s.events_this_turn.append(
            ConstructionCompleted(position=pos, new_plantation_id=new_id)
        )
def _remove_oldest_plantation(s: EngineState) -> None:
    """Remove the oldest plantation (min built_on_turn).
    If it's the HQ, sets hq_id=None and applies 5% score penalty;
    respawn phase handles recreation on next tick."""
    from simulator.engine.events import PlantationDestroyed, HQLost
    if not s.plantations:
        return
    min_built = min(p.built_on_turn for p in s.plantations.values())
    oldest = [pid for pid, p in s.plantations.items() if p.built_on_turn == min_built]
    idx = int(s.rng.integers(0, len(oldest)))
    victim_id = oldest[idx]
    victim = s.plantations.pop(victim_id)
    was_main = victim.is_main
    s.events_this_turn.append(
        PlantationDestroyed(plantation_id=victim_id, position=victim.position,
                            was_main=was_main)
    )
    if was_main:
        penalty = max(1, s.score * 5 // 100)
        s.score -= penalty
        s.events_this_turn.append(HQLost(prev_hq_id=victim_id, penalty=penalty))
        s.hq_id = None


def _resolve_sabotage(s: EngineState, cmds: CommandSet) -> None:
    from simulator.engine.validation import is_valid_command
    from simulator.engine.effects import RelayUsage, apply_relay_penalty
    from simulator.engine.events import PlantationDestroyed, Scored

    pos_to_id = {p.position: pid for pid, p in s.plantations.items()}
    usage = RelayUsage()
    se_base = s.config.se

    for cmd in cmds.commands:
        if not is_valid_command(s, cmd):
            continue
        target_id = pos_to_id.get(cmd.target)
        if target_id is None or target_id not in s.enemy_ids:
            continue
        enemy = s.plantations[target_id]
        if enemy.immunity_until_turn > s.turn_no:
            continue
        relay_id = pos_to_id[cmd.relay]
        dmg = apply_relay_penalty(se_base, relay_id, usage)
        enemy.hp -= dmg
        if enemy.hp <= 0:
            reward = s.cell_reward(enemy.position)
            s.score += reward
            s.events_this_turn.append(Scored(amount=reward, source="sabotage"))
            s.events_this_turn.append(
                PlantationDestroyed(plantation_id=target_id,
                                    position=enemy.position, was_main=False)
            )
            del s.plantations[target_id]
            s.enemy_ids.discard(target_id)
def _resolve_beaver_attack(s: EngineState, cmds: CommandSet) -> None:
    from simulator.engine.validation import is_valid_command
    from simulator.engine.effects import RelayUsage, apply_relay_penalty
    from simulator.engine.events import BeaverKilled, Scored

    pos_to_beaver = {b.position: bid for bid, b in s.beavers.items()}
    pos_to_id = {p.position: pid for pid, p in s.plantations.items()}
    usage = RelayUsage()
    be_base = s.config.be

    for cmd in cmds.commands:
        if not is_valid_command(s, cmd):
            continue
        bid = pos_to_beaver.get(cmd.target)
        if bid is None:
            continue
        b = s.beavers[bid]
        relay_id = pos_to_id.get(cmd.relay, "?")
        dmg = apply_relay_penalty(be_base, relay_id, usage)
        b.hp -= dmg
        if b.hp <= 0:
            reward = s.cell_reward(b.position) * s.config.beaver_kill_multiplier
            s.score += reward
            s.events_this_turn.append(Scored(amount=reward, source="beaver"))
            s.events_this_turn.append(
                BeaverKilled(beaver_id=bid, position=b.position, reward=reward)
            )
            del s.beavers[bid]
def _resolve_relocate(s: EngineState, cmds: CommandSet) -> None:
    from simulator.engine.spatial import manhattan

    if not cmds.relocate or len(cmds.relocate) < 2:
        return
    src, dst = cmds.relocate[0], cmds.relocate[-1]
    if manhattan(src, dst) != 1:
        return  # only orthogonal adjacent
    pos_to_id = {p.position: pid for pid, p in s.plantations.items()}
    src_id = pos_to_id.get(src)
    dst_id = pos_to_id.get(dst)
    if src_id is None or dst_id is None:
        return
    if src_id != s.hq_id:
        return  # HQ must be at src
    if dst_id not in s.connected_ids:
        return
    s.plantations[src_id].is_main = False
    s.plantations[dst_id].is_main = True
    s.hq_id = dst_id
def _resolve_beaver_damage(s: EngineState) -> None:
    from simulator.engine.spatial import in_square_range

    dmg = s.effective_beaver_damage()
    ar = s.config.beaver_attack_radius
    for b in s.beavers.values():
        for p in s.plantations.values():
            if in_square_range(b.position, p.position, ar):
                if p.immunity_until_turn > s.turn_no:
                    continue
                p.hp -= dmg
        for pos, c in list(s.constructions.items()):
            if in_square_range(b.position, pos, ar):
                c.progress = max(0, c.progress - dmg)
    # Regeneration
    for b in s.beavers.values():
        b.hp = min(s.config.beaver_max_hp, b.hp + s.config.beaver_regen)
def _resolve_degradation(s: EngineState) -> None:
    ds = s.effective_ds()
    # Isolated plantations lose HP
    for pid, p in s.plantations.items():
        if pid == s.hq_id:
            continue
        if p.immunity_until_turn > s.turn_no:
            continue  # Skip immune plantations
        if pid not in s.connected_ids and pid not in s.enemy_ids:
            p.hp -= ds
    # Constructions without progress this turn
    for pos, c in list(s.constructions.items()):
        if c.last_progress_turn < s.turn_no:
            c.progress = max(0, c.progress - ds)
def _resolve_meteo(s: EngineState) -> None: ...
def _resolve_terraform_and_score(s: EngineState) -> None: ...
def _resolve_respawn(s: EngineState) -> None: ...
def _spawn_new_meteo(s: EngineState) -> None: ...
def _cleanup_dead(s: EngineState) -> None: ...
def _recompute_connectivity(s: EngineState) -> None: ...
