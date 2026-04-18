from models import GameState, Plantation, Command, Pos
from config import DEFAULT_AR, DEFAULT_SR
from typing import Optional


def in_range(pos1: Pos, pos2: Pos, radius: int) -> bool:
    """Square range check: |ΔX| <= R and |ΔY| <= R."""
    return abs(pos1[0] - pos2[0]) <= radius and abs(pos1[1] - pos2[1]) <= radius


def chebyshev(pos1: Pos, pos2: Pos) -> int:
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))


def manhattan(pos1: Pos, pos2: Pos) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_cells_in_range(center: Pos, radius: int, size: Pos) -> list[Pos]:
    """All valid map positions within square radius of center."""
    x, y = center
    result = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size[0] and 0 <= ny < size[1]:
                result.append((nx, ny))
    return result


def find_best_relay(
    state: GameState,
    author: Plantation,
    target: Pos,
    relay_usage: dict[str, int],
    ar: int = DEFAULT_AR,
    sr: int = DEFAULT_SR,
) -> Optional[Plantation]:
    """
    Find the best relay plantation for author to act on target.
    Author → relay (within SR of author) → target (within AR of relay).
    Prefer relays with fewer uses (lower efficiency penalty).
    """
    best: Optional[Plantation] = None
    best_penalty = 999

    author_pos = tuple(author.position)

    # Try using author as its own relay first (direct action)
    if in_range(author_pos, target, ar):
        penalty = relay_usage.get(author.id, 0)
        if penalty < best_penalty:
            best_penalty = penalty
            best = author

    # Try other connected plantations as relays
    for relay in state.connected_plantations:
        if relay.id == author.id:
            continue
        relay_pos = tuple(relay.position)
        if in_range(author_pos, relay_pos, sr) and in_range(relay_pos, target, ar):
            penalty = relay_usage.get(relay.id, 0)
            if penalty < best_penalty:
                best_penalty = penalty
                best = relay

    return best


def build_command(
    author: Plantation,
    relay: Plantation,
    target: Pos,
    relay_usage: dict[str, int],
) -> Command:
    relay_usage[relay.id] = relay_usage.get(relay.id, 0) + 1
    return Command(
        author=tuple(author.position),
        relay=tuple(relay.position),
        target=target,
    )


def effective_cs(base_cs: int, relay_uses_before: int) -> int:
    return max(0, base_cs - relay_uses_before)


def find_expansion_targets(
    state: GameState,
    plantation: Plantation,
    ar: int = DEFAULT_AR,
) -> list[tuple[Pos, float]]:
    """
    Find free cells within AR of plantation that can be built on.
    Returns list of (pos, score) sorted by score descending.
    """
    targets = []
    pos = tuple(plantation.position)
    for cell_pos in get_cells_in_range(pos, ar, state.size):
        if not state.is_cell_free(cell_pos):
            continue
        score = _expansion_score(state, cell_pos)
        targets.append((cell_pos, score))
    targets.sort(key=lambda x: -x[1])
    return targets


def _expansion_score(state: GameState, pos: Pos) -> float:
    score = 0.0
    # Boosted cell bonus
    if state.is_boosted_cell(pos):
        score += 1000.0
    # Prefer cells away from beavers (avoid their AR=2 attack range)
    for b in state.beavers:
        if in_range(tuple(b.position), pos, 2):
            score -= 500.0
    # Prefer cells that connect to existing network (adjacent to existing plantation)
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nb = (pos[0] + dx, pos[1] + dy)
        if nb in state.plantation_map:
            score += 100.0
    return score


def find_repair_targets(
    state: GameState,
    plantation: Plantation,
    mhp: int,
    threshold: float,
    ar: int = DEFAULT_AR,
    sr: int = DEFAULT_SR,
) -> list[Plantation]:
    """Find own plantations within reach that need repair."""
    targets = []
    author_pos = tuple(plantation.position)
    for p in state.connected_plantations:
        if p.id == plantation.id:
            continue
        if p.hp >= mhp * threshold:
            continue
        target_pos = tuple(p.position)
        # Direct or via relay
        if in_range(author_pos, target_pos, ar):
            targets.append(p)
            continue
        for relay in state.connected_plantations:
            relay_pos = tuple(relay.position)
            if (in_range(author_pos, relay_pos, sr)
                    and in_range(relay_pos, target_pos, ar)):
                targets.append(p)
                break
    # Sort: most damaged first; HQ always first
    targets.sort(key=lambda p: (0 if p.is_main else 1, p.hp))
    return targets


def assess_enemy_strength(state: GameState) -> dict[str, dict]:
    """
    Group visible enemy plantations by proximity clusters.
    Returns dict keyed by approximate cluster center with stats.
    """
    enemies = state.enemies
    if not enemies:
        return {}

    clusters: dict[str, dict] = {}
    for e in enemies:
        key = f"{e.position[0]//10}_{e.position[1]//10}"
        if key not in clusters:
            clusters[key] = {"count": 0, "total_hp": 0, "min_hp": 9999, "positions": []}
        c = clusters[key]
        c["count"] += 1
        c["total_hp"] += e.hp
        c["min_hp"] = min(c["min_hp"], e.hp)
        c["positions"].append(tuple(e.position))

    for key, c in clusters.items():
        c["avg_hp"] = c["total_hp"] / c["count"]
        # Strength score: more plantations + higher HP = stronger
        c["strength"] = c["count"] * c["avg_hp"]

    return clusters


def find_sabotage_targets(
    state: GameState,
    plantation: Plantation,
    ar: int = DEFAULT_AR,
    sr: int = DEFAULT_SR,
) -> list[Enemy]:
    """Find reachable enemy plantations sorted by threat/value."""
    author_pos = tuple(plantation.position)
    reachable = []
    for e in state.enemies:
        e_pos = tuple(e.position)
        # Must not be under construction (we can't attack constructions)
        # Check direct or via relay
        if in_range(author_pos, e_pos, ar):
            reachable.append(e)
            continue
        for relay in state.connected_plantations:
            relay_pos = tuple(relay.position)
            if (in_range(author_pos, relay_pos, sr)
                    and in_range(relay_pos, e_pos, ar)):
                reachable.append(e)
                break
    # Sort: low HP first (easy kills), then by proximity
    reachable.sort(key=lambda e: (e.hp, chebyshev(author_pos, tuple(e.position))))
    return reachable


def find_beaver_targets(
    state: GameState,
    plantation: Plantation,
    ar: int = DEFAULT_AR,
    sr: int = DEFAULT_SR,
) -> list:
    """Find reachable beaver lairs."""
    author_pos = tuple(plantation.position)
    reachable = []
    for b in state.beavers:
        b_pos = tuple(b.position)
        if in_range(author_pos, b_pos, ar):
            reachable.append(b)
            continue
        for relay in state.connected_plantations:
            relay_pos = tuple(relay.position)
            if (in_range(author_pos, relay_pos, sr)
                    and in_range(relay_pos, b_pos, ar)):
                reachable.append(b)
                break
    reachable.sort(key=lambda b: b.hp)
    return reachable


def is_storm_approaching(state: GameState, pos: Pos, turns_ahead: int = 5) -> bool:
    """Check if a sandstorm will hit the given position within N turns."""
    for f in state.meteo_forecasts:
        if f.kind != "sandstorm" or f.position is None:
            continue
        if not f.forming and f.radius and in_range(tuple(f.position), pos, f.radius):
            return True
        if f.next_position and f.radius and in_range(tuple(f.next_position), pos, f.radius):
            return True
    return False


def earthquake_incoming(state: GameState, turns_ahead: int = 2) -> bool:
    for f in state.meteo_forecasts:
        if f.kind == "earthquake" and f.turns_until <= turns_ahead:
            return True
    return False
