import math
from collections import deque
from typing import Iterator, Optional
from simulator.engine.types import Pos, PlantID


def chebyshev(a: Pos, b: Pos) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Pos, b: Pos) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def in_square_range(center: Pos, target: Pos, radius: int) -> bool:
    """Chebyshev <= radius. Used for AR/SR/VR."""
    return chebyshev(center, target) <= radius


def in_disc(center: Pos, target: Pos, radius: float) -> bool:
    """Euclidean <= radius. Used for sandstorms."""
    return euclidean(center, target) <= radius


def iter_square_range(center: Pos, radius: int, map_size: tuple[int, int]) -> Iterator[Pos]:
    """Yield cells in a square of `radius` around `center`, clamped to the map."""
    cx, cy = center
    w, h = map_size
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h:
                yield (x, y)


def bfs_connected(hq_id: Optional[PlantID], plantations: dict[PlantID, Pos]) -> set[PlantID]:
    """BFS from HQ via cardinal adjacency (no diagonals).
    Returns the set of connected plantation ids, including hq_id itself."""
    if hq_id is None or hq_id not in plantations:
        return set()
    pos_to_id = {pos: pid for pid, pos in plantations.items()}
    start = plantations[hq_id]
    visited: set[PlantID] = {hq_id}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nb = (cur[0] + dx, cur[1] + dy)
            if nb in pos_to_id and pos_to_id[nb] not in visited:
                visited.add(pos_to_id[nb])
                queue.append(nb)
    return visited
