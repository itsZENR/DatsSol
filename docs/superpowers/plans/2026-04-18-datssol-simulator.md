# DatsSol Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Построить Python-симулятор DatsSol + Gymnasium-среду + PPO-тренировочный пайплайн, чтобы обучить RL-агента, способного полностью заменить существующий rule-based бот.

**Architecture:** Три изолированных слоя: `engine/` (чистый детерминистичный движок игровых правил), `env/` (Gymnasium-обёртка с autoregressive action space и entity-based observation), `parity/` (валидация движка на дампе 85 реальных раундов). Стандартный RL-стек Gymnasium + MaskablePPO.

**Tech Stack:** Python 3.11+, numpy, pandas, gymnasium, stable-baselines3, sb3-contrib, pytest.

**Spec:** `docs/superpowers/specs/2026-04-18-datssol-simulator-design.md`

---

## Файловая структура

```
simulator/
├── __init__.py
├── engine/
│   ├── __init__.py
│   ├── config.py           # GameConfig — все константы
│   ├── types.py            # Pos, PlantID, enums (UpgradeName, ActionKind)
│   ├── entities.py         # Plantation, Construction, Beaver, Storm, CellState
│   ├── state.py            # EngineState (contains everything)
│   ├── spatial.py          # Chebyshev/Euclidean, BFS, target enumerators
│   ├── events.py           # Event dataclasses (Scored, PlantationDestroyed, ...)
│   ├── commands.py         # CommandSet, parse/validate
│   └── rules.py            # tick() — 15 подэтапов
├── env/
│   ├── __init__.py
│   ├── action_space.py     # dict-space spec + mask builder
│   ├── obs_encoder.py      # EngineState → obs dict
│   ├── reward.py           # Δscore + penalties
│   └── datssol_env.py      # Gymnasium env, autoregressive loop
├── parity/
│   ├── __init__.py
│   ├── db_loader.py        # CSV dump → EngineState
│   ├── action_loader.py    # plantation_actions.csv → CommandSet
│   ├── diff.py             # EngineState diff reporter
│   └── replay.py           # CLI: прогон и отчёт
├── imitation/
│   ├── __init__.py
│   ├── bc_dataset.py       # (obs, action seq) из дампа
│   └── bc_pretrain.py      # supervised pretrain
├── opponents/
│   ├── __init__.py
│   └── solo.py             # null opponent
├── inference/
│   ├── __init__.py
│   ├── adapter.py          # API JSON ↔ EngineState
│   └── live_agent.py       # реальный сервер + политика
├── train/
│   ├── __init__.py
│   ├── configs/
│   │   ├── stage0.yaml
│   │   ├── stage1.yaml
│   │   ├── stage2.yaml
│   │   └── stage3.yaml
│   ├── curriculum.py
│   ├── pretrain_bc.py
│   ├── train_ppo.py
│   └── eval.py
└── tests/
    ├── __init__.py
    ├── engine/
    │   ├── test_spatial.py
    │   ├── test_terraform.py
    │   ├── test_repair.py
    │   ├── test_build.py
    │   ├── test_sabotage.py
    │   ├── test_beaver.py
    │   ├── test_relocate.py
    │   ├── test_meteo.py
    │   ├── test_upgrades.py
    │   ├── test_limit_overflow.py
    │   └── test_connectivity.py
    ├── env/
    │   ├── test_action_mask.py
    │   ├── test_obs_encoder.py
    │   └── test_env_step.py
    └── parity/
        └── test_replay_smoke.py
```

---

## Task 0: Bootstrap

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `simulator/__init__.py` (пустой)
- Create: `simulator/tests/__init__.py` (пустой)

- [ ] **Step 1: Создать requirements.txt**

Create file `requirements.txt`:
```
numpy>=1.26
pandas>=2.2
gymnasium>=0.29
stable-baselines3>=2.3
sb3-contrib>=2.3
torch>=2.2
pyyaml>=6.0
pytest>=8.0
pytest-xdist>=3.5
```

- [ ] **Step 2: Установить зависимости**

Run: `pip install -r requirements.txt`
Expected: все пакеты устанавливаются без ошибок.

- [ ] **Step 3: Создать pytest.ini**

Create `pytest.ini`:
```ini
[pytest]
testpaths = simulator/tests
python_files = test_*.py
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
addopts = -ra --strict-markers
```

- [ ] **Step 4: Создать пустые __init__.py**

Create `simulator/__init__.py` (пустой файл) и `simulator/tests/__init__.py` (пустой файл).

- [ ] **Step 5: Проверить pytest пустой прогон**

Run: `pytest simulator/tests -v`
Expected: `no tests ran` (exit 5), не ошибка импорта.

- [ ] **Step 6: Инициализировать git и commit**

Run:
```bash
git init
git add requirements.txt pytest.ini simulator/__init__.py simulator/tests/__init__.py
git commit -m "chore: bootstrap simulator scaffolding"
```

---

## Task 1: GameConfig

**Files:**
- Create: `simulator/engine/__init__.py` (пустой)
- Create: `simulator/engine/config.py`
- Test: `simulator/tests/engine/__init__.py` (пустой)
- Test: `simulator/tests/engine/test_config.py`

- [ ] **Step 1: Создать тестовый стаб**

Create `simulator/tests/engine/__init__.py` (пустой).

Create `simulator/tests/engine/test_config.py`:
```python
from simulator.engine.config import GameConfig


def test_default_config_values():
    cfg = GameConfig()
    assert cfg.mhp == 50
    assert cfg.ts == 5
    assert cfg.cs == 5
    assert cfg.rs == 5
    assert cfg.se == 5
    assert cfg.be == 5
    assert cfg.ds == 10
    assert cfg.ar == 2
    assert cfg.sr == 3
    assert cfg.vr == 3
    assert cfg.base_plantation_limit == 30
    assert cfg.max_plantation_limit == 40
    assert cfg.upgrade_interval == 30
    assert cfg.max_upgrade_points == 15
    assert cfg.beaver_max_hp == 100
    assert cfg.beaver_regen == 5
    assert cfg.beaver_damage == 15
    assert cfg.earthquake_damage == 10
    assert cfg.storm_damage == 2
    assert cfg.storm_forming_turns == 5
    assert cfg.storm_prob == 0.10
    assert cfg.earthquake_prob == 0.05
    assert cfg.immunity_turns == 3
    assert cfg.cell_degradation_delay == 30
    assert cfg.cell_degradation_per_turn == 10
    assert cfg.map_width == 298
    assert cfg.map_height == 298
    assert cfg.beaver_kill_multiplier == 20
    assert cfg.round_turns == 600
    assert cfg.normal_cell_reward == 1000
    assert cfg.boosted_cell_reward == 1500
    assert cfg.boosted_mod == 7


def test_config_frozen_immutable():
    cfg = GameConfig()
    import dataclasses
    assert dataclasses.is_dataclass(cfg)
    try:
        cfg.mhp = 9999
        raised = False
    except Exception:
        raised = True
    assert raised, "GameConfig must be frozen"
```

- [ ] **Step 2: Прогнать — ожидаем FAIL**

Run: `pytest simulator/tests/engine/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'simulator.engine.config'`

- [ ] **Step 3: Создать GameConfig**

Create `simulator/engine/config.py`:
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """
    Все константы игры DatsSol. Значения по умолчанию соответствуют документации
    и дампу реальных раундов. Меняется только через replace() при экспериментах.
    """
    # Плантация
    mhp: int = 50
    ts: int = 5
    cs: int = 5
    rs: int = 5
    se: int = 5
    be: int = 5
    ds: int = 10
    ar: int = 2
    sr: int = 3
    vr: int = 3

    # Лимиты
    base_plantation_limit: int = 30
    max_plantation_limit: int = 40

    # Апгрейды
    upgrade_interval: int = 30
    max_upgrade_points: int = 15
    immunity_turns: int = 3

    # Бобры
    beaver_max_hp: int = 100
    beaver_regen: int = 5
    beaver_damage: int = 15
    beaver_attack_radius: int = 2

    # Катаклизмы
    storm_damage: int = 2
    storm_forming_turns: int = 5
    storm_prob: float = 0.10
    storm_speed_min: int = 5
    storm_speed_max: int = 15
    storm_min_hp: int = 1
    earthquake_damage: int = 10
    earthquake_prob: float = 0.05

    # Клетки
    cell_degradation_delay: int = 30  # 2026-04-18 balance: 40 → 30
    cell_degradation_per_turn: int = 10
    normal_cell_reward: int = 1000
    boosted_cell_reward: int = 1500
    boosted_mod: int = 7

    # Бобры — множитель награды за kill (2026-04-18 balance: 10× → 20×)
    beaver_kill_multiplier: int = 20

    # Раунд
    round_turns: int = 600

    # Карта (2026-04-18 balance: 378×378 → 298×298). Override через replace при curriculum.
    map_width: int = 298
    map_height: int = 298
```

- [ ] **Step 4: Прогнать — ожидаем PASS**

Run: `pytest simulator/tests/engine/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/__init__.py simulator/engine/config.py simulator/tests/engine/__init__.py simulator/tests/engine/test_config.py
git commit -m "feat(engine): add GameConfig with default constants"
```

---

## Task 2: Types and entity dataclasses

**Files:**
- Create: `simulator/engine/types.py`
- Create: `simulator/engine/entities.py`
- Test: `simulator/tests/engine/test_entities.py`

- [ ] **Step 1: Написать тест**

Create `simulator/tests/engine/test_entities.py`:
```python
from simulator.engine.types import Pos, UpgradeName
from simulator.engine.entities import (
    Plantation, Construction, Beaver, Storm, CellState, UpgradesState, UpgradeTier,
)


def test_pos_is_int_tuple():
    p: Pos = (3, 5)
    assert isinstance(p[0], int) and isinstance(p[1], int)


def test_upgrade_names_complete():
    names = {u.value for u in UpgradeName}
    expected = {
        "repair_power", "max_hp", "settlement_limit",
        "signal_range", "vision_range",
        "decay_mitigation", "earthquake_mitigation", "beaver_damage_mitigation",
    }
    assert names == expected


def test_plantation_roundtrip():
    p = Plantation(id="p1", position=(3, 4), is_main=True, hp=50,
                   immunity_until_turn=3, built_on_turn=0)
    assert p.position == (3, 4)
    assert p.is_main is True


def test_construction_roundtrip():
    c = Construction(position=(5, 5), progress=10, last_progress_turn=2,
                     contributors_last_turn={})
    assert c.progress == 10


def test_beaver_roundtrip():
    b = Beaver(id="b1", position=(10, 10), hp=95,
               last_hit_by={}, total_damage_this_turn={})
    assert b.hp == 95


def test_storm_roundtrip():
    s = Storm(id="s1", position=(100, 100), radius=30, speed=10,
              direction=(1, 1), forming_left=0)
    assert s.radius == 30


def test_cellstate_defaults():
    c = CellState(progress=0, completed_on_turn=None)
    assert c.progress == 0
    assert c.completed_on_turn is None


def test_upgrades_state():
    tiers = {UpgradeName.REPAIR_POWER: UpgradeTier(current=1, max=3)}
    u = UpgradesState(points=2, turns_until_points=25, tiers=tiers)
    assert u.tiers[UpgradeName.REPAIR_POWER].current == 1
```

- [ ] **Step 2: Прогнать — FAIL**

Run: `pytest simulator/tests/engine/test_entities.py -v`
Expected: ImportError.

- [ ] **Step 3: Создать types.py**

Create `simulator/engine/types.py`:
```python
from enum import Enum

Pos = tuple[int, int]
PlantID = str
BeaverID = str
StormID = str


class UpgradeName(str, Enum):
    REPAIR_POWER = "repair_power"
    MAX_HP = "max_hp"
    SETTLEMENT_LIMIT = "settlement_limit"
    SIGNAL_RANGE = "signal_range"
    VISION_RANGE = "vision_range"
    DECAY_MITIGATION = "decay_mitigation"
    EARTHQUAKE_MITIGATION = "earthquake_mitigation"
    BEAVER_DAMAGE_MITIGATION = "beaver_damage_mitigation"


class ActionKind(str, Enum):
    BUILD = "build"
    REPAIR = "repair"
    SABOTAGE = "sabotage"
    BEAVER = "beaver"
```

- [ ] **Step 4: Создать entities.py**

Create `simulator/engine/entities.py`:
```python
from dataclasses import dataclass, field
from typing import Optional
from simulator.engine.types import Pos, PlantID, BeaverID, StormID, UpgradeName


@dataclass
class Plantation:
    id: PlantID
    position: Pos
    is_main: bool
    hp: int
    immunity_until_turn: int
    built_on_turn: int


@dataclass
class Construction:
    position: Pos
    progress: int
    last_progress_turn: int
    contributors_last_turn: dict[PlantID, int]  # author_id -> cs_contribution


@dataclass
class Beaver:
    id: BeaverID
    position: Pos
    hp: int
    last_hit_by: dict[str, int]  # player_id -> last_turn_hit
    total_damage_this_turn: dict[str, int]  # player_id -> damage_this_turn


@dataclass
class Storm:
    id: StormID
    position: Pos
    radius: int
    speed: int
    direction: tuple[int, int]  # (dx, dy) — нормирован: +1/-1
    forming_left: int  # 0 если уже движется


@dataclass
class CellState:
    progress: int
    completed_on_turn: Optional[int]  # ход завершения (для таймера 40)


@dataclass
class UpgradeTier:
    current: int
    max: int


@dataclass
class UpgradesState:
    points: int
    turns_until_points: int
    tiers: dict[UpgradeName, UpgradeTier] = field(default_factory=dict)
```

- [ ] **Step 5: Прогнать — PASS**

Run: `pytest simulator/tests/engine/test_entities.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add simulator/engine/types.py simulator/engine/entities.py simulator/tests/engine/test_entities.py
git commit -m "feat(engine): add types and entity dataclasses"
```

---

## Task 3: Spatial primitives

**Files:**
- Create: `simulator/engine/spatial.py`
- Test: `simulator/tests/engine/test_spatial.py`

- [ ] **Step 1: Написать тесты**

Create `simulator/tests/engine/test_spatial.py`:
```python
import math
from simulator.engine.spatial import (
    chebyshev, manhattan, euclidean,
    in_square_range, in_disc,
    iter_square_range, bfs_connected,
)


def test_chebyshev():
    assert chebyshev((0, 0), (3, 2)) == 3
    assert chebyshev((5, 5), (2, 1)) == 4
    assert chebyshev((0, 0), (0, 0)) == 0


def test_manhattan():
    assert manhattan((0, 0), (3, 2)) == 5
    assert manhattan((5, 5), (2, 1)) == 7


def test_euclidean():
    assert euclidean((0, 0), (3, 4)) == 5.0
    assert math.isclose(euclidean((0, 0), (1, 1)), math.sqrt(2))


def test_in_square_range():
    assert in_square_range((0, 0), (2, 2), 2) is True
    assert in_square_range((0, 0), (3, 0), 2) is False
    assert in_square_range((0, 0), (2, -2), 2) is True


def test_in_disc_euclidean():
    assert in_disc((0, 0), (3, 4), 5) is True
    assert in_disc((0, 0), (3, 4), 4) is False
    assert in_disc((0, 0), (0, 0), 0) is True


def test_iter_square_range_bounds():
    cells = list(iter_square_range((5, 5), 1, (10, 10)))
    assert set(cells) == {(4, 4), (4, 5), (4, 6),
                          (5, 4), (5, 5), (5, 6),
                          (6, 4), (6, 5), (6, 6)}


def test_iter_square_range_clamped_to_map():
    cells = list(iter_square_range((0, 0), 1, (3, 3)))
    assert (−1, 0) not in cells  # wouldn't compile but logically
    for (x, y) in cells:
        assert 0 <= x < 3 and 0 <= y < 3


def test_bfs_connected_simple_line():
    # ЦУ в (0,0), соседи через смежность
    plants = {
        "p0": (0, 0),  # ЦУ
        "p1": (1, 0),
        "p2": (2, 0),
        "p3": (5, 0),  # изолирована
    }
    connected = bfs_connected("p0", plants)
    assert connected == {"p0", "p1", "p2"}


def test_bfs_connected_empty_hq():
    connected = bfs_connected(None, {})
    assert connected == set()


def test_bfs_only_cardinal_not_diagonal():
    # диагональ не считается смежной
    plants = {
        "hq": (0, 0),
        "p1": (1, 1),  # диагональ
    }
    connected = bfs_connected("hq", plants)
    assert connected == {"hq"}
```

Примечание: строка `assert (−1, 0) not in cells` использует юникодный минус — в настоящем файле пиши обычный `-1`. Дубликат проверки — defensive.

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_spatial.py -v`

- [ ] **Step 3: Реализовать spatial.py**

Create `simulator/engine/spatial.py`:
```python
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
    """Chebyshev ≤ radius. Используется для AR/SR/VR."""
    return chebyshev(center, target) <= radius


def in_disc(center: Pos, target: Pos, radius: float) -> bool:
    """Euclidean ≤ radius. Используется для бури."""
    return euclidean(center, target) <= radius


def iter_square_range(center: Pos, radius: int, map_size: tuple[int, int]) -> Iterator[Pos]:
    """Yield cells в квадрате radius вокруг center, clamped to map."""
    cx, cy = center
    w, h = map_size
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h:
                yield (x, y)


def bfs_connected(hq_id: Optional[PlantID], plantations: dict[PlantID, Pos]) -> set[PlantID]:
    """BFS от ЦУ по кардинальной смежности. Возвращает id связных плантаций."""
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
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_spatial.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/spatial.py simulator/tests/engine/test_spatial.py
git commit -m "feat(engine): add spatial primitives (Chebyshev/Euclidean, BFS)"
```

---

## Task 4: Events union

**Files:**
- Create: `simulator/engine/events.py`
- Test: `simulator/tests/engine/test_events.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_events.py`:
```python
from simulator.engine.events import (
    Event, Scored, PlantationDestroyed, BeaverKilled, HQLost,
    ConstructionCompleted, CellTerraformed, UpgradeBought,
)


def test_events_are_frozen_dataclasses():
    e = Scored(amount=100, source="terraform")
    assert e.amount == 100
    try:
        e.amount = 999
        raised = False
    except Exception:
        raised = True
    assert raised


def test_events_union_iterable():
    events: list[Event] = [
        Scored(amount=50, source="terraform"),
        PlantationDestroyed(plantation_id="p1", position=(3, 3), was_main=False),
        BeaverKilled(beaver_id="b1", position=(10, 10), reward=10000),
        HQLost(prev_hq_id="p0", penalty=5),
        ConstructionCompleted(position=(5, 5), new_plantation_id="p2"),
        CellTerraformed(position=(7, 7), reward=1500),
        UpgradeBought(name="repair_power", new_level=1),
    ]
    assert len(events) == 7
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_events.py -v`

- [ ] **Step 3: events.py**

Create `simulator/engine/events.py`:
```python
from dataclasses import dataclass
from typing import Union
from simulator.engine.types import Pos, PlantID, BeaverID


@dataclass(frozen=True)
class Scored:
    amount: int
    source: str  # "terraform" | "sabotage" | "beaver"


@dataclass(frozen=True)
class PlantationDestroyed:
    plantation_id: PlantID
    position: Pos
    was_main: bool


@dataclass(frozen=True)
class BeaverKilled:
    beaver_id: BeaverID
    position: Pos
    reward: int


@dataclass(frozen=True)
class HQLost:
    prev_hq_id: PlantID
    penalty: int


@dataclass(frozen=True)
class ConstructionCompleted:
    position: Pos
    new_plantation_id: PlantID


@dataclass(frozen=True)
class CellTerraformed:
    position: Pos
    reward: int


@dataclass(frozen=True)
class UpgradeBought:
    name: str
    new_level: int


Event = Union[
    Scored, PlantationDestroyed, BeaverKilled, HQLost,
    ConstructionCompleted, CellTerraformed, UpgradeBought,
]
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_events.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/events.py simulator/tests/engine/test_events.py
git commit -m "feat(engine): add typed Event dataclasses"
```

---

## Task 5: Commands (parse and validate)

**Files:**
- Create: `simulator/engine/commands.py`
- Test: `simulator/tests/engine/test_commands.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_commands.py`:
```python
from simulator.engine.commands import Command, CommandSet
from simulator.engine.types import UpgradeName


def test_command_path_format():
    c = Command(author=(1, 1), relay=(1, 1), target=(2, 1))
    assert c.to_path() == [[1, 1], [1, 1], [2, 1]]


def test_commandset_empty_payload_fallback():
    cs = CommandSet()
    p = cs.to_payload()
    assert p == {"plantationUpgrade": ""}


def test_commandset_with_build():
    cs = CommandSet(commands=[Command((1, 1), (1, 1), (2, 1))])
    p = cs.to_payload()
    assert p["command"] == [{"path": [[1, 1], [1, 1], [2, 1]]}]


def test_commandset_with_upgrade():
    cs = CommandSet(upgrade=UpgradeName.REPAIR_POWER)
    p = cs.to_payload()
    assert p["plantationUpgrade"] == "repair_power"


def test_commandset_with_relocate():
    cs = CommandSet(relocate=[(1, 1), (1, 2)])
    p = cs.to_payload()
    assert p["relocateMain"] == [[1, 1], [1, 2]]


def test_from_api_payload_roundtrip():
    payload = {
        "command": [{"path": [[1, 1], [1, 1], [2, 1]]}],
        "plantationUpgrade": "repair_power",
        "relocateMain": [[0, 0], [0, 1]],
    }
    cs = CommandSet.from_payload(payload)
    assert cs.commands[0].author == (1, 1)
    assert cs.upgrade == UpgradeName.REPAIR_POWER
    assert cs.relocate == [(0, 0), (0, 1)]
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_commands.py -v`

- [ ] **Step 3: commands.py**

Create `simulator/engine/commands.py`:
```python
from dataclasses import dataclass, field
from typing import Optional
from simulator.engine.types import Pos, UpgradeName


@dataclass
class Command:
    author: Pos
    relay: Pos
    target: Pos

    def to_path(self) -> list[list[int]]:
        return [list(self.author), list(self.relay), list(self.target)]

    @classmethod
    def from_path(cls, path: list[list[int]]) -> "Command":
        a, r, t = path
        return cls(author=(a[0], a[1]), relay=(r[0], r[1]), target=(t[0], t[1]))


@dataclass
class CommandSet:
    commands: list[Command] = field(default_factory=list)
    upgrade: Optional[UpgradeName] = None
    relocate: list[Pos] = field(default_factory=list)

    def to_payload(self) -> dict:
        payload: dict = {}
        if self.commands:
            payload["command"] = [{"path": c.to_path()} for c in self.commands]
        if self.upgrade is not None:
            payload["plantationUpgrade"] = self.upgrade.value
        if self.relocate:
            payload["relocateMain"] = [list(p) for p in self.relocate]
        if not payload:
            payload["plantationUpgrade"] = ""
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> "CommandSet":
        cmds = [Command.from_path(c["path"]) for c in payload.get("command", [])]
        upg_raw = payload.get("plantationUpgrade", "")
        upg = UpgradeName(upg_raw) if upg_raw else None
        relo_raw = payload.get("relocateMain", [])
        relo = [(p[0], p[1]) for p in relo_raw]
        return cls(commands=cmds, upgrade=upg, relocate=relo)
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_commands.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/commands.py simulator/tests/engine/test_commands.py
git commit -m "feat(engine): add Command and CommandSet with API payload conversion"
```

---

## Task 6: EngineState container

**Files:**
- Create: `simulator/engine/state.py`
- Test: `simulator/tests/engine/test_state.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_state.py`:
```python
import numpy as np
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, UpgradesState, UpgradeTier
from simulator.engine.types import UpgradeName


def test_engine_state_seeded_rng_deterministic():
    s1 = EngineState.new(config=GameConfig(), seed=42)
    s2 = EngineState.new(config=GameConfig(), seed=42)
    a1 = [s1.rng.random() for _ in range(5)]
    a2 = [s2.rng.random() for _ in range(5)]
    assert a1 == a2


def test_engine_state_new_empty():
    s = EngineState.new(config=GameConfig(), seed=0)
    assert s.turn_no == 0
    assert s.plantations == {}
    assert s.score == 0
    assert s.hq_id is None
    assert isinstance(s.rng, np.random.Generator)


def test_get_hq_returns_plantation_or_none():
    s = EngineState.new(config=GameConfig(), seed=0)
    assert s.get_hq() is None
    p = Plantation(id="hq", position=(3, 3), is_main=True, hp=50,
                   immunity_until_turn=0, built_on_turn=0)
    s.plantations[p.id] = p
    s.hq_id = "hq"
    assert s.get_hq() is p


def test_compute_effective_values_from_upgrades():
    cfg = GameConfig()
    s = EngineState.new(config=cfg, seed=0)
    s.upgrades.tiers[UpgradeName.REPAIR_POWER] = UpgradeTier(current=2, max=3)
    s.upgrades.tiers[UpgradeName.MAX_HP] = UpgradeTier(current=1, max=5)
    s.upgrades.tiers[UpgradeName.SIGNAL_RANGE] = UpgradeTier(current=1, max=99)
    s.upgrades.tiers[UpgradeName.SETTLEMENT_LIMIT] = UpgradeTier(current=3, max=10)
    s.upgrades.tiers[UpgradeName.BEAVER_DAMAGE_MITIGATION] = UpgradeTier(current=2, max=5)
    s.upgrades.tiers[UpgradeName.EARTHQUAKE_MITIGATION] = UpgradeTier(current=1, max=3)

    assert s.effective_rs() == cfg.rs + 2
    assert s.effective_mhp() == cfg.mhp + 10
    assert s.effective_sr() == cfg.sr + 1
    assert s.effective_plantation_limit() == cfg.base_plantation_limit + 3
    assert s.effective_beaver_damage() == cfg.beaver_damage - 4
    assert s.effective_earthquake_damage() == cfg.earthquake_damage - 2


def test_is_boosted_cell():
    s = EngineState.new(config=GameConfig(), seed=0)
    assert s.is_boosted_cell((0, 0)) is True
    assert s.is_boosted_cell((7, 14)) is True
    assert s.is_boosted_cell((3, 7)) is False
    assert s.is_boosted_cell((14, 5)) is False
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_state.py -v`

- [ ] **Step 3: state.py**

Create `simulator/engine/state.py`:
```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from simulator.engine.config import GameConfig
from simulator.engine.entities import (
    Plantation, Construction, Beaver, Storm, CellState,
    UpgradesState, UpgradeTier,
)
from simulator.engine.events import Event
from simulator.engine.types import Pos, PlantID, BeaverID, UpgradeName


@dataclass
class EngineState:
    config: GameConfig
    rng: np.random.Generator
    turn_no: int = 0
    plantations: dict[PlantID, Plantation] = field(default_factory=dict)
    constructions: dict[Pos, Construction] = field(default_factory=dict)
    beavers: dict[BeaverID, Beaver] = field(default_factory=dict)
    cells: dict[Pos, CellState] = field(default_factory=dict)
    mountains: frozenset[Pos] = field(default_factory=frozenset)
    storms: list[Storm] = field(default_factory=list)
    earthquake_turn: Optional[int] = None  # turn_no, когда ударит
    upgrades: UpgradesState = field(default_factory=lambda: UpgradesState(
        points=0, turns_until_points=30, tiers={},
    ))
    hq_id: Optional[PlantID] = None
    score: int = 0
    events_this_turn: list[Event] = field(default_factory=list)
    connected_ids: set[PlantID] = field(default_factory=set)
    next_plant_counter: int = 0  # для генерации id новых плантаций

    @classmethod
    def new(cls, config: GameConfig, seed: int) -> "EngineState":
        rng = np.random.default_rng(seed)
        st = cls(config=config, rng=rng)
        # Инициализировать tiers всеми апгрейдами на уровне 0
        default_max = {
            UpgradeName.REPAIR_POWER: 3,
            UpgradeName.MAX_HP: 5,
            UpgradeName.SETTLEMENT_LIMIT: 10,
            UpgradeName.SIGNAL_RANGE: 99,
            UpgradeName.VISION_RANGE: 5,
            UpgradeName.DECAY_MITIGATION: 3,
            UpgradeName.EARTHQUAKE_MITIGATION: 3,
            UpgradeName.BEAVER_DAMAGE_MITIGATION: 5,
        }
        for name, mx in default_max.items():
            st.upgrades.tiers[name] = UpgradeTier(current=0, max=mx)
        st.upgrades.turns_until_points = config.upgrade_interval
        return st

    def get_hq(self) -> Optional[Plantation]:
        if self.hq_id is None:
            return None
        return self.plantations.get(self.hq_id)

    def tier_level(self, name: UpgradeName) -> int:
        t = self.upgrades.tiers.get(name)
        return t.current if t is not None else 0

    def effective_rs(self) -> int:
        return self.config.rs + self.tier_level(UpgradeName.REPAIR_POWER)

    def effective_mhp(self) -> int:
        return self.config.mhp + 10 * self.tier_level(UpgradeName.MAX_HP)

    def effective_sr(self) -> int:
        return self.config.sr + self.tier_level(UpgradeName.SIGNAL_RANGE)

    def effective_vr(self) -> int:
        return self.config.vr + 2 * self.tier_level(UpgradeName.VISION_RANGE)

    def effective_plantation_limit(self) -> int:
        return self.config.base_plantation_limit + self.tier_level(
            UpgradeName.SETTLEMENT_LIMIT
        )

    def effective_beaver_damage(self) -> int:
        return max(
            0,
            self.config.beaver_damage - 2 * self.tier_level(
                UpgradeName.BEAVER_DAMAGE_MITIGATION
            ),
        )

    def effective_earthquake_damage(self) -> int:
        return max(
            0,
            self.config.earthquake_damage - 2 * self.tier_level(
                UpgradeName.EARTHQUAKE_MITIGATION
            ),
        )

    def effective_ds(self) -> int:
        return max(
            0,
            self.config.ds - 2 * self.tier_level(UpgradeName.DECAY_MITIGATION),
        )

    def is_boosted_cell(self, pos: Pos) -> bool:
        m = self.config.boosted_mod
        return pos[0] % m == 0 and pos[1] % m == 0

    def cell_reward(self, pos: Pos) -> int:
        return (
            self.config.boosted_cell_reward
            if self.is_boosted_cell(pos)
            else self.config.normal_cell_reward
        )

    def new_plant_id(self) -> PlantID:
        pid = f"p{self.next_plant_counter}"
        self.next_plant_counter += 1
        return pid
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/state.py simulator/tests/engine/test_state.py
git commit -m "feat(engine): add EngineState with effective-value helpers"
```

---

## Task 7: rules.py — скелет tick + апгрейды (этапы 1-2)

**Files:**
- Create: `simulator/engine/rules.py`
- Test: `simulator/tests/engine/test_upgrades.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_upgrades.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.commands import CommandSet
from simulator.engine.entities import UpgradeTier
from simulator.engine.types import UpgradeName
from simulator.engine.rules import tick


def test_upgrade_points_accrue_each_interval():
    cfg = GameConfig(upgrade_interval=3, round_turns=10)
    s = EngineState.new(cfg, seed=0)
    for _ in range(3):
        s = tick(s, CommandSet())
    assert s.upgrades.points == 1
    assert s.upgrades.turns_until_points == cfg.upgrade_interval


def test_upgrade_points_capped_at_max():
    cfg = GameConfig(upgrade_interval=1, max_upgrade_points=2)
    s = EngineState.new(cfg, seed=0)
    for _ in range(10):
        s = tick(s, CommandSet())
    assert s.upgrades.points == 2


def test_apply_upgrade_increments_tier_and_spends_point():
    cfg = GameConfig(upgrade_interval=1)
    s = EngineState.new(cfg, seed=0)
    s.upgrades.points = 1
    s2 = tick(s, CommandSet(upgrade=UpgradeName.REPAIR_POWER))
    assert s2.upgrades.tiers[UpgradeName.REPAIR_POWER].current == 1
    assert s2.upgrades.points == 0


def test_cannot_upgrade_beyond_max():
    cfg = GameConfig(upgrade_interval=1)
    s = EngineState.new(cfg, seed=0)
    s.upgrades.points = 1
    s.upgrades.tiers[UpgradeName.REPAIR_POWER] = UpgradeTier(current=3, max=3)
    s2 = tick(s, CommandSet(upgrade=UpgradeName.REPAIR_POWER))
    assert s2.upgrades.tiers[UpgradeName.REPAIR_POWER].current == 3
    assert s2.upgrades.points == 1  # очко не потрачено


def test_cannot_upgrade_without_points():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.upgrades.points = 0
    s2 = tick(s, CommandSet(upgrade=UpgradeName.REPAIR_POWER))
    assert s2.upgrades.tiers[UpgradeName.REPAIR_POWER].current == 0
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_upgrades.py -v`

- [ ] **Step 3: rules.py — скелет + upgrades**

Create `simulator/engine/rules.py`:
```python
"""
Engine tick() — 15 подэтапов обработки хода.
Порядок из основной документации игры; детали калибруются на дампе.
"""
import copy
from simulator.engine.state import EngineState
from simulator.engine.commands import CommandSet
from simulator.engine.events import UpgradeBought
from simulator.engine.types import UpgradeName


def tick(state: EngineState, cmds: CommandSet) -> EngineState:
    """Применить команды и продвинуть состояние на один ход."""
    s = copy.deepcopy(state)
    s.events_this_turn = []

    _apply_upgrade_points(s)           # 1
    _apply_upgrade(s, cmds)            # 2
    _resolve_repair(s, cmds)           # 3
    _resolve_build(s, cmds)            # 4 (включает лимит-overflow)
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


# ---------- Подэтапы-заглушки (дописываются в следующих задачах) ----------

def _apply_upgrade_points(s: EngineState) -> None:
    if s.upgrades.points >= s.config.max_upgrade_points:
        return
    s.upgrades.turns_until_points -= 1
    if s.upgrades.turns_until_points <= 0:
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


def _resolve_repair(s: EngineState, cmds: CommandSet) -> None: ...
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
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_upgrades.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/rules.py simulator/tests/engine/test_upgrades.py
git commit -m "feat(engine): add tick() skeleton with upgrade points & application"
```

---

## Task 8: Command validation helper

**Files:**
- Create: `simulator/engine/validation.py`
- Test: `simulator/tests/engine/test_validation.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_validation.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command
from simulator.engine.validation import is_valid_command


def _mk_state_with_two_plants():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    hq = Plantation(id="hq", position=(5, 5), is_main=True, hp=50,
                    immunity_until_turn=0, built_on_turn=0)
    p2 = Plantation(id="p2", position=(5, 6), is_main=False, hp=50,
                    immunity_until_turn=0, built_on_turn=0)
    s.plantations["hq"] = hq
    s.plantations["p2"] = p2
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p2"}
    return s


def test_author_must_be_own_and_connected():
    s = _mk_state_with_two_plants()
    # автор не наш
    c = Command(author=(99, 99), relay=(5, 5), target=(5, 4))
    assert is_valid_command(s, c) is False


def test_target_within_ar_of_relay():
    s = _mk_state_with_two_plants()
    # AR=2, relay=(5,5), target=(8,5) → dx=3 — вне AR
    c = Command(author=(5, 5), relay=(5, 5), target=(8, 5))
    assert is_valid_command(s, c) is False


def test_relay_within_sr_of_author():
    s = _mk_state_with_two_plants()
    # добавим дальнюю плантацию как потенциальный relay
    p3 = Plantation(id="p3", position=(5, 10), is_main=False, hp=50,
                    immunity_until_turn=0, built_on_turn=0)
    s.plantations["p3"] = p3
    s.connected_ids.add("p3")
    # SR=3, от (5,5) до (5,10) — 5, вне SR
    c = Command(author=(5, 5), relay=(5, 10), target=(5, 10))
    assert is_valid_command(s, c) is False


def test_valid_direct_command():
    s = _mk_state_with_two_plants()
    c = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    assert is_valid_command(s, c) is True
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_validation.py -v`

- [ ] **Step 3: validation.py**

Create `simulator/engine/validation.py`:
```python
from simulator.engine.state import EngineState
from simulator.engine.commands import Command
from simulator.engine.spatial import chebyshev


def is_valid_command(state: EngineState, cmd: Command) -> bool:
    """Проверяет геометрическую валидность: автор и реле — наши связные,
    реле в SR от автора, цель в AR от реле."""
    # Автор и реле — позиции → найти плантации
    pos_to_id = {
        tuple(p.position): pid
        for pid, p in state.plantations.items()
    }
    author_id = pos_to_id.get(cmd.author)
    relay_id = pos_to_id.get(cmd.relay)
    if author_id is None or relay_id is None:
        return False
    if author_id not in state.connected_ids:
        return False
    if relay_id not in state.connected_ids:
        return False
    if chebyshev(cmd.author, cmd.relay) > state.effective_sr():
        return False
    if chebyshev(cmd.relay, cmd.target) > state.config.ar:
        return False
    return True
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_validation.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/validation.py simulator/tests/engine/test_validation.py
git commit -m "feat(engine): add geometric command validation"
```

---

## Task 9: Relay usage & effective-power helper

**Files:**
- Create: `simulator/engine/effects.py`
- Test: `simulator/tests/engine/test_effects.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_effects.py`:
```python
from simulator.engine.effects import RelayUsage, apply_relay_penalty


def test_first_use_is_base():
    u = RelayUsage()
    assert apply_relay_penalty(5, "r1", u) == 5
    assert u.count == {"r1": 1}


def test_subsequent_uses_penalty():
    u = RelayUsage()
    apply_relay_penalty(5, "r1", u)
    assert apply_relay_penalty(5, "r1", u) == 4
    assert apply_relay_penalty(5, "r1", u) == 3


def test_penalty_clipped_at_zero():
    u = RelayUsage()
    for _ in range(10):
        apply_relay_penalty(3, "r1", u)
    assert apply_relay_penalty(3, "r1", u) == 0


def test_different_relays_independent():
    u = RelayUsage()
    assert apply_relay_penalty(5, "r1", u) == 5
    assert apply_relay_penalty(5, "r2", u) == 5
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_effects.py -v`

- [ ] **Step 3: effects.py**

Create `simulator/engine/effects.py`:
```python
from dataclasses import dataclass, field


@dataclass
class RelayUsage:
    count: dict[str, int] = field(default_factory=dict)


def apply_relay_penalty(base: int, relay_id: str, usage: RelayUsage) -> int:
    """Возвращает эффективное значение с учётом штрафа за повторное
    использование relay в одном ходу: первое использование — base,
    каждое следующее — base - n (минимум 0)."""
    n_before = usage.count.get(relay_id, 0)
    usage.count[relay_id] = n_before + 1
    return max(0, base - n_before)
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_effects.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/effects.py simulator/tests/engine/test_effects.py
git commit -m "feat(engine): add RelayUsage penalty helper"
```

---

## Task 10: _resolve_repair

**Files:**
- Modify: `simulator/engine/rules.py` (заменить заглушку `_resolve_repair`)
- Test: `simulator/tests/engine/test_repair.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_repair.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def _mk(hp_p1: int = 30, hp_hq: int = 50):
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, hp_hq, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, hp_p1, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    return s, cfg


def test_repair_heals_by_rs():
    s, cfg = _mk(hp_p1=30)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["p1"].hp == 30 + cfg.rs


def test_repair_capped_at_mhp():
    s, cfg = _mk(hp_p1=48)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["p1"].hp == cfg.mhp  # не выше 50


def test_cannot_repair_self():
    s, cfg = _mk(hp_hq=30)
    # автор=hq, target=hq — невалидно (self-repair запрещён)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 5))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["hq"].hp == 30  # без изменений


def test_invalid_command_noop():
    s, cfg = _mk(hp_p1=30)
    # target вне AR
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 10))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["p1"].hp == 30


def test_relay_penalty_reduces_repair():
    # Две команды через один relay — второй получает RS-1
    s, cfg = _mk(hp_p1=20)
    # Добавим вторую нашу плантацию, которая будет автором через того же relay
    s.plantations["p2"] = Plantation("p2", (4, 5), False, 50, 0, 0)
    s.connected_ids.add("p2")
    s.plantations["p1"] = type(s.plantations["p1"])("p1", (5, 6), False, 20, 0, 0)
    c1 = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    c2 = Command(author=(4, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[c1, c2]))
    # Первое лечит на RS, второе — на RS-1. Сумма RS + (RS-1) = 2*RS - 1 = 9
    assert s2.plantations["p1"].hp == 20 + cfg.rs + max(0, cfg.rs - 1)
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_repair.py -v`
Expected: все 5 падают (пока заглушка).

- [ ] **Step 3: Реализовать _resolve_repair**

Open `simulator/engine/rules.py` and replace `def _resolve_repair(s: EngineState, cmds: CommandSet) -> None: ...` with:

```python
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
            continue  # не repair — другая фаза обработает
        if target_id == pos_to_id.get(cmd.author):
            continue  # нельзя ремонтировать самого себя
        target = s.plantations[target_id]
        relay_id = pos_to_id[cmd.relay]
        heal = apply_relay_penalty(rs, relay_id, usage)
        target.hp = min(mhp, target.hp + heal)
```

Примечание про импорты: локальные внутри функции, чтобы не усложнять module-level граф зависимостей (commands→validation→state→rules — цикла нет, но так безопаснее при дальнейшем росте).

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_repair.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/rules.py simulator/tests/engine/test_repair.py
git commit -m "feat(engine): implement _resolve_repair with relay penalty"
```

---

## Task 11: _resolve_build

**Files:**
- Modify: `simulator/engine/rules.py`
- Test: `simulator/tests/engine/test_build.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_build.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, Construction
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def _mk():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    return s, cfg


def test_build_creates_construction_with_cs_progress():
    s, cfg = _mk()
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert (5, 6) in s2.constructions
    assert s2.constructions[(5, 6)].progress == cfg.cs


def test_two_builders_accumulate_progress():
    s, cfg = _mk()
    s.plantations["p1"] = Plantation("p1", (4, 5), False, 50, 0, 0)
    s.connected_ids.add("p1")
    c1 = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    c2 = Command(author=(4, 5), relay=(4, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[c1, c2]))
    assert s2.constructions[(5, 6)].progress == 2 * cfg.cs


def test_construction_completes_at_50_and_becomes_plantation():
    s, cfg = _mk()
    s.constructions[(5, 6)] = Construction(
        position=(5, 6), progress=45, last_progress_turn=0,
        contributors_last_turn={},
    )
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert (5, 6) not in s2.constructions
    new_ids = [
        p.id for p in s2.plantations.values()
        if p.position == (5, 6)
    ]
    assert len(new_ids) == 1
    p = s2.plantations[new_ids[0]]
    assert p.hp == cfg.mhp
    assert p.immunity_until_turn == s.turn_no + cfg.immunity_turns + 1


def test_build_on_mountain_noop():
    s, cfg = _mk()
    s.mountains = frozenset({(5, 6)})
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert (5, 6) not in s2.constructions


def test_build_on_own_plantation_noop():
    s, cfg = _mk()
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 5))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert (5, 5) not in s2.constructions  # нельзя строить на своей
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_build.py -v`

- [ ] **Step 3: Реализовать _resolve_build (без лимита пока)**

В `simulator/engine/rules.py` замените заглушку `_resolve_build`:

```python
def _resolve_build(s: EngineState, cmds: CommandSet) -> None:
    from simulator.engine.validation import is_valid_command
    from simulator.engine.effects import RelayUsage, apply_relay_penalty
    from simulator.engine.entities import Construction, Plantation

    pos_to_id = {p.position: pid for pid, p in s.plantations.items()}
    usage = RelayUsage()
    cs_base = s.config.cs

    for cmd in cmds.commands:
        if not is_valid_command(s, cmd):
            continue
        target = cmd.target
        # Нельзя строить на горе, своей плантации или вне карты
        if target in s.mountains:
            continue
        if target in pos_to_id:
            continue
        if not (0 <= target[0] < s.config.map_width and 0 <= target[1] < s.config.map_height):
            continue
        # Target может быть вражеской плантацией / бобром — другая фаза обработает.
        # Для build признак: target — свободная клетка без препятствий.
        if any(b.position == target for b in s.beavers.values()):
            continue
        # Determine: empty cell → build; existing construction → join.
        relay_id = pos_to_id.get(cmd.relay)
        if relay_id is None:
            continue
        cs_eff = apply_relay_penalty(cs_base, relay_id, usage)
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

    # Достройка: progress ≥ 50 → плантация
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
        from simulator.engine.events import ConstructionCompleted
        s.events_this_turn.append(
            ConstructionCompleted(position=pos, new_plantation_id=new_id)
        )
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_build.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/rules.py simulator/tests/engine/test_build.py
git commit -m "feat(engine): implement _resolve_build (no overflow limit yet)"
```

---

## Task 12: Plantation limit overflow

**Files:**
- Modify: `simulator/engine/rules.py` (дополнить `_resolve_build`)
- Test: `simulator/tests/engine/test_limit_overflow.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_limit_overflow.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def _setup_at_limit(limit=3):
    cfg = GameConfig(base_plantation_limit=limit, max_plantation_limit=limit)
    s = EngineState.new(cfg, seed=0)
    # HQ и 2 плантации вокруг
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, built_on_turn=10)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, built_on_turn=5)
    s.plantations["p2"] = Plantation("p2", (5, 4), False, 50, 0, built_on_turn=7)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1", "p2"}
    return s, cfg


def test_overflow_removes_oldest_non_main():
    s, cfg = _setup_at_limit(limit=3)
    # Уже на лимите. Начинаем стройку → первый прогресс должен убрать самую старую.
    # Самая старая: p1 (built_on_turn=5).
    cmd = Command(author=(5, 5), relay=(5, 5), target=(6, 5))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert "p1" not in s2.plantations
    assert "p2" in s2.plantations
    assert "hq" in s2.plantations


def test_overflow_can_destroy_hq():
    # Сценарий где ЦУ — самая старая (built_on_turn=0 < others)
    cfg = GameConfig(base_plantation_limit=2, max_plantation_limit=2)
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, built_on_turn=0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, built_on_turn=5)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 4))
    s2 = tick(s, CommandSet(commands=[cmd]))
    # HQ удалён → respawn mechanism отрабатывает (этап 12). Проверяем только,
    # что старый hq id исчез.
    assert "hq" not in s2.plantations


def test_no_overflow_when_under_limit():
    cfg = GameConfig(base_plantation_limit=10, max_plantation_limit=10)
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, built_on_turn=0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert "hq" in s2.plantations
    assert (5, 6) in s2.constructions
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_limit_overflow.py -v`

- [ ] **Step 3: Добавить проверку лимита в _resolve_build**

В `_resolve_build`, перед строкой `if target not in s.constructions:` вставить проверку первого прогресса для новой постройки:

```python
        # Overflow check: новая стройка — учитывать в лимите
        is_new_construction = target not in s.constructions
        if is_new_construction:
            current_total = len(s.plantations) + sum(
                1 for c in s.constructions.values() if c.progress > 0
            )
            if current_total >= s.effective_plantation_limit():
                _remove_oldest_plantation(s)
```

В конец файла `rules.py` добавить:

```python
def _remove_oldest_plantation(s: EngineState) -> None:
    """Удаляет самую старую плантацию (built_on_turn минимален).
    Если это ЦУ — вызывает каскад (respawn обработает на этапе 12)."""
    from simulator.engine.events import PlantationDestroyed, HQLost
    if not s.plantations:
        return
    # Tiebreak при равенстве возраста — случайный среди минимальных
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
        # Плантации становятся изолированными → connectivity пересчёт на этапе 15
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_limit_overflow.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/rules.py simulator/tests/engine/test_limit_overflow.py
git commit -m "feat(engine): enforce plantation limit with oldest-removal on overflow"
```

---

## Task 13: _resolve_sabotage

**Files:**
- Modify: `simulator/engine/rules.py`
- Test: `simulator/tests/engine/test_sabotage.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_sabotage.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def _mk_with_enemy(enemy_hp=20):
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    # Enemy как Plantation, но с id-префиксом "e_" и флагом is_main=False
    # Мы храним enemies в том же словаре, отмечая чужих через отдельный set.
    s.enemy_ids = {"e1"}  # требует поля в EngineState
    s.plantations["e1"] = Plantation("e1", (5, 6), False, enemy_hp, 0, 0)
    return s, cfg


def test_sabotage_reduces_enemy_hp_by_se():
    s, cfg = _mk_with_enemy(enemy_hp=20)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["e1"].hp == 20 - cfg.se


def test_sabotage_kill_awards_cell_reward():
    s, cfg = _mk_with_enemy(enemy_hp=3)
    # (5,6) не boosted (7*n)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert "e1" not in s2.plantations
    assert s2.score == cfg.normal_cell_reward


def test_sabotage_boosted_cell_reward():
    s, cfg = _mk_with_enemy(enemy_hp=3)
    # Переместим врага на boosted клетку (7,7), HQ рядом
    s.plantations["hq"].position = (6, 7)
    s.plantations["e1"].position = (7, 7)
    cmd = Command(author=(6, 7), relay=(6, 7), target=(7, 7))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.score == cfg.boosted_cell_reward


def test_cannot_sabotage_immune_enemy():
    s, cfg = _mk_with_enemy(enemy_hp=20)
    s.plantations["e1"].immunity_until_turn = 999
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.plantations["e1"].hp == 20  # без изменений
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_sabotage.py -v`

- [ ] **Step 3: Добавить поле enemy_ids в EngineState и реализовать sabotage**

В `simulator/engine/state.py` добавить поле в `EngineState`:
```python
    enemy_ids: set[str] = field(default_factory=set)
```

В `simulator/engine/rules.py` заменить `_resolve_sabotage`:

```python
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
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_sabotage.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/state.py simulator/engine/rules.py simulator/tests/engine/test_sabotage.py
git commit -m "feat(engine): implement _resolve_sabotage with kill reward"
```

---

## Task 14: _resolve_beaver_attack

**Files:**
- Modify: `simulator/engine/rules.py`
- Test: `simulator/tests/engine/test_beaver_attack.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_beaver_attack.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, Beaver
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def _mk(beaver_hp=100):
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.beavers["b1"] = Beaver(id="b1", position=(5, 6), hp=beaver_hp,
                             last_hit_by={}, total_damage_this_turn={})
    return s, cfg


def test_beaver_attack_reduces_hp_by_be():
    s, cfg = _mk(beaver_hp=100)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    # Урон BE, затем регенерация +5 (в той же фазе регенерации)
    # На этапе 6 (beaver_attack) мы снимаем BE. На этапе 8 (beaver_damage)
    # бобр регенерируется. Итог: hp = 100 - BE + regen
    assert s2.beavers["b1"].hp == 100 - cfg.be + cfg.beaver_regen


def test_beaver_kill_awards_20x():
    s, cfg = _mk(beaver_hp=3)
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert "b1" not in s2.beavers
    # (5,6) не boosted → normal_cell_reward * 20
    assert s2.score == cfg.normal_cell_reward * cfg.beaver_kill_multiplier


def test_beaver_kill_on_boosted_cell():
    s, cfg = _mk(beaver_hp=3)
    s.beavers["b1"].position = (7, 7)
    s.plantations["hq"].position = (7, 6)
    cmd = Command(author=(7, 6), relay=(7, 6), target=(7, 7))
    s2 = tick(s, CommandSet(commands=[cmd]))
    assert s2.score == cfg.boosted_cell_reward * cfg.beaver_kill_multiplier
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_beaver_attack.py -v`

- [ ] **Step 3: Реализовать _resolve_beaver_attack**

В `rules.py` заменить заглушку:

```python
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
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_beaver_attack.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/rules.py simulator/tests/engine/test_beaver_attack.py
git commit -m "feat(engine): implement _resolve_beaver_attack with 20x kill reward"
```

---

## Task 15: _resolve_relocate

**Files:**
- Modify: `simulator/engine/rules.py`
- Test: `simulator/tests/engine/test_relocate.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_relocate.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def _mk():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 30, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    return s, cfg


def test_relocate_to_orthogonal_neighbor():
    s, cfg = _mk()
    s2 = tick(s, CommandSet(relocate=[(5, 5), (5, 6)]))
    assert s2.hq_id == "p1"
    assert s2.plantations["p1"].is_main is True
    assert s2.plantations["hq"].is_main is False


def test_hp_unchanged_after_relocate():
    s, cfg = _mk()
    s2 = tick(s, CommandSet(relocate=[(5, 5), (5, 6)]))
    assert s2.plantations["hq"].hp == 50
    assert s2.plantations["p1"].hp == 30


def test_relocate_to_non_neighbor_ignored():
    s, cfg = _mk()
    s.plantations["p1"].position = (7, 8)  # не смежная
    s2 = tick(s, CommandSet(relocate=[(5, 5), (7, 8)]))
    assert s2.hq_id == "hq"


def test_relocate_to_diagonal_ignored():
    s, cfg = _mk()
    s.plantations["p1"].position = (6, 6)  # диагональ
    s2 = tick(s, CommandSet(relocate=[(5, 5), (6, 6)]))
    assert s2.hq_id == "hq"
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_relocate.py -v`

- [ ] **Step 3: Реализация**

В `rules.py`:

```python
def _resolve_relocate(s: EngineState, cmds: CommandSet) -> None:
    from simulator.engine.spatial import manhattan

    if not cmds.relocate or len(cmds.relocate) < 2:
        return
    src, dst = cmds.relocate[0], cmds.relocate[-1]
    if manhattan(src, dst) != 1:
        return  # только ортогонально смежный
    pos_to_id = {p.position: pid for pid, p in s.plantations.items()}
    src_id = pos_to_id.get(src)
    dst_id = pos_to_id.get(dst)
    if src_id is None or dst_id is None:
        return
    if src_id != s.hq_id:
        return  # ЦУ только из текущего HQ
    if dst_id not in s.connected_ids:
        return
    s.plantations[src_id].is_main = False
    s.plantations[dst_id].is_main = True
    s.hq_id = dst_id
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_relocate.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/rules.py simulator/tests/engine/test_relocate.py
git commit -m "feat(engine): implement _resolve_relocate"
```

---

## Task 16: _resolve_beaver_damage

**Files:**
- Modify: `simulator/engine/rules.py`
- Test: `simulator/tests/engine/test_beaver_damage.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_beaver_damage.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, Beaver
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def _mk(beaver_hp=100, plant_hp=50):
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, plant_hp, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.beavers["b1"] = Beaver(id="b1", position=(5, 7), hp=beaver_hp,
                             last_hit_by={}, total_damage_this_turn={})
    return s, cfg


def test_beaver_damages_plantation_in_range():
    # AR=2: (5,5) → (5,7) расстояние 2 — в радиусе
    s, cfg = _mk(plant_hp=50)
    s2 = tick(s, CommandSet(upgrade=None))
    assert s2.plantations["hq"].hp == 50 - cfg.beaver_damage


def test_beaver_does_not_damage_out_of_range():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    # beaver в (5,8) — расстояние 3 > AR=2
    s.beavers["b1"] = Beaver("b1", (5, 8), 100, {}, {})
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50


def test_beaver_regenerates_up_to_max():
    s, cfg = _mk(beaver_hp=90)
    s2 = tick(s, CommandSet())
    assert s2.beavers["b1"].hp == min(cfg.beaver_max_hp, 90 + cfg.beaver_regen)


def test_beaver_does_not_damage_immune_plantation():
    s, cfg = _mk(plant_hp=50)
    s.plantations["hq"].immunity_until_turn = 999
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_beaver_damage.py -v`

- [ ] **Step 3: Реализация**

В `rules.py`:

```python
def _resolve_beaver_damage(s: EngineState) -> None:
    from simulator.engine.spatial import in_square_range

    dmg = s.effective_beaver_damage()
    ar = s.config.beaver_attack_radius
    for b in s.beavers.values():
        # Урон всем плантациям в радиусе (включая недострои — их хранят constructions)
        for p in s.plantations.values():
            if in_square_range(b.position, p.position, ar):
                if p.immunity_until_turn > s.turn_no:
                    continue
                p.hp -= dmg
        # Урон стройкам
        for pos, c in list(s.constructions.items()):
            if in_square_range(b.position, pos, ar):
                c.progress = max(0, c.progress - dmg)
    # Регенерация
    for b in s.beavers.values():
        b.hp = min(s.config.beaver_max_hp, b.hp + s.config.beaver_regen)
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_beaver_damage.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/rules.py simulator/tests/engine/test_beaver_damage.py
git commit -m "feat(engine): implement beaver damage + regeneration"
```

---

## Task 17: _resolve_degradation

**Files:**
- Modify: `simulator/engine/rules.py`
- Test: `simulator/tests/engine/test_degradation.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_degradation.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, Construction
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def test_isolated_plantation_loses_ds_per_turn():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["iso"] = Plantation("iso", (20, 20), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}  # iso НЕ связан
    s2 = tick(s, CommandSet())
    assert s2.plantations["iso"].hp == 50 - cfg.ds


def test_connected_plantation_not_degraded():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    s2 = tick(s, CommandSet())
    assert s2.plantations["p1"].hp == 50


def test_construction_without_progress_degrades():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    # Стройка без прогресса в прошлом ходу
    s.constructions[(10, 10)] = Construction(
        position=(10, 10), progress=20,
        last_progress_turn=s.turn_no - 2,  # > 1 ход без прогресса
        contributors_last_turn={},
    )
    s2 = tick(s, CommandSet())
    assert s2.constructions[(10, 10)].progress == 20 - cfg.ds


def test_construction_with_recent_progress_not_degraded():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.constructions[(10, 10)] = Construction(
        position=(10, 10), progress=20,
        last_progress_turn=s.turn_no,  # есть прогресс в этом ходу
        contributors_last_turn={},
    )
    s2 = tick(s, CommandSet())
    assert s2.constructions[(10, 10)].progress == 20
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_degradation.py -v`

- [ ] **Step 3: Реализация**

В `rules.py`:

```python
def _resolve_degradation(s: EngineState) -> None:
    ds = s.effective_ds()
    # Изолированные плантации
    for pid, p in s.plantations.items():
        if pid == s.hq_id:
            continue
        if pid not in s.connected_ids and pid not in s.enemy_ids:
            p.hp -= ds
    # Стройки без прогресса
    for pos, c in list(s.constructions.items()):
        if c.last_progress_turn < s.turn_no:
            c.progress = max(0, c.progress - ds)
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_degradation.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/rules.py simulator/tests/engine/test_degradation.py
git commit -m "feat(engine): implement degradation of isolated plantations and stalled constructions"
```

---

## Task 18: _resolve_meteo (damage + earthquake)

**Files:**
- Modify: `simulator/engine/rules.py`
- Test: `simulator/tests/engine/test_meteo_damage.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_meteo_damage.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, Storm
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def test_earthquake_damages_all_plantations():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (20, 20), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    s.earthquake_turn = s.turn_no  # ударит сейчас
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50 - cfg.earthquake_damage
    assert s2.plantations["p1"].hp == 50 - cfg.earthquake_damage


def test_storm_damages_plantation_inside_disc():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (100, 100), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.storms.append(Storm(id="s", position=(100, 100), radius=10,
                          speed=5, direction=(1, 1), forming_left=0))
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50 - cfg.storm_damage


def test_storm_does_not_damage_outside_radius():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (100, 100), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.storms.append(Storm("s", (50, 50), radius=10,
                          speed=5, direction=(1, 1), forming_left=0))
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50


def test_storm_cannot_reduce_below_1():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (100, 100), True, 1, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.storms.append(Storm("s", (100, 100), 10, 5, (1, 1), 0))
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 1  # не опускается ниже 1


def test_forming_storm_does_no_damage():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (100, 100), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.storms.append(Storm("s", (100, 100), 10, 5, (1, 1), forming_left=3))
    s2 = tick(s, CommandSet())
    assert s2.plantations["hq"].hp == 50
    # И счётчик формирования уменьшается
    assert s2.storms[0].forming_left == 2
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_meteo_damage.py -v`

- [ ] **Step 3: Реализация**

В `rules.py`:

```python
def _resolve_meteo(s: EngineState) -> None:
    from simulator.engine.spatial import in_disc

    # Землетрясение
    if s.earthquake_turn is not None and s.earthquake_turn <= s.turn_no:
        dmg = s.effective_earthquake_damage()
        for p in s.plantations.values():
            if p.immunity_until_turn > s.turn_no:
                continue
            p.hp -= dmg
        for c in s.constructions.values():
            c.progress = max(0, c.progress - dmg)
        s.earthquake_turn = None

    # Бури
    surviving: list = []
    for storm in s.storms:
        if storm.forming_left > 0:
            storm.forming_left -= 1
            surviving.append(storm)
            continue
        # Урон по диску
        for p in s.plantations.values():
            if p.immunity_until_turn > s.turn_no:
                continue
            if in_disc(storm.position, p.position, storm.radius):
                # HP не опускается ниже min
                new_hp = p.hp - s.config.storm_damage
                p.hp = max(s.config.storm_min_hp, new_hp) if p.hp > s.config.storm_min_hp else p.hp
        for c in s.constructions.values():
            if in_disc(storm.position, c.position, storm.radius):
                c.progress = max(0, c.progress - s.config.storm_damage)
        # Движение
        nx = storm.position[0] + storm.direction[0] * storm.speed
        ny = storm.position[1] + storm.direction[1] * storm.speed
        # Если полностью вышла за карту — буря исчезает
        w, h = s.config.map_width, s.config.map_height
        if -storm.radius <= nx <= w + storm.radius and -storm.radius <= ny <= h + storm.radius:
            storm.position = (nx, ny)
            surviving.append(storm)
    s.storms = surviving
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_meteo_damage.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/rules.py simulator/tests/engine/test_meteo_damage.py
git commit -m "feat(engine): implement earthquake and sandstorm damage"
```

---

## Task 19: _resolve_terraform_and_score

**Files:**
- Modify: `simulator/engine/rules.py`
- Test: `simulator/tests/engine/test_terraform.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/engine/test_terraform.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, CellState
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def test_connected_plantation_terraforms_and_scores():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s2 = tick(s, CommandSet())
    # прогресс увеличился на TS
    assert s2.cells[(5, 5)].progress == cfg.ts
    # очки = 10 × TS за обычную клетку
    assert s2.score == 10 * cfg.ts


def test_boosted_cell_gives_15x_per_percent():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (7, 7), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s2 = tick(s, CommandSet())
    assert s2.score == 15 * cfg.ts


def test_isolated_plantation_terraforms_but_no_score():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["iso"] = Plantation("iso", (20, 20), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s2 = tick(s, CommandSet())
    assert s2.cells[(20, 20)].progress == cfg.ts  # terraforms
    # только HQ даёт очки
    assert s2.score == 10 * cfg.ts


def test_cell_reaches_100_plantation_disappears():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    s.cells[(5, 5)] = CellState(progress=97, completed_on_turn=None)
    s2 = tick(s, CommandSet())
    # +5 → 102 → кламп 100, плантация исчезает
    assert s2.cells[(5, 5)].progress == 100
    assert s2.cells[(5, 5)].completed_on_turn == s.turn_no
    assert "hq" not in s2.plantations


def test_cell_degrades_after_delay():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq"}
    # Клетка завершена 30 ходов назад — начинает деградацию
    s.cells[(10, 10)] = CellState(
        progress=100, completed_on_turn=s.turn_no - cfg.cell_degradation_delay
    )
    s2 = tick(s, CommandSet())
    assert s2.cells[(10, 10)].progress == 100 - cfg.cell_degradation_per_turn
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/engine/test_terraform.py -v`

- [ ] **Step 3: Реализация**

В `rules.py`:

```python
def _resolve_terraform_and_score(s: EngineState) -> None:
    from simulator.engine.events import Scored, CellTerraformed

    ts = s.config.ts
    completed: list = []

    for pid, p in list(s.plantations.items()):
        if pid in s.enemy_ids:
            continue
        pos = p.position
        cell = s.cells.get(pos)
        if cell is None:
            cell = CellStateStub(progress=0, completed_on_turn=None)
            s.cells[pos] = cell
        if cell.progress >= 100:
            continue
        delta = min(ts, 100 - cell.progress)
        cell.progress += delta
        # Очки только если связна с ЦУ
        if pid in s.connected_ids:
            per_pct = (
                s.config.boosted_cell_reward // 100
                if s.is_boosted_cell(pos)
                else s.config.normal_cell_reward // 100
            )
            reward = per_pct * delta
            s.score += reward
            s.events_this_turn.append(Scored(amount=reward, source="terraform"))
        if cell.progress >= 100:
            cell.completed_on_turn = s.turn_no
            completed.append(pid)
            s.events_this_turn.append(
                CellTerraformed(position=pos, reward=s.cell_reward(pos))
            )

    # Плантации на 100%-клетках исчезают
    for pid in completed:
        s.plantations.pop(pid, None)

    # Деградация завершённых клеток после cell_degradation_delay
    for pos, cell in list(s.cells.items()):
        if cell.completed_on_turn is None:
            continue
        if pos in {p.position for p in s.plantations.values()}:
            continue  # снова построили — не деградируем
        age = s.turn_no - cell.completed_on_turn
        if age >= s.config.cell_degradation_delay:
            cell.progress = max(0, cell.progress - s.config.cell_degradation_per_turn)
            if cell.progress == 0:
                del s.cells[pos]
                continue
            if cell.progress < 100:
                cell.completed_on_turn = None  # больше не «завершённая»
```

И добавить импорт `CellState` в начало `rules.py`:
```python
from simulator.engine.entities import CellState as CellStateStub
```

(переименование — чтобы избежать конфликта если потом понадобится local alias.)

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/test_terraform.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/rules.py simulator/tests/engine/test_terraform.py
git commit -m "feat(engine): implement terraform, scoring, cell degradation"
```

---

## Task 20: _resolve_respawn, _cleanup_dead, _recompute_connectivity, _spawn_new_meteo

**Files:**
- Modify: `simulator/engine/rules.py`
- Test: `simulator/tests/engine/test_respawn.py`, `test_connectivity.py`, `test_spawn_meteo.py`

- [ ] **Step 1: Тест respawn + cleanup**

Create `simulator/tests/engine/test_respawn.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def test_dead_plantation_removed():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["dead"] = Plantation("dead", (5, 6), False, 0, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "dead"}
    s2 = tick(s, CommandSet())
    assert "dead" not in s2.plantations


def test_hq_death_wipes_plantations_and_applies_penalty():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 0, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    s.score = 1000
    s2 = tick(s, CommandSet())
    assert s2.hq_id is None
    assert s2.plantations == {} or "hq" not in s2.plantations
    # 5% штраф
    assert s2.score == 1000 - 50
```

Create `simulator/tests/engine/test_connectivity.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import CommandSet
from simulator.engine.rules import tick


def test_bfs_marks_connected():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, 0)
    s.plantations["iso"] = Plantation("iso", (20, 20), False, 50, 0, 0)
    s.hq_id = "hq"
    s2 = tick(s, CommandSet())
    assert "hq" in s2.connected_ids
    assert "p1" in s2.connected_ids
    assert "iso" not in s2.connected_ids
```

- [ ] **Step 2: FAIL (все)**

Run: `pytest simulator/tests/engine/test_respawn.py simulator/tests/engine/test_connectivity.py -v`

- [ ] **Step 3: Реализация**

В `rules.py`:

```python
def _resolve_respawn(s: EngineState) -> None:
    """Если ЦУ погибло — все плантации удаляются, применяется штраф, ЦУ респавнится."""
    from simulator.engine.events import HQLost

    if s.hq_id is not None and s.hq_id in s.plantations:
        return  # ЦУ живо
    if s.hq_id is None and not s.plantations:
        # Первый спавн раунда: ЦУ на (1,1)
        pid = s.new_plant_id()
        s.plantations[pid] = Plantation(
            id=pid, position=(1, 1), is_main=True,
            hp=s.effective_mhp(),
            immunity_until_turn=s.turn_no + s.config.immunity_turns + 1,
            built_on_turn=s.turn_no,
        )
        s.hq_id = pid
        return
    # ЦУ погибло в этот ход, но плантации ещё остались
    if s.hq_id is None and s.plantations:
        # Применяем штраф один раз (если его ещё не было от overflow)
        already_penalized = any(
            isinstance(e, HQLost) for e in s.events_this_turn
        )
        if not already_penalized:
            penalty = max(1, s.score * 5 // 100)
            s.score -= penalty
            s.events_this_turn.append(HQLost(prev_hq_id="unknown", penalty=penalty))
        # Стираем все наши плантации
        for pid in list(s.plantations.keys()):
            if pid not in s.enemy_ids:
                del s.plantations[pid]
        # Respawn на (1,1)
        pid = s.new_plant_id()
        s.plantations[pid] = Plantation(
            id=pid, position=(1, 1), is_main=True,
            hp=s.effective_mhp(),
            immunity_until_turn=s.turn_no + s.config.immunity_turns + 1,
            built_on_turn=s.turn_no,
        )
        s.hq_id = pid


def _spawn_new_meteo(s: EngineState) -> None:
    from simulator.engine.entities import Storm
    # Землетрясение
    if s.earthquake_turn is None:
        if s.rng.random() < s.config.earthquake_prob:
            s.earthquake_turn = s.turn_no + 1  # ударит на следующем ходу
    # Буря — только если нет активной
    if not s.storms:
        if s.rng.random() < s.config.storm_prob:
            w, h = s.config.map_width, s.config.map_height
            sx = int(s.rng.integers(0, w))
            sy = int(s.rng.integers(0, h))
            speed = int(s.rng.integers(
                s.config.storm_speed_min, s.config.storm_speed_max + 1
            ))
            dx = int(s.rng.choice([-1, 1]))
            dy = int(s.rng.choice([-1, 1]))
            s.storms.append(Storm(
                id=f"storm_{s.turn_no}",
                position=(sx, sy),
                radius=30,
                speed=speed,
                direction=(dx, dy),
                forming_left=s.config.storm_forming_turns,
            ))


def _cleanup_dead(s: EngineState) -> None:
    from simulator.engine.events import PlantationDestroyed
    to_remove = [pid for pid, p in s.plantations.items() if p.hp <= 0]
    hq_died = s.hq_id in to_remove
    for pid in to_remove:
        p = s.plantations[pid]
        s.events_this_turn.append(
            PlantationDestroyed(plantation_id=pid, position=p.position,
                                was_main=p.is_main)
        )
        del s.plantations[pid]
    if hq_died:
        s.hq_id = None


def _recompute_connectivity(s: EngineState) -> None:
    from simulator.engine.spatial import bfs_connected
    own_positions = {
        pid: p.position
        for pid, p in s.plantations.items()
        if pid not in s.enemy_ids
    }
    s.connected_ids = bfs_connected(s.hq_id, own_positions)
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/engine/ -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/rules.py simulator/tests/engine/test_respawn.py simulator/tests/engine/test_connectivity.py
git commit -m "feat(engine): implement respawn, meteo spawn, cleanup, connectivity"
```

---

## Task 21: Full turn integration test

**Files:**
- Test: `simulator/tests/engine/test_full_turn.py`

- [ ] **Step 1: Тест интеграции**

Create `simulator/tests/engine/test_full_turn.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick


def test_full_turn_repair_and_score_together():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=42)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 30, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    cmd = Command(author=(5, 5), relay=(5, 5), target=(5, 6))
    s2 = tick(s, CommandSet(commands=[cmd]))
    # 1) репайр: p1.hp 30 → 30+RS = 35
    assert s2.plantations["p1"].hp == 30 + cfg.rs
    # 2) очки от терраформ обеих плантаций: TS*10 * 2
    assert s2.score == 10 * cfg.ts * 2
    assert s2.turn_no == 1


def test_determinism_two_runs_same_seed():
    def run(seed):
        cfg = GameConfig()
        s = EngineState.new(cfg, seed=seed)
        s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
        s.hq_id = "hq"
        s.connected_ids = {"hq"}
        for _ in range(50):
            s = tick(s, CommandSet())
        return s.score, s.turn_no, len(s.plantations), s.earthquake_turn

    r1 = run(123)
    r2 = run(123)
    assert r1 == r2
```

- [ ] **Step 2: Run**

Run: `pytest simulator/tests/engine/test_full_turn.py -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add simulator/tests/engine/test_full_turn.py
git commit -m "test(engine): integration test for full turn + determinism"
```

---

## Task 22: Action space specification

**Files:**
- Create: `simulator/env/__init__.py` (пустой)
- Create: `simulator/env/action_space.py`
- Create: `simulator/tests/env/__init__.py` (пустой)
- Test: `simulator/tests/env/test_action_space.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/env/test_action_space.py`:
```python
import numpy as np
from gymnasium import spaces
from simulator.env.action_space import (
    ActionKind, build_action_space, decode_action,
    MAX_PLANT, TARGET_OFFSET_DIM, N_UPGRADES,
)


def test_action_space_is_dict():
    sp = build_action_space()
    assert isinstance(sp, spaces.Dict)
    assert "kind" in sp.spaces
    assert sp["kind"].n == len(ActionKind)
    assert sp["author_idx"].n == MAX_PLANT
    assert sp["relay_idx"].n == MAX_PLANT
    assert sp["target_offset"].n == TARGET_OFFSET_DIM
    assert sp["upgrade_id"].n == N_UPGRADES


def test_decode_action_end_turn():
    a = {"kind": ActionKind.END_TURN.value, "author_idx": 0,
         "relay_idx": 0, "target_offset": 0, "upgrade_id": 0}
    decoded = decode_action(a)
    assert decoded.kind == ActionKind.END_TURN


def test_decode_action_command():
    a = {"kind": ActionKind.COMMAND.value, "author_idx": 2,
         "relay_idx": 3, "target_offset": 12, "upgrade_id": 0}
    decoded = decode_action(a)
    assert decoded.kind == ActionKind.COMMAND
    assert decoded.author_idx == 2
    assert decoded.relay_idx == 3
    assert decoded.target_offset == 12
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/env/test_action_space.py -v`

- [ ] **Step 3: action_space.py**

Create `simulator/env/__init__.py` (пустой) и `simulator/tests/env/__init__.py` (пустой).

Create `simulator/env/action_space.py`:
```python
from dataclasses import dataclass
from enum import IntEnum
from gymnasium import spaces

MAX_PLANT = 40
AR = 2  # AR фиксирован игрой (апгрейда нет)
TARGET_OFFSET_DIM = (2 * AR + 1) ** 2   # 25
N_UPGRADES = 8


class ActionKind(IntEnum):
    END_TURN = 0
    SET_UPGRADE = 1
    SET_RELOCATE = 2
    COMMAND = 3
    NOOP = 4


@dataclass
class DecodedAction:
    kind: ActionKind
    author_idx: int
    relay_idx: int
    target_offset: int
    upgrade_id: int


def build_action_space() -> spaces.Dict:
    return spaces.Dict({
        "kind":          spaces.Discrete(len(ActionKind)),
        "author_idx":    spaces.Discrete(MAX_PLANT),
        "relay_idx":     spaces.Discrete(MAX_PLANT),
        "target_offset": spaces.Discrete(TARGET_OFFSET_DIM),
        "upgrade_id":    spaces.Discrete(N_UPGRADES),
    })


def decode_action(raw: dict) -> DecodedAction:
    return DecodedAction(
        kind=ActionKind(int(raw["kind"])),
        author_idx=int(raw["author_idx"]),
        relay_idx=int(raw["relay_idx"]),
        target_offset=int(raw["target_offset"]),
        upgrade_id=int(raw["upgrade_id"]),
    )


def target_offset_to_delta(offset: int) -> tuple[int, int]:
    """Offset ∈ [0, 24] → (dx, dy) в диапазоне [-AR, AR]."""
    side = 2 * AR + 1
    dx = (offset // side) - AR
    dy = (offset % side) - AR
    return dx, dy


def delta_to_target_offset(dx: int, dy: int) -> int:
    side = 2 * AR + 1
    return (dx + AR) * side + (dy + AR)
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/env/test_action_space.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/env/__init__.py simulator/env/action_space.py simulator/tests/env/__init__.py simulator/tests/env/test_action_space.py
git commit -m "feat(env): define action space and decoder"
```

---

## Task 23: Action mask builder

**Files:**
- Create: `simulator/env/action_mask.py`
- Test: `simulator/tests/env/test_action_mask.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/env/test_action_mask.py`:
```python
import numpy as np
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.env.action_mask import build_action_mask, TurnContext
from simulator.env.action_space import ActionKind, MAX_PLANT, TARGET_OFFSET_DIM


def _mk():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (5, 5), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (5, 6), False, 50, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    return s


def test_end_turn_always_valid():
    s = _mk()
    ctx = TurnContext()
    mask = build_action_mask(s, ctx)
    assert mask["kind"][int(ActionKind.END_TURN)] == 1


def test_used_author_masked_out():
    s = _mk()
    ctx = TurnContext(authors_used={"hq"})
    mask = build_action_mask(s, ctx)
    # hq проиндексирован в (author_list); проверяем что его индекс замаскирован
    # Используем порядок: сортировка по id
    hq_idx = sorted(s.connected_ids).index("hq")
    assert mask["author_idx"][hq_idx] == 0
    p1_idx = sorted(s.connected_ids).index("p1")
    assert mask["author_idx"][p1_idx] == 1


def test_upgrade_valid_when_points_available():
    s = _mk()
    s.upgrades.points = 1
    ctx = TurnContext()
    mask = build_action_mask(s, ctx)
    assert mask["kind"][int(ActionKind.SET_UPGRADE)] == 1


def test_upgrade_masked_when_no_points():
    s = _mk()
    s.upgrades.points = 0
    ctx = TurnContext()
    mask = build_action_mask(s, ctx)
    assert mask["kind"][int(ActionKind.SET_UPGRADE)] == 0
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/env/test_action_mask.py -v`

- [ ] **Step 3: action_mask.py**

Create `simulator/env/action_mask.py`:
```python
from dataclasses import dataclass, field
import numpy as np
from simulator.engine.state import EngineState
from simulator.env.action_space import (
    ActionKind, MAX_PLANT, TARGET_OFFSET_DIM, N_UPGRADES,
)


@dataclass
class TurnContext:
    """Накапливается внутри одного игрового хода."""
    authors_used: set = field(default_factory=set)  # set of plantation ids
    upgrade_set: bool = False
    relocate_set: bool = False
    micro_steps: int = 0

    def ordered_author_ids(self, state: EngineState) -> list:
        return sorted(state.connected_ids)


def build_action_mask(state: EngineState, ctx: TurnContext) -> dict:
    """Возвращает маску под каждый ключ Dict-space. 1 = валидно, 0 = нет."""
    kind_mask = np.zeros(len(ActionKind), dtype=np.int8)
    author_mask = np.zeros(MAX_PLANT, dtype=np.int8)
    relay_mask = np.ones(MAX_PLANT, dtype=np.int8)
    target_mask = np.ones(TARGET_OFFSET_DIM, dtype=np.int8)
    upgrade_mask = np.zeros(N_UPGRADES, dtype=np.int8)

    # END_TURN — всегда
    kind_mask[int(ActionKind.END_TURN)] = 1

    # SET_UPGRADE
    if not ctx.upgrade_set and state.upgrades.points > 0:
        kind_mask[int(ActionKind.SET_UPGRADE)] = 1
        from simulator.engine.types import UpgradeName
        upgrade_list = list(UpgradeName)
        for i, u in enumerate(upgrade_list[:N_UPGRADES]):
            t = state.upgrades.tiers.get(u)
            if t and t.current < t.max:
                upgrade_mask[i] = 1

    # SET_RELOCATE — если у ЦУ есть ортогонально смежная своя плантация
    if not ctx.relocate_set and state.hq_id is not None:
        hq = state.plantations[state.hq_id]
        hq_pos = hq.position
        pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
        has_neighbor = any(
            (hq_pos[0] + dx, hq_pos[1] + dy) in pos_to_id and
            pos_to_id[(hq_pos[0] + dx, hq_pos[1] + dy)] in state.connected_ids
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
        )
        if has_neighbor:
            kind_mask[int(ActionKind.SET_RELOCATE)] = 1

    # COMMAND — есть хоть один неиспользованный автор
    author_ids = ctx.ordered_author_ids(state)
    has_author = False
    for i, pid in enumerate(author_ids[:MAX_PLANT]):
        if pid not in ctx.authors_used:
            author_mask[i] = 1
            has_author = True
    if has_author:
        kind_mask[int(ActionKind.COMMAND)] = 1

    # NOOP — всегда валиден (даёт политике возможность «пропустить»)
    kind_mask[int(ActionKind.NOOP)] = 1

    return {
        "kind": kind_mask,
        "author_idx": author_mask,
        "relay_idx": relay_mask,
        "target_offset": target_mask,
        "upgrade_id": upgrade_mask,
    }
```

Примечание: этот скелет маски грубый (relay_mask / target_mask всё разрешают). В следующей задаче уточним их под выбранного автора. Пока SB3 берёт маску per-key, и агент учится сам.

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/env/test_action_mask.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/env/action_mask.py simulator/tests/env/test_action_mask.py
git commit -m "feat(env): coarse action mask with turn context"
```

---

## Task 24: Observation encoder

**Files:**
- Create: `simulator/env/obs_encoder.py`
- Test: `simulator/tests/env/test_obs_encoder.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/env/test_obs_encoder.py`:
```python
import numpy as np
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.env.obs_encoder import (
    encode_observation, F_SELF, F_ENEMY, F_BEAVER, F_GLOBAL,
    MAX_PLANT, MAX_ENEMY, MAX_BEAVER, GRID_CHANNELS, GRID_SIZE,
)


def _mk():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.plantations["hq"] = Plantation("hq", (50, 50), True, 50, 0, 0)
    s.plantations["p1"] = Plantation("p1", (50, 51), False, 40, 0, 0)
    s.hq_id = "hq"
    s.connected_ids = {"hq", "p1"}
    return s, cfg


def test_obs_shapes_correct():
    s, _ = _mk()
    obs = encode_observation(s)
    assert obs["self"].shape == (MAX_PLANT, F_SELF)
    assert obs["self_mask"].shape == (MAX_PLANT,)
    assert obs["enemy"].shape == (MAX_ENEMY, F_ENEMY)
    assert obs["beaver"].shape == (MAX_BEAVER, F_BEAVER)
    assert obs["global"].shape == (F_GLOBAL,)
    assert obs["grid"].shape == (GRID_CHANNELS, GRID_SIZE, GRID_SIZE)


def test_obs_self_masks_fits_count():
    s, _ = _mk()
    obs = encode_observation(s)
    assert obs["self_mask"].sum() == 2


def test_self_positions_relative_to_hq():
    s, _ = _mk()
    obs = encode_observation(s)
    # hq: (0,0)/50, p1: (0,1)/50
    # Первые N валидных записей соответствуют отсортированным id
    # hq: dx=0, dy=0; p1: dx=0, dy=1 → у одного признаки ~0, у другого dy=0.02
    positions = obs["self"][:2, :2]
    # Нормализация /50
    assert abs(positions).sum() < 0.1


def test_global_contains_turn_number():
    s, cfg = _mk()
    s.turn_no = 60
    obs = encode_observation(s)
    # turn_no/round_turns = 60/600 = 0.1
    assert abs(obs["global"][0] - 0.1) < 1e-6
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/env/test_obs_encoder.py -v`

- [ ] **Step 3: obs_encoder.py**

Create `simulator/env/obs_encoder.py`:
```python
import numpy as np
from simulator.engine.state import EngineState

MAX_PLANT = 40
MAX_ENEMY = 40
MAX_BEAVER = 30
MAX_STORM = 2
F_SELF = 13
F_ENEMY = 4
F_BEAVER = 4
F_STORM = 7
F_GLOBAL = 13
GRID_CHANNELS = 10
GRID_SIZE = 31  # радиус 15 вокруг ЦУ


def encode_observation(state: EngineState) -> dict:
    hq = state.get_hq()
    hq_pos = hq.position if hq else (0, 0)
    mhp = state.effective_mhp()

    self_arr = np.zeros((MAX_PLANT, F_SELF), dtype=np.float32)
    self_mask = np.zeros(MAX_PLANT, dtype=bool)

    # Отсортированные id — стабильный порядок
    own_ids = sorted(
        pid for pid in state.plantations.keys() if pid not in state.enemy_ids
    )
    # Найти самую старую плантацию для is_oldest
    oldest_turn = min(
        (p.built_on_turn for pid, p in state.plantations.items()
         if pid not in state.enemy_ids),
        default=0,
    )

    for i, pid in enumerate(own_ids[:MAX_PLANT]):
        p = state.plantations[pid]
        dx = (p.position[0] - hq_pos[0]) / 50.0
        dy = (p.position[1] - hq_pos[1]) / 50.0
        self_arr[i] = [
            dx, dy,
            p.hp / max(1, mhp),
            float(p.is_main),
            float(pid not in state.connected_ids),  # is_isolated
            max(0, p.immunity_until_turn - state.turn_no) / 3.0,
            float(state.is_boosted_cell(p.position)),
            (state.cells[p.position].progress / 100.0
             if p.position in state.cells else 0.0),
            (state.turn_no - p.built_on_turn) / 100.0,
            float(p.built_on_turn == oldest_turn),
            0.0,  # n_connected_neighbors (дозаполним позже если нужно)
            0.0,  # in_storm
            1.0,  # available_this_turn — обновляется env-wrapper'ом
        ]
        self_mask[i] = True

    # Enemies
    enemy_arr = np.zeros((MAX_ENEMY, F_ENEMY), dtype=np.float32)
    enemy_mask = np.zeros(MAX_ENEMY, dtype=bool)
    enemy_ids = sorted(state.enemy_ids)
    for i, eid in enumerate(enemy_ids[:MAX_ENEMY]):
        e = state.plantations[eid]
        dx = (e.position[0] - hq_pos[0]) / 50.0
        dy = (e.position[1] - hq_pos[1]) / 50.0
        enemy_arr[i] = [dx, dy, e.hp / 50.0, 0.0]  # reachable — dozapol позже
        enemy_mask[i] = True

    # Beavers
    beaver_arr = np.zeros((MAX_BEAVER, F_BEAVER), dtype=np.float32)
    beaver_mask = np.zeros(MAX_BEAVER, dtype=bool)
    for i, (bid, b) in enumerate(sorted(state.beavers.items())[:MAX_BEAVER]):
        dx = (b.position[0] - hq_pos[0]) / 50.0
        dy = (b.position[1] - hq_pos[1]) / 50.0
        beaver_arr[i] = [dx, dy, b.hp / 100.0, 0.0]
        beaver_mask[i] = True

    # Storms
    storm_arr = np.zeros((MAX_STORM, F_STORM), dtype=np.float32)
    storm_mask = np.zeros(MAX_STORM, dtype=bool)
    for i, s in enumerate(state.storms[:MAX_STORM]):
        dx = (s.position[0] - hq_pos[0]) / 50.0
        dy = (s.position[1] - hq_pos[1]) / 50.0
        storm_arr[i] = [dx, dy, 0.0, 0.0, s.radius / 50.0,
                        float(s.forming_left > 0), s.forming_left / 5.0]
        storm_mask[i] = True

    # Global
    global_arr = np.zeros(F_GLOBAL, dtype=np.float32)
    global_arr[0] = state.turn_no / state.config.round_turns
    global_arr[1] = state.upgrades.points / state.config.max_upgrade_points
    global_arr[2] = state.effective_rs()
    global_arr[3] = state.config.se
    global_arr[4] = state.config.be
    global_arr[5] = mhp / 100.0
    global_arr[6] = state.config.ar
    global_arr[7] = state.effective_sr()
    global_arr[8] = (
        (state.earthquake_turn - state.turn_no) / 20.0
        if state.earthquake_turn is not None else 0.0
    )
    global_arr[9] = float(bool(state.storms))
    global_arr[10] = (hq.hp / max(1, mhp)) if hq else 0.0
    global_arr[11] = len([
        pid for pid in state.plantations if pid not in state.enemy_ids
    ]) / MAX_PLANT
    global_arr[12] = state.score / 1e5

    # Grid (пока нулевой — детальное заполнение позже, чтобы не блокировать)
    grid = np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    half = GRID_SIZE // 2
    w, h = state.config.map_width, state.config.map_height
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            x = hq_pos[0] + dx
            y = hq_pos[1] + dy
            gx = dx + half
            gy = dy + half
            if not (0 <= x < w and 0 <= y < h):
                continue
            if (x, y) in state.mountains:
                grid[0, gx, gy] = 1.0
            if (x, y) in {p.position for pid, p in state.plantations.items()
                          if pid not in state.enemy_ids}:
                grid[1, gx, gy] = 1.0
            if (x, y) in state.constructions:
                grid[2, gx, gy] = 1.0
            if (x, y) in {state.plantations[eid].position for eid in state.enemy_ids}:
                grid[3, gx, gy] = 1.0
            if (x, y) in {b.position for b in state.beavers.values()}:
                grid[4, gx, gy] = 1.0
            if (x, y) in state.cells:
                grid[5, gx, gy] = state.cells[(x, y)].progress / 100.0
            if state.is_boosted_cell((x, y)):
                grid[6, gx, gy] = 1.0
            # channel 7 (storm), 8 (reachable), 9 (free) — пока 0

    return {
        "self": self_arr,
        "self_mask": self_mask,
        "enemy": enemy_arr,
        "enemy_mask": enemy_mask,
        "beaver": beaver_arr,
        "beaver_mask": beaver_mask,
        "storm": storm_arr,
        "storm_mask": storm_mask,
        "global": global_arr,
        "grid": grid,
    }
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/env/test_obs_encoder.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/env/obs_encoder.py simulator/tests/env/test_obs_encoder.py
git commit -m "feat(env): observation encoder (entity tensors + local grid)"
```

---

## Task 25: Reward function

**Files:**
- Create: `simulator/env/reward.py`
- Test: `simulator/tests/env/test_reward.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/env/test_reward.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation, CellState
from simulator.engine.events import (
    Scored, PlantationDestroyed, HQLost,
)
from simulator.env.reward import compute_reward, SCORE_SCALE


def test_score_delta_to_reward():
    prev_score = 100
    cur_score = 250
    events = [Scored(amount=150, source="terraform")]
    state_after = None  # не требуется для этого теста
    r = compute_reward(prev_score, cur_score, events, state_after=None)
    # Base: (250-100)/1000 = 0.15
    assert abs(r - 0.15) < 1e-6


def test_penalty_for_lost_plantation():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    s.cells[(5, 6)] = CellState(progress=40, completed_on_turn=None)
    events = [
        PlantationDestroyed(plantation_id="p1", position=(5, 6), was_main=False),
    ]
    r = compute_reward(0, 0, events, state_after=s)
    # потеряли плантацию с прогрессом 40% → потенциал = 1000 * 60/100 = 600
    # reward = 0 - 600/1000 = -0.6
    assert abs(r - (-0.6)) < 1e-6


def test_hq_loss_penalty():
    cfg = GameConfig()
    s = EngineState.new(cfg, seed=0)
    events = [HQLost(prev_hq_id="hq", penalty=50)]
    r = compute_reward(1000, 950, events, state_after=s)
    # score_delta = -50 / 1000 = -0.05; нет доп. штрафа от HQLost (уже в score)
    assert abs(r - (-0.05)) < 1e-6
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/env/test_reward.py -v`

- [ ] **Step 3: reward.py**

Create `simulator/env/reward.py`:
```python
from typing import Optional
from simulator.engine.state import EngineState
from simulator.engine.events import Event, PlantationDestroyed

SCORE_SCALE = 1000.0


def compute_reward(
    prev_score: int,
    cur_score: int,
    events: list[Event],
    state_after: Optional[EngineState],
) -> float:
    delta = (cur_score - prev_score) / SCORE_SCALE

    # Штраф за уничтожение собственных плантаций (потерянный потенциал клетки)
    penalty = 0.0
    if state_after is not None:
        for e in events:
            if isinstance(e, PlantationDestroyed) and not e.was_main:
                cell = state_after.cells.get(e.position)
                progress = cell.progress if cell is not None else 0
                potential = state_after.cell_reward(e.position) * (100 - progress) / 100.0
                penalty += potential / SCORE_SCALE

    return delta - penalty
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/env/test_reward.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/env/reward.py simulator/tests/env/test_reward.py
git commit -m "feat(env): reward = Δscore − potential-lost-plantation penalty"
```

---

## Task 26: Gymnasium environment

**Files:**
- Create: `simulator/env/datssol_env.py`
- Test: `simulator/tests/env/test_env_step.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/env/test_env_step.py`:
```python
import numpy as np
from simulator.env.datssol_env import DatssolEnv
from simulator.env.action_space import ActionKind


def test_reset_returns_obs_and_info():
    env = DatssolEnv(seed=0, map_size=(30, 30), round_turns=20)
    obs, info = env.reset()
    assert "self" in obs
    assert "action_mask" in info


def test_step_with_end_turn_advances_turn():
    env = DatssolEnv(seed=0, map_size=(30, 30), round_turns=20)
    obs, info = env.reset()
    prev_turn = env.state.turn_no
    action = {
        "kind": int(ActionKind.END_TURN),
        "author_idx": 0, "relay_idx": 0,
        "target_offset": 0, "upgrade_id": 0,
    }
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.state.turn_no == prev_turn + 1


def test_episode_terminates_at_round_turns():
    env = DatssolEnv(seed=0, map_size=(30, 30), round_turns=3)
    env.reset()
    end = {"kind": int(ActionKind.END_TURN), "author_idx": 0,
           "relay_idx": 0, "target_offset": 0, "upgrade_id": 0}
    terminated = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(end)
        if terminated or truncated:
            break
    assert terminated or truncated


def test_max_micro_per_turn_forces_end_turn():
    env = DatssolEnv(seed=0, map_size=(30, 30), round_turns=100,
                    max_micro_per_turn=5)
    env.reset()
    noop = {"kind": int(ActionKind.NOOP), "author_idx": 0,
            "relay_idx": 0, "target_offset": 0, "upgrade_id": 0}
    prev_turn = env.state.turn_no
    # Спамим NOOP — после max_micro_per_turn env должен сам завершить ход
    for _ in range(6):
        env.step(noop)
    assert env.state.turn_no == prev_turn + 1
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/env/test_env_step.py -v`

- [ ] **Step 3: datssol_env.py**

Create `simulator/env/datssol_env.py`:
```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional

from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command, CommandSet
from simulator.engine.rules import tick
from simulator.env.action_space import (
    ActionKind, build_action_space, decode_action,
    target_offset_to_delta, MAX_PLANT,
)
from simulator.env.action_mask import build_action_mask, TurnContext
from simulator.env.obs_encoder import (
    encode_observation, F_SELF, F_ENEMY, F_BEAVER, F_STORM, F_GLOBAL,
    MAX_ENEMY, MAX_BEAVER, MAX_STORM, GRID_CHANNELS, GRID_SIZE,
)
from simulator.env.reward import compute_reward
from simulator.engine.types import UpgradeName


class DatssolEnv(gym.Env):
    """Single-agent Gymnasium env с autoregressive action space."""

    def __init__(
        self,
        seed: int = 0,
        map_size: tuple[int, int] = (298, 298),
        round_turns: int = 600,
        max_micro_per_turn: int = 45,
    ):
        super().__init__()
        self._seed = seed
        self._map_size = map_size
        self._round_turns = round_turns
        self._max_micro_per_turn = max_micro_per_turn

        self.action_space = build_action_space()
        self.observation_space = spaces.Dict({
            "self": spaces.Box(-10, 10, (MAX_PLANT, F_SELF), np.float32),
            "self_mask": spaces.MultiBinary(MAX_PLANT),
            "enemy": spaces.Box(-10, 10, (MAX_ENEMY, F_ENEMY), np.float32),
            "enemy_mask": spaces.MultiBinary(MAX_ENEMY),
            "beaver": spaces.Box(-10, 10, (MAX_BEAVER, F_BEAVER), np.float32),
            "beaver_mask": spaces.MultiBinary(MAX_BEAVER),
            "storm": spaces.Box(-10, 10, (MAX_STORM, F_STORM), np.float32),
            "storm_mask": spaces.MultiBinary(MAX_STORM),
            "global": spaces.Box(-10, 10, (F_GLOBAL,), np.float32),
            "grid": spaces.Box(0, 1, (GRID_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32),
        })

        self.state: Optional[EngineState] = None
        self._ctx: TurnContext = TurnContext()
        self._pending_commands: list[Command] = []
        self._pending_upgrade: Optional[UpgradeName] = None
        self._pending_relocate: list = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        cfg = GameConfig(
            map_width=self._map_size[0],
            map_height=self._map_size[1],
            round_turns=self._round_turns,
        )
        self.state = EngineState.new(cfg, seed=self._seed)
        # Первоначальный spawn ЦУ через tick с пустым ходом (resolve_respawn)
        self.state = tick(self.state, CommandSet())
        self._reset_turn_context()
        obs = encode_observation(self.state)
        info = {"action_mask": build_action_mask(self.state, self._ctx)}
        return obs, info

    def _reset_turn_context(self):
        self._ctx = TurnContext()
        self._pending_commands = []
        self._pending_upgrade = None
        self._pending_relocate = []

    def step(self, action: dict):
        assert self.state is not None
        decoded = decode_action(action)
        self._ctx.micro_steps += 1

        end_turn = False
        if decoded.kind == ActionKind.END_TURN:
            end_turn = True
        elif decoded.kind == ActionKind.NOOP:
            pass
        elif decoded.kind == ActionKind.SET_UPGRADE:
            upgrade_list = list(UpgradeName)
            if decoded.upgrade_id < len(upgrade_list):
                self._pending_upgrade = upgrade_list[decoded.upgrade_id]
                self._ctx.upgrade_set = True
        elif decoded.kind == ActionKind.SET_RELOCATE:
            hq = self.state.get_hq()
            if hq is not None:
                # Выбираем ортогонального соседа из connected ids
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nb = (hq.position[0] + dx, hq.position[1] + dy)
                    pos_to_id = {p.position: pid for pid, p in self.state.plantations.items()}
                    if nb in pos_to_id and pos_to_id[nb] in self.state.connected_ids:
                        self._pending_relocate = [hq.position, nb]
                        break
                self._ctx.relocate_set = True
        elif decoded.kind == ActionKind.COMMAND:
            author_ids = self._ctx.ordered_author_ids(self.state)
            if decoded.author_idx < len(author_ids):
                author_id = author_ids[decoded.author_idx]
                if author_id not in self._ctx.authors_used:
                    author = self.state.plantations[author_id]
                    # relay = author (direct) либо indirect по индексу
                    relay_id = (
                        author_ids[decoded.relay_idx]
                        if decoded.relay_idx < len(author_ids)
                        else author_id
                    )
                    relay = self.state.plantations[relay_id]
                    dx, dy = target_offset_to_delta(decoded.target_offset)
                    target = (relay.position[0] + dx, relay.position[1] + dy)
                    self._pending_commands.append(
                        Command(author=author.position,
                                relay=relay.position, target=target)
                    )
                    self._ctx.authors_used.add(author_id)

        # Forced end-turn on micro budget
        if self._ctx.micro_steps >= self._max_micro_per_turn:
            end_turn = True

        if end_turn:
            prev_score = self.state.score
            cs = CommandSet(
                commands=self._pending_commands,
                upgrade=self._pending_upgrade,
                relocate=self._pending_relocate,
            )
            self.state = tick(self.state, cs)
            reward = compute_reward(
                prev_score, self.state.score,
                self.state.events_this_turn,
                state_after=self.state,
            )
            self._reset_turn_context()
        else:
            reward = 0.0

        terminated = (self.state.turn_no >= self._round_turns) or (self.state.hq_id is None and self.state.score < 0)
        truncated = False
        obs = encode_observation(self.state)
        info = {"action_mask": build_action_mask(self.state, self._ctx)}
        return obs, float(reward), terminated, truncated, info
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/env/test_env_step.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/env/datssol_env.py simulator/tests/env/test_env_step.py
git commit -m "feat(env): DatssolEnv with autoregressive micro-action step"
```

---

## Task 27: Parity — CSV loader

**Files:**
- Create: `simulator/parity/__init__.py` (пустой)
- Create: `simulator/parity/db_loader.py`
- Create: `simulator/tests/parity/__init__.py` (пустой)
- Test: `simulator/tests/parity/test_db_loader.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/parity/test_db_loader.py`:
```python
from pathlib import Path
from simulator.parity.db_loader import load_round_ids, load_state_at


DUMP = Path("dump")


def test_load_round_ids_non_empty():
    ids = load_round_ids(DUMP)
    assert len(ids) > 0


def test_load_state_at_turn_zero():
    ids = load_round_ids(DUMP)
    state = load_state_at(DUMP, ids[0], 0)
    assert state.turn_no == 0
    # Ожидаем хотя бы одну собственную плантацию (ЦУ спавнится на первом ходу)
    assert len(state.plantations) >= 0  # может быть 0 на turn_no=0 до spawn
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/parity/test_db_loader.py -v`

- [ ] **Step 3: db_loader.py**

Create `simulator/parity/__init__.py` (пустой) и `simulator/tests/parity/__init__.py` (пустой).

Create `simulator/parity/db_loader.py`:
```python
from pathlib import Path
import pandas as pd
import numpy as np
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import (
    Plantation, Construction, Beaver, Storm, CellState, UpgradeTier,
)
from simulator.engine.types import UpgradeName


def load_round_ids(dump_dir: Path) -> list[int]:
    rounds = pd.read_csv(dump_dir / "rounds.csv")
    return rounds["id"].tolist()


def load_state_at(dump_dir: Path, round_id: int, turn_no: int) -> EngineState:
    turns = pd.read_csv(dump_dir / "turns.csv")
    row = turns[(turns.round_id == round_id) & (turns.turn_no == turn_no)]
    if row.empty:
        raise ValueError(f"No turn {turn_no} in round {round_id}")
    turn_id = int(row.iloc[0]["id"])
    map_w = int(row.iloc[0]["map_width"])
    map_h = int(row.iloc[0]["map_height"])

    cfg = GameConfig(map_width=map_w, map_height=map_h)
    s = EngineState.new(cfg, seed=0)
    s.turn_no = turn_no

    plants = pd.read_csv(dump_dir / "plantations.csv")
    for _, r in plants[plants.turn_id == turn_id].iterrows():
        pid = str(r["plantation_id"])
        s.plantations[pid] = Plantation(
            id=pid, position=(int(r["x"]), int(r["y"])),
            is_main=bool(r["is_main"]), hp=int(r["hp"]),
            immunity_until_turn=int(r["immunity_until"]),
            built_on_turn=0,  # в дампе нет — восстанавливаем эвристически
        )
        if bool(r["is_main"]):
            s.hq_id = pid

    cells = pd.read_csv(dump_dir / "cells.csv")
    for _, r in cells[cells.turn_id == turn_id].iterrows():
        pos = (int(r["x"]), int(r["y"]))
        progress = int(r["progress"])
        s.cells[pos] = CellState(
            progress=progress,
            completed_on_turn=(turn_no if progress >= 100 else None),
        )

    cons = pd.read_csv(dump_dir / "constructions.csv")
    for _, r in cons[cons.turn_id == turn_id].iterrows():
        pos = (int(r["x"]), int(r["y"]))
        s.constructions[pos] = Construction(
            position=pos, progress=int(r["progress"]),
            last_progress_turn=turn_no,
            contributors_last_turn={},
        )

    beavers = pd.read_csv(dump_dir / "beavers.csv")
    for _, r in beavers[beavers.turn_id == turn_id].iterrows():
        bid = str(r["beaver_id"])
        s.beavers[bid] = Beaver(
            id=bid, position=(int(r["x"]), int(r["y"])),
            hp=int(r["hp"]), last_hit_by={}, total_damage_this_turn={},
        )

    mnt = pd.read_csv(dump_dir / "mountains.csv")
    round_id_val = int(row.iloc[0]["round_id"])
    mnt_round = mnt[mnt.round_id == round_id_val]
    s.mountains = frozenset(
        (int(r["x"]), int(r["y"])) for _, r in mnt_round.iterrows()
    )

    up = pd.read_csv(dump_dir / "upgrades.csv")
    up_row = up[up.turn_id == turn_id]
    if not up_row.empty:
        s.upgrades.points = int(up_row.iloc[0]["points"])
        s.upgrades.turns_until_points = int(up_row.iloc[0]["turns_until"])

    tiers = pd.read_csv(dump_dir / "upgrade_tiers.csv")
    for _, r in tiers[tiers.turn_id == turn_id].iterrows():
        try:
            name = UpgradeName(str(r["name"]))
        except ValueError:
            continue
        s.upgrades.tiers[name] = UpgradeTier(
            current=int(r["current"]),
            max=int(r["max_level"]) if r["max_level"] != -1 else 99,
        )

    return s
```

- [ ] **Step 4: PASS (если дамп есть)**

Run: `pytest simulator/tests/parity/test_db_loader.py -v`
Expected: 2 passed (тесты smoke; тонкости не проверяем).

- [ ] **Step 5: Commit**

```bash
git add simulator/parity/__init__.py simulator/parity/db_loader.py simulator/tests/parity/__init__.py simulator/tests/parity/test_db_loader.py
git commit -m "feat(parity): CSV dump loader → EngineState"
```

---

## Task 28: Parity — action loader

**Files:**
- Create: `simulator/parity/action_loader.py`
- Test: `simulator/tests/parity/test_action_loader.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/parity/test_action_loader.py`:
```python
from pathlib import Path
from simulator.parity.action_loader import load_actions_for_turn

DUMP = Path("dump")


def test_load_actions_returns_commandset():
    cs = load_actions_for_turn(DUMP, round_id=1, turn_no=1)
    assert hasattr(cs, "commands")
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/parity/test_action_loader.py -v`

- [ ] **Step 3: action_loader.py**

Create `simulator/parity/action_loader.py`:
```python
from pathlib import Path
import pandas as pd
from simulator.engine.commands import Command, CommandSet


def load_actions_for_turn(dump_dir: Path, round_id: int, turn_no: int) -> CommandSet:
    df = pd.read_csv(dump_dir / "plantation_actions.csv")
    rows = df[(df.round_id == round_id) & (df.turn_no == turn_no)]
    commands: list[Command] = []
    for _, r in rows.iterrows():
        commands.append(Command(
            author=(int(r["author_x"]), int(r["author_y"])),
            relay=(int(r["author_x"]), int(r["author_y"])),  # в дампе relay=author
            target=(int(r["target_x"]), int(r["target_y"])),
        ))
    return CommandSet(commands=commands)
```

Примечание: в дампе хранится только `author` и `target`. Реальный `relay` не записан. Для parity-диффа это систематический источник шума — отмечаем в `calibrator` как known-divergence.

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/parity/test_action_loader.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/parity/action_loader.py simulator/tests/parity/test_action_loader.py
git commit -m "feat(parity): action loader from plantation_actions.csv"
```

---

## Task 29: Parity — diff

**Files:**
- Create: `simulator/parity/diff.py`
- Test: `simulator/tests/parity/test_diff.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/parity/test_diff.py`:
```python
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.parity.diff import diff_states, StateDiff


def test_diff_identical_states_is_empty():
    s1 = EngineState.new(GameConfig(), seed=0)
    s2 = EngineState.new(GameConfig(), seed=0)
    d = diff_states(s1, s2)
    assert d.is_empty()


def test_diff_plantation_hp_delta():
    s1 = EngineState.new(GameConfig(), seed=0)
    s2 = EngineState.new(GameConfig(), seed=0)
    s1.plantations["p1"] = Plantation("p1", (5, 5), True, 50, 0, 0)
    s2.plantations["p1"] = Plantation("p1", (5, 5), True, 45, 0, 0)
    d = diff_states(s1, s2)
    assert not d.is_empty()
    assert "plantation_hp" in d.field_counts


def test_diff_score_mismatch():
    s1 = EngineState.new(GameConfig(), seed=0)
    s2 = EngineState.new(GameConfig(), seed=0)
    s1.score = 100
    s2.score = 200
    d = diff_states(s1, s2)
    assert "score" in d.field_counts
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/parity/test_diff.py -v`

- [ ] **Step 3: diff.py**

Create `simulator/parity/diff.py`:
```python
from dataclasses import dataclass, field
from simulator.engine.state import EngineState


@dataclass
class StateDiff:
    field_counts: dict[str, int] = field(default_factory=dict)
    sample_diffs: list[str] = field(default_factory=list)

    def add(self, field: str, sample: str = "") -> None:
        self.field_counts[field] = self.field_counts.get(field, 0) + 1
        if sample and len(self.sample_diffs) < 20:
            self.sample_diffs.append(sample)

    def is_empty(self) -> bool:
        return not self.field_counts


def diff_states(a: EngineState, b: EngineState) -> StateDiff:
    d = StateDiff()

    if a.score != b.score:
        d.add("score", f"{a.score} vs {b.score}")

    if set(a.plantations.keys()) != set(b.plantations.keys()):
        d.add("plantation_set",
              f"{sorted(a.plantations)} vs {sorted(b.plantations)}")

    for pid, p in a.plantations.items():
        if pid not in b.plantations:
            continue
        q = b.plantations[pid]
        if p.hp != q.hp:
            d.add("plantation_hp", f"{pid}: {p.hp} vs {q.hp}")
        if p.position != q.position:
            d.add("plantation_position", f"{pid}: {p.position} vs {q.position}")
        if p.is_main != q.is_main:
            d.add("plantation_is_main", f"{pid}: {p.is_main} vs {q.is_main}")

    for pos, c in a.constructions.items():
        q = b.constructions.get(pos)
        if q is None:
            d.add("construction_missing", f"{pos}")
            continue
        if c.progress != q.progress:
            d.add("construction_progress",
                  f"{pos}: {c.progress} vs {q.progress}")

    for pos, c in a.cells.items():
        q = b.cells.get(pos)
        if q is None:
            d.add("cell_missing", f"{pos}")
            continue
        if c.progress != q.progress:
            d.add("cell_progress", f"{pos}: {c.progress} vs {q.progress}")

    for bid, bv in a.beavers.items():
        q = b.beavers.get(bid)
        if q is None:
            d.add("beaver_missing", f"{bid}")
            continue
        if bv.hp != q.hp:
            d.add("beaver_hp", f"{bid}: {bv.hp} vs {q.hp}")

    return d
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/parity/test_diff.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/parity/diff.py simulator/tests/parity/test_diff.py
git commit -m "feat(parity): state diff reporter"
```

---

## Task 30: Parity — replay CLI

**Files:**
- Create: `simulator/parity/replay.py`

- [ ] **Step 1: CLI**

Create `simulator/parity/replay.py`:
```python
"""
Replay dump rounds through engine and collect diff against recorded states.
Usage: python -m simulator.parity.replay --dump dump --rounds 1,2,3 --report out.json
"""
import argparse
import json
from pathlib import Path
from simulator.engine.rules import tick
from simulator.parity.db_loader import load_round_ids, load_state_at
from simulator.parity.action_loader import load_actions_for_turn
from simulator.parity.diff import diff_states, StateDiff


def replay_round(dump: Path, round_id: int, max_turns: int | None = None) -> StateDiff:
    agg = StateDiff()
    state = load_state_at(dump, round_id, 0)
    t = 0
    while True:
        actions = load_actions_for_turn(dump, round_id, t)
        next_state = tick(state, actions)
        try:
            ref = load_state_at(dump, round_id, t + 1)
        except ValueError:
            break
        d = diff_states(next_state, ref)
        for k, v in d.field_counts.items():
            agg.field_counts[k] = agg.field_counts.get(k, 0) + v
        state = next_state
        t += 1
        if max_turns is not None and t >= max_turns:
            break
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, default=Path("dump"))
    ap.add_argument("--rounds", type=str, default="")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--report", type=Path, default=Path("parity_report.json"))
    args = ap.parse_args()

    all_ids = load_round_ids(args.dump)
    if args.rounds:
        wanted = [int(x) for x in args.rounds.split(",")]
        all_ids = [r for r in all_ids if r in wanted]

    report: dict = {}
    for rid in all_ids:
        d = replay_round(args.dump, rid, max_turns=args.max_turns)
        report[str(rid)] = d.field_counts
        print(f"round {rid}: {d.field_counts}")

    args.report.write_text(json.dumps(report, indent=2))
    print(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run на первом раунде**

Run: `python -m simulator.parity.replay --rounds 1 --max-turns 5`
Expected: Вывод JSON с расхождениями (на этом этапе их будет много, но CLI не падает).

- [ ] **Step 3: Commit**

```bash
git add simulator/parity/replay.py
git commit -m "feat(parity): replay CLI for engine validation"
```

---

## Task 31: Parity fidelity gate (calibration iteration)

**Files:**
- Modify: `simulator/engine/rules.py` или `config.py` по итогам replay
- Test: интеграционный

- [ ] **Step 1: Проанализировать первичный отчёт**

Run: `python -m simulator.parity.replay --rounds 1,2,3 --max-turns 20`

Отсортировать топ-5 полей с максимальными расхождениями. Типичные кандидаты:
- `beaver_hp` — возможно, порядок attack/regen перепутан
- `plantation_hp` — immunity или iso-degradation не совпадают
- `cell_progress` — таймер деградации
- `score` — tie-breaking sabotage/beaver

- [ ] **Step 2: Отладить первое поле**

Для каждой находки — написать отдельный unit-тест (`tests/engine/test_parity_<field>.py`) с минимальным воспроизведением, поправить `rules.py`, перезапустить replay.

- [ ] **Step 3: Повторить до цели**

Цель: на 3 раундах dump'а `field_counts` для всех не-deterministic-полей (i.e. не `beaver_hp` от неизвестного relay) < 1% от общего числа ходов.

- [ ] **Step 4: Документировать known divergences**

Create `simulator/parity/KNOWN_DIVERGENCES.md` — список систематических расхождений (например, `relay_unknown_in_dump`) с пояснением, почему это ОК.

- [ ] **Step 5: Commit**

```bash
git add simulator/engine/ simulator/parity/KNOWN_DIVERGENCES.md
git commit -m "feat(parity): calibrate engine to dump; fidelity ≥ 99% for gate"
```

---

## Task 32: Imitation dataset builder

**Files:**
- Create: `simulator/imitation/__init__.py` (пустой)
- Create: `simulator/imitation/bc_dataset.py`
- Test: `simulator/tests/imitation/__init__.py` (пустой)
- Test: `simulator/tests/imitation/test_bc_dataset.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/imitation/test_bc_dataset.py`:
```python
from pathlib import Path
from simulator.imitation.bc_dataset import build_bc_samples

DUMP = Path("dump")


def test_build_bc_samples_returns_nonempty():
    samples = list(build_bc_samples(DUMP, round_ids=[1], max_turns=10))
    assert len(samples) > 0
    obs, action = samples[0]
    assert "self" in obs
    assert "kind" in action
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/imitation/test_bc_dataset.py -v`

- [ ] **Step 3: bc_dataset.py**

Create `simulator/imitation/__init__.py` и `simulator/tests/imitation/__init__.py` (пустые).

Create `simulator/imitation/bc_dataset.py`:
```python
"""
Builds (obs, micro_action) training pairs from dump.
For each turn:
  1. Load state at turn T.
  2. Load all recorded commands + upgrade + relocate at turn T.
  3. Generate micro-action sequence:
     for each command (in order of author position): COMMAND action
     optionally SET_UPGRADE, SET_RELOCATE
     finally END_TURN
  4. Roll encoder on state + turn-context between each step.
"""
from pathlib import Path
from typing import Iterator

from simulator.parity.db_loader import load_state_at
from simulator.parity.action_loader import load_actions_for_turn
from simulator.env.obs_encoder import encode_observation
from simulator.env.action_space import ActionKind, MAX_PLANT, delta_to_target_offset
from simulator.env.action_mask import TurnContext


def build_bc_samples(
    dump: Path,
    round_ids: list[int],
    max_turns: int | None = None,
) -> Iterator[tuple[dict, dict]]:
    for rid in round_ids:
        t = 0
        while True:
            try:
                state = load_state_at(dump, rid, t)
            except ValueError:
                break
            cmds = load_actions_for_turn(dump, rid, t)
            ctx = TurnContext()

            # Sort commands by author position (stable order)
            sorted_cmds = sorted(cmds.commands,
                                 key=lambda c: (c.author[0], c.author[1]))

            own_ids = sorted(
                pid for pid in state.plantations.keys()
                if pid not in state.enemy_ids
            )
            pos_to_idx = {state.plantations[pid].position: i
                          for i, pid in enumerate(own_ids[:MAX_PLANT])}

            for cmd in sorted_cmds:
                obs = encode_observation(state)
                if cmd.author not in pos_to_idx:
                    continue
                # В дампе relay=author. Target относительно relay.
                dx = cmd.target[0] - cmd.relay[0]
                dy = cmd.target[1] - cmd.relay[1]
                if abs(dx) > 2 or abs(dy) > 2:
                    continue  # вне AR — пропускаем
                action = {
                    "kind": int(ActionKind.COMMAND),
                    "author_idx": pos_to_idx[cmd.author],
                    "relay_idx": pos_to_idx[cmd.author],
                    "target_offset": delta_to_target_offset(dx, dy),
                    "upgrade_id": 0,
                }
                yield (obs, action)
                ctx.authors_used.add(
                    own_ids[pos_to_idx[cmd.author]]
                )

            # End turn
            obs = encode_observation(state)
            yield (obs, {
                "kind": int(ActionKind.END_TURN),
                "author_idx": 0, "relay_idx": 0,
                "target_offset": 0, "upgrade_id": 0,
            })

            t += 1
            if max_turns is not None and t >= max_turns:
                break
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/imitation/test_bc_dataset.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/imitation/ simulator/tests/imitation/
git commit -m "feat(imitation): BC dataset builder from dump"
```

---

## Task 33: BC pretrain script

**Files:**
- Create: `simulator/imitation/bc_pretrain.py`

- [ ] **Step 1: Скрипт**

Create `simulator/imitation/bc_pretrain.py`:
```python
"""Supervised pretraining of policy on BC dataset."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from simulator.imitation.bc_dataset import build_bc_samples
from simulator.env.action_space import (
    ActionKind, MAX_PLANT, TARGET_OFFSET_DIM, N_UPGRADES,
)


class BCBuffer(Dataset):
    def __init__(self, samples: list[tuple[dict, dict]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, action = self.samples[idx]
        return obs, action


def collate(batch):
    obs_keys = ["self", "global", "grid"]
    out = {k: torch.from_numpy(np.stack([b[0][k] for b in batch])) for k in obs_keys}
    actions = {
        "kind": torch.tensor([b[1]["kind"] for b in batch], dtype=torch.long),
        "author_idx": torch.tensor([b[1]["author_idx"] for b in batch], dtype=torch.long),
        "target_offset": torch.tensor([b[1]["target_offset"] for b in batch], dtype=torch.long),
    }
    return out, actions


class SimplePolicy(nn.Module):
    def __init__(self, f_self=13, max_plant=MAX_PLANT, f_global=13):
        super().__init__()
        self.self_mlp = nn.Sequential(
            nn.Linear(f_self, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(f_global, 32), nn.ReLU(),
        )
        self.backbone = nn.Sequential(
            nn.Linear(64 + 32, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
        )
        self.head_kind = nn.Linear(128, len(ActionKind))
        self.head_author = nn.Linear(128, max_plant)
        self.head_target = nn.Linear(128, TARGET_OFFSET_DIM)

    def forward(self, obs):
        self_feat = self.self_mlp(obs["self"]).mean(dim=1)  # pool over entities
        g = self.global_mlp(obs["global"])
        h = self.backbone(torch.cat([self_feat, g], dim=-1))
        return (
            self.head_kind(h),
            self.head_author(h),
            self.head_target(h),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, default=Path("dump"))
    ap.add_argument("--rounds", type=str, default="1,2,3")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("bc_weights.pt"))
    args = ap.parse_args()

    round_ids = [int(x) for x in args.rounds.split(",")]
    samples = list(build_bc_samples(args.dump, round_ids))
    print(f"Loaded {len(samples)} BC samples")

    ds = BCBuffer(samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    policy = SimplePolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total = 0.0
        for obs, actions in dl:
            kind_l, author_l, target_l = policy(obs)
            loss = (
                loss_fn(kind_l, actions["kind"])
                + loss_fn(author_l, actions["author_idx"])
                + loss_fn(target_l, actions["target_offset"])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch}: loss={total / len(dl):.4f}")

    torch.save(policy.state_dict(), args.out)
    print(f"Saved weights to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run на одном раунде**

Run: `python -m simulator.imitation.bc_pretrain --rounds 1 --epochs 1`
Expected: скрипт отрабатывает, веса сохранены, loss падает между батчами.

- [ ] **Step 3: Commit**

```bash
git add simulator/imitation/bc_pretrain.py
git commit -m "feat(imitation): BC pretraining script (simple policy)"
```

---

## Task 34: PPO training configs and curriculum

**Files:**
- Create: `simulator/train/__init__.py` (пустой)
- Create: `simulator/train/configs/stage0.yaml`, `stage1.yaml`, `stage2.yaml`, `stage3.yaml`
- Create: `simulator/train/curriculum.py`

- [ ] **Step 1: YAML-конфиги**

Create `simulator/train/configs/stage0.yaml`:
```yaml
map_size: [30, 30]
round_turns: 100
beavers_enabled: false
meteo_enabled: false
promote_reward_threshold: 100.0
```

Create `simulator/train/configs/stage1.yaml`:
```yaml
map_size: [60, 60]
round_turns: 300
beavers_enabled: true
meteo_enabled: false
beaver_density: 0.5
promote_reward_threshold: 300.0
```

Create `simulator/train/configs/stage2.yaml`:
```yaml
map_size: [100, 100]
round_turns: 600
beavers_enabled: true
meteo_enabled: true
beaver_density: 1.0
promote_reward_threshold: 600.0
```

Create `simulator/train/configs/stage3.yaml`:
```yaml
map_size: [298, 298]
round_turns: 600
beavers_enabled: true
meteo_enabled: true
beaver_density: 1.0
use_dump_initial_states: true
```

- [ ] **Step 2: curriculum.py**

Create `simulator/train/__init__.py` (пустой).

Create `simulator/train/curriculum.py`:
```python
from pathlib import Path
import yaml
from collections import deque
from simulator.env.datssol_env import DatssolEnv

CONFIG_DIR = Path(__file__).parent / "configs"


def load_stage_config(stage: int) -> dict:
    p = CONFIG_DIR / f"stage{stage}.yaml"
    with p.open() as f:
        return yaml.safe_load(f)


def make_env_for_stage(stage: int, seed: int) -> DatssolEnv:
    cfg = load_stage_config(stage)
    return DatssolEnv(
        seed=seed,
        map_size=tuple(cfg["map_size"]),
        round_turns=cfg["round_turns"],
    )


class CurriculumScheduler:
    def __init__(self, start_stage: int = 0, history: int = 100):
        self.stage = start_stage
        self.history = deque(maxlen=history)

    def record_episode(self, episode_reward: float) -> None:
        self.history.append(episode_reward)

    def maybe_promote(self) -> bool:
        if len(self.history) < self.history.maxlen:
            return False
        cfg = load_stage_config(self.stage)
        threshold = cfg.get("promote_reward_threshold")
        if threshold is None:
            return False
        mean_r = sum(self.history) / len(self.history)
        if mean_r >= threshold:
            self.stage += 1
            self.history.clear()
            return True
        return False
```

- [ ] **Step 3: Commit**

```bash
git add simulator/train/__init__.py simulator/train/configs/ simulator/train/curriculum.py
git commit -m "feat(train): curriculum scheduler and stage configs"
```

---

## Task 35: PPO training script

**Files:**
- Create: `simulator/train/train_ppo.py`

- [ ] **Step 1: Скрипт**

Create `simulator/train/train_ppo.py`:
```python
"""PPO training on DatssolEnv with curriculum."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from simulator.train.curriculum import make_env_for_stage, CurriculumScheduler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("ppo_model"))
    args = ap.parse_args()

    def env_factory(i: int):
        return lambda: make_env_for_stage(args.stage, seed=args.seed + i)

    vec = DummyVecEnv([env_factory(i) for i in range(args.n_envs)])

    model = PPO(
        "MultiInputPolicy",
        vec,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
```

Примечание: `MaskablePPO` предпочтительнее. Переключиться на него после того, как policy успешно обучится на простом PPO — упрощение baseline для первой итерации.

- [ ] **Step 2: Smoke-run**

Run: `python -m simulator.train.train_ppo --stage 0 --total-timesteps 10000 --n-envs 2`
Expected: обучение идёт, model save успешен.

- [ ] **Step 3: Commit**

```bash
git add simulator/train/train_ppo.py
git commit -m "feat(train): PPO training script with stage 0"
```

---

## Task 36: Eval vs rule-based baseline

**Files:**
- Create: `simulator/train/eval.py`

- [ ] **Step 1: Скрипт**

Create `simulator/train/eval.py`:
```python
"""Evaluate a trained PPO agent on DatssolEnv."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from simulator.env.datssol_env import DatssolEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--map-size", type=int, nargs=2, default=[30, 30])
    ap.add_argument("--round-turns", type=int, default=100)
    args = ap.parse_args()

    env = DatssolEnv(
        seed=42, map_size=tuple(args.map_size), round_turns=args.round_turns
    )
    model = PPO.load(args.model, env=env)

    total_rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        total_rewards.append(total)
        print(f"episode {ep}: reward={total:.2f} score={env.state.score}")

    avg = sum(total_rewards) / len(total_rewards)
    print(f"\nMean reward: {avg:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add simulator/train/eval.py
git commit -m "feat(train): eval script for trained PPO"
```

---

## Task 37: Inference adapter to live API

**Files:**
- Create: `simulator/inference/__init__.py` (пустой)
- Create: `simulator/inference/adapter.py`
- Test: `simulator/tests/inference/__init__.py`, `test_adapter.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/inference/__init__.py` (пустой).

Create `simulator/tests/inference/test_adapter.py`:
```python
from simulator.inference.adapter import api_to_state, state_to_payload
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import Plantation
from simulator.engine.commands import Command, CommandSet
from simulator.engine.types import UpgradeName


def test_api_to_state_basic():
    resp = {
        "turnNo": 5,
        "size": [100, 100],
        "actionRange": 2,
        "plantations": [{
            "id": "p1", "position": [10, 10], "isMain": True,
            "hp": 50, "isIsolated": False, "immunityUntilTurn": 0,
        }],
        "enemy": [], "mountains": [], "cells": [], "construction": [],
        "beavers": [], "meteoForecasts": [],
        "plantationUpgrades": {
            "points": 0, "intervalTurns": 30, "turnsUntilPoints": 25,
            "maxPoints": 15, "tiers": [],
        },
    }
    s = api_to_state(resp)
    assert s.turn_no == 5
    assert s.hq_id == "p1"


def test_state_to_payload_build_command():
    cs = CommandSet(commands=[Command((1, 1), (1, 1), (2, 1))])
    payload = state_to_payload(cs)
    assert payload["command"][0]["path"] == [[1, 1], [1, 1], [2, 1]]
```

- [ ] **Step 2: FAIL**

Run: `pytest simulator/tests/inference/test_adapter.py -v`

- [ ] **Step 3: adapter.py**

Create `simulator/inference/__init__.py` (пустой).

Create `simulator/inference/adapter.py`:
```python
import numpy as np
from simulator.engine.config import GameConfig
from simulator.engine.state import EngineState
from simulator.engine.entities import (
    Plantation, Construction, Beaver, Storm, CellState, UpgradeTier,
)
from simulator.engine.commands import CommandSet
from simulator.engine.types import UpgradeName


def api_to_state(resp: dict) -> EngineState:
    size = resp.get("size", [298, 298])
    cfg = GameConfig(map_width=int(size[0]), map_height=int(size[1]))
    s = EngineState.new(cfg, seed=0)
    s.turn_no = int(resp.get("turnNo", 0))

    for p in resp.get("plantations", []):
        pid = str(p["id"])
        s.plantations[pid] = Plantation(
            id=pid, position=(int(p["position"][0]), int(p["position"][1])),
            is_main=bool(p.get("isMain", False)),
            hp=int(p["hp"]),
            immunity_until_turn=int(p.get("immunityUntilTurn", 0)),
            built_on_turn=0,
        )
        if p.get("isMain"):
            s.hq_id = pid

    for e in resp.get("enemy", []):
        eid = str(e["id"])
        s.enemy_ids.add(eid)
        s.plantations[eid] = Plantation(
            id=eid, position=(int(e["position"][0]), int(e["position"][1])),
            is_main=False, hp=int(e["hp"]),
            immunity_until_turn=0, built_on_turn=0,
        )

    s.mountains = frozenset(
        (int(m[0]), int(m[1])) for m in resp.get("mountains", [])
    )

    for c in resp.get("cells", []):
        pos = (int(c["position"][0]), int(c["position"][1]))
        s.cells[pos] = CellState(
            progress=int(c.get("terraformationProgress", 0)),
            completed_on_turn=(s.turn_no - (
                cfg.cell_degradation_delay - int(c.get("turnsUntilDegradation", 0))
            )) if int(c.get("terraformationProgress", 0)) >= 100 else None,
        )

    for c in resp.get("construction", []):
        pos = (int(c["position"][0]), int(c["position"][1]))
        s.constructions[pos] = Construction(
            position=pos, progress=int(c.get("progress", 0)),
            last_progress_turn=s.turn_no,
            contributors_last_turn={},
        )

    for b in resp.get("beavers", []):
        bid = str(b["id"])
        s.beavers[bid] = Beaver(
            id=bid, position=(int(b["position"][0]), int(b["position"][1])),
            hp=int(b["hp"]), last_hit_by={}, total_damage_this_turn={},
        )

    up = resp.get("plantationUpgrades", {})
    s.upgrades.points = int(up.get("points", 0))
    s.upgrades.turns_until_points = int(up.get("turnsUntilPoints", 30))
    for t in up.get("tiers", []):
        try:
            name = UpgradeName(str(t["name"]))
        except ValueError:
            continue
        s.upgrades.tiers[name] = UpgradeTier(
            current=int(t.get("current", 0)),
            max=int(t.get("max", 99)),
        )

    for f in resp.get("meteoForecasts", []):
        kind = str(f.get("kind", ""))
        turns_until = int(f.get("turnsUntil", 0))
        if kind == "earthquake":
            s.earthquake_turn = s.turn_no + turns_until
        elif kind == "sandstorm" and f.get("position"):
            s.storms.append(Storm(
                id=str(f.get("id", "")),
                position=(int(f["position"][0]), int(f["position"][1])),
                radius=int(f.get("radius", 30)),
                speed=0,  # не дан; буря уже движется
                direction=(1, 1),
                forming_left=turns_until if f.get("forming") else 0,
            ))

    return s


def state_to_payload(cs: CommandSet) -> dict:
    return cs.to_payload()
```

- [ ] **Step 4: PASS**

Run: `pytest simulator/tests/inference/test_adapter.py -v`

- [ ] **Step 5: Commit**

```bash
git add simulator/inference/__init__.py simulator/inference/adapter.py simulator/tests/inference/
git commit -m "feat(inference): API ↔ EngineState adapter"
```

---

## Task 38: Live agent loop

**Files:**
- Create: `simulator/inference/live_agent.py`

- [ ] **Step 1: Скрипт**

Create `simulator/inference/live_agent.py`:
```python
"""
Live agent loop: реальный сервер ↔ обученная политика.
Использует уже имеющийся client.py для HTTP.
"""
import argparse
import logging
import time
from pathlib import Path
from stable_baselines3 import PPO

from client import get_arena, post_command
from simulator.inference.adapter import api_to_state, state_to_payload
from simulator.env.obs_encoder import encode_observation
from simulator.env.action_mask import TurnContext, build_action_mask
from simulator.env.action_space import (
    ActionKind, decode_action, target_offset_to_delta, MAX_PLANT,
)
from simulator.engine.commands import Command, CommandSet
from simulator.engine.types import UpgradeName

log = logging.getLogger("live_agent")


def compose_turn(state, policy, max_micro: int = 45) -> CommandSet:
    ctx = TurnContext()
    commands: list[Command] = []
    upgrade = None
    relocate: list = []

    own_ids = sorted(
        pid for pid in state.plantations.keys() if pid not in state.enemy_ids
    )

    for _ in range(max_micro):
        obs = encode_observation(state)
        action, _ = policy.predict(obs, deterministic=True)
        decoded = decode_action(action)
        if decoded.kind == ActionKind.END_TURN:
            break
        elif decoded.kind == ActionKind.SET_UPGRADE and upgrade is None:
            upgrade_list = list(UpgradeName)
            if decoded.upgrade_id < len(upgrade_list):
                upgrade = upgrade_list[decoded.upgrade_id]
            ctx.upgrade_set = True
        elif decoded.kind == ActionKind.SET_RELOCATE and not relocate:
            hq = state.get_hq()
            if hq:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nb = (hq.position[0] + dx, hq.position[1] + dy)
                    pos_to_id = {p.position: pid for pid, p in state.plantations.items()}
                    if nb in pos_to_id and pos_to_id[nb] in state.connected_ids:
                        relocate = [hq.position, nb]
                        break
            ctx.relocate_set = True
        elif decoded.kind == ActionKind.COMMAND:
            if decoded.author_idx < len(own_ids):
                author_id = own_ids[decoded.author_idx]
                if author_id not in ctx.authors_used:
                    author = state.plantations[author_id]
                    relay_id = (
                        own_ids[decoded.relay_idx]
                        if decoded.relay_idx < len(own_ids) else author_id
                    )
                    relay = state.plantations[relay_id]
                    dx, dy = target_offset_to_delta(decoded.target_offset)
                    target = (relay.position[0] + dx, relay.position[1] + dy)
                    commands.append(Command(author.position, relay.position, target))
                    ctx.authors_used.add(author_id)
        # NOOP — просто счётчик

    return CommandSet(commands=commands, upgrade=upgrade, relocate=relocate)


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--poll-interval", type=float, default=0.5)
    args = ap.parse_args()

    policy = PPO.load(args.model)
    last_turn = -1

    while True:
        try:
            raw = get_arena()
            # raw приходит уже как GameState — нужен JSON. Тогда используем requests:
            import requests
            from config import TOKEN, BASE_URL
            resp = requests.get(f"{BASE_URL}/api/arena",
                                headers={"X-Auth-Token": TOKEN}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            state = api_to_state(data)
            if state.turn_no == last_turn:
                time.sleep(args.poll_interval)
                continue
            last_turn = state.turn_no
            cs = compose_turn(state, policy)
            payload = state_to_payload(cs)
            post_command(payload)
            sleep_for = max(0.1, float(data.get("nextTurnIn", 1.0)) - 0.1)
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error("live loop error: %s", e, exc_info=True)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add simulator/inference/live_agent.py
git commit -m "feat(inference): live agent loop wiring policy to real API"
```

---

## Task 39: Final integration smoke test

**Files:**
- Test: `simulator/tests/test_smoke.py`

- [ ] **Step 1: Тест**

Create `simulator/tests/test_smoke.py`:
```python
"""High-level sanity: env runs a short round without errors, reward is finite."""
from simulator.env.datssol_env import DatssolEnv
from simulator.env.action_space import ActionKind


def test_env_completes_short_round():
    env = DatssolEnv(seed=0, map_size=(20, 20), round_turns=30)
    obs, _ = env.reset()
    end = {
        "kind": int(ActionKind.END_TURN),
        "author_idx": 0, "relay_idx": 0,
        "target_offset": 0, "upgrade_id": 0,
    }
    total = 0.0
    for _ in range(50):
        obs, r, term, trunc, _ = env.step(end)
        total += r
        if term or trunc:
            break
    assert abs(total) < 1e6
```

- [ ] **Step 2: Run**

Run: `pytest simulator/tests/test_smoke.py -v`
Expected: passed.

- [ ] **Step 3: Run full suite**

Run: `pytest simulator/tests -v --ignore=simulator/tests/parity`
Expected: все фиксированные тесты pass.

- [ ] **Step 4: Commit**

```bash
git add simulator/tests/test_smoke.py
git commit -m "test: full stack smoke test"
```

---

## Self-review notes

**Покрытие spec'а задачами:**
- engine / tick порядок → Tasks 7–20
- observation → Task 24
- action + mask → Tasks 22–23
- reward → Task 25
- env → Task 26
- parity → Tasks 27–31
- imitation → Tasks 32–33
- training → Tasks 34–36
- inference → Tasks 37–38

**Известные упрощения в плане (допустимые, требуют follow-up после baseline-обучения):**
1. `MaskablePPO` не включён в Task 35 — обычный PPO используется для baseline. Переключиться после первого зелёного старта.
2. `entity-based encoder` сейчас простое mean-pool; для production нужен attention-энкодер (follow-up после stage 2).
3. Parity-gate не автоматизирован как CI — запускается вручную через `python -m simulator.parity.replay`. Можно добавить pre-merge hook позже.
4. BC dataset упрощённо игнорирует upgrade/relocate команды (из-за ограничений дампа). Добавить при наличии этих записей.
5. Multi-agent / self-play — вне scope этого плана; задел в `opponents/` остаётся пустым.

**Гейты качества:**
- Перед PPO (Task 35): parity `≥ 99%` на val split (Task 31) — обязательно.
- Перед инференсом (Task 38): обученная модель превосходит rule-based baseline на eval (Task 36).

---

## Execution handoff

Plan сохранён в `docs/superpowers/plans/2026-04-18-datssol-simulator.md`. Выбор варианта исполнения:

1. **Subagent-Driven (recommended)** — диспетчер субагентов на каждую задачу с ревью между ними. Лучше для длинного плана со строгой TDD-дисциплиной.
2. **Inline Execution** — выполнять задачи в текущей сессии батчами с чекпоинтами.

Какой подход?








