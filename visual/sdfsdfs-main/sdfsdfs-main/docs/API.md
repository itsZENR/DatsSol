# DatsSol API Reference

**Test:** `https://games-test.datsteam.dev`
**Prod:** `https://games.datsteam.dev`
**Swagger UI:** https://games-test.datsteam.dev/static/datssol/openapi/

## Авторизация

Все запросы требуют заголовок:

```
X-Auth-Token: <ваш_токен>
```

---

## Endpoints

### GET /api/arena

Получить текущее состояние мира.

#### Response 200

```jsonc
{
  "turnNo": 42,                    // номер текущего хода
  "nextTurnIn": 0.85,              // секунд до следующего хода
  "size": [800, 600],              // размер карты [width, height]
  "actionRange": 2,                // радиус действия плантаций

  "plantations": [                 // ваши плантации
    {
      "id": "abc123",
      "position": [396, 303],
      "isMain": true,              // true = ЦУ
      "isIsolated": false,         // true = потеряна связь с ЦУ
      "immunityUntilTurn": 45,     // иммунитет до этого хода
      "hp": 50
    }
  ],

  "enemy": [                       // вражеские плантации в зоне обзора
    {
      "id": "def456",
      "position": [400, 305],
      "hp": 30
    }
  ],

  "mountains": [                   // непроходимые клетки
    [10, 20], [11, 20]
  ],

  "cells": [                       // терраформированные клетки
    {
      "position": [396, 303],
      "terraformationProgress": 65,       // прогресс %
      "turnsUntilDegradation": 80         // ходов до деградации
    }
  ],

  "construction": [                // стройки
    {
      "position": [396, 302],
      "progress": 25                      // прогресс строительства (из 50)
    }
  ],

  "beavers": [                     // логова бобров
    {
      "id": "beaver1",
      "position": [410, 310],
      "hp": 75
    }
  ],

  "plantationUpgrades": {
    "points": 2,                   // доступные очки апгрейда
    "intervalTurns": 30,           // интервал начисления
    "turnsUntilPoints": 12,        // ходов до следующего очка
    "maxPoints": 15,
    "tiers": [
      { "name": "repair_power",           "current": 1, "max": 3  },
      { "name": "max_hp",                 "current": 0, "max": 5  },
      { "name": "settlement_limit",       "current": 0, "max": 10 },
      { "name": "signal_range",           "current": 0, "max": -1 },
      { "name": "vision_range",           "current": 0, "max": 5  },
      { "name": "decay_mitigation",       "current": 0, "max": 3  },
      { "name": "earthquake_mitigation",  "current": 0, "max": 3  },
      { "name": "beaver_damage_mitigation","current": 0, "max": 5 }
    ]
  },

  "meteoForecasts": [              // прогноз катаклизмов (всегда присутствует)
    {
      "kind": "earthquake",
      "turnsUntil": 0              // 0 = удар в этом ходу
    },
    {
      "kind": "sandstorm",
      "id": "storm-7",
      "forming": false,            // false = уже движется и наносит урон
      "position": [200, 150],      // текущий центр
      "nextPosition": [210, 160],  // центр на следующем ходу
      "radius": 12,                // радиус (dx²+dy² ≤ r²)
      "turnsUntil": null           // omitted когда движется
    }
  ]
}
```

---

### POST /api/command

Отправить действия на текущий ход. Нужно отправить хотя бы одно из: `command`, `plantationUpgrade`, `relocateMain`.

#### Request Body

```jsonc
{
  "command": [
    {
      "path": [
        [x1, y1],    // автор команды (своя плантация)
        [x2, y2],    // выходная точка (может = автору)
        [x3, y3]     // цель команды
      ]
    }
  ],
  "plantationUpgrade": "repair_power",   // опционально
  "relocateMain": [[fromX, fromY], [toX, toY]]  // опционально
}
```

#### Определение типа действия по цели

| Что на целевой клетке | Действие |
|----------------------|----------|
| Своя плантация | Ремонт |
| Чужая плантация | Диверсия |
| Логово бобров | Атака |
| Пустая клетка | Строительство |
| Чужая стройка | Своя стройка на той же клетке |

#### Примеры

**Строительство + перенос ЦУ:**
```json
{
  "command": [
    {
      "path": [[396, 303], [396, 303], [396, 302]]
    },
    {
      "path": [[396, 302], [396, 303], [396, 304]]
    }
  ],
  "relocateMain": [[396, 301], [396, 302]]
}
```

**Только апгрейд:**
```json
{
  "plantationUpgrade": "repair_power"
}
```

#### Response 200

```json
{
  "code": 0,
  "errors": []
}
```

#### Возможные ошибки

| Сообщение | Причина |
|-----------|---------|
| `command already submitted this turn` | Повторная отправка в том же ходу |
| `empty command: no plantation actions, no relocateMain, and no plantationUpgrade provided` | Пустой запрос |

---

### GET /api/logs

Получить игровые логи.

#### Response 200

```json
[
  {
    "time": "2026-04-17T12:00:00Z",
    "message": "main plantation spawned at [396, 303]"
  }
]
```

#### Если не зарегистрирован

```json
{
  "code": 3,
  "errors": ["you are not registered in the game, please register first"]
}
```

---

## Schemas

### Координаты

Все координаты — массив `[x, y]` (два целых числа).

### Апгрейды (plantationUpgrade values)

| Код | Эффект | Макс |
|-----|--------|------|
| `repair_power` | RS +1 | 3 |
| `max_hp` | MHP +10 | 5 |
| `settlement_limit` | Лимит +1 | 10 |
| `signal_range` | SR +1 | - |
| `vision_range` | VR +2 | 5 |
| `decay_mitigation` | DS −2 | 3 |
| `earthquake_mitigation` | Урон от землетрясения −2 | 3 |
| `beaver_damage_mitigation` | Урон от бобров −2 | 5 |

### Радиус бури

Буря использует **круговой** радиус: `dx² + dy² ≤ r²` (отличается от квадратных радиусов плантаций).
