# DatsSol Simulator — Design

- Версия: 1.0
- Дата: 2026-04-18
- Статус: Draft (до approve пользователем)

## Цель

Построить Python-симулятор игры DatsSol, пригодный для обучения RL-агента, способного полностью заменить существующий rule-based бот (`strategies.py`). Симулятор должен быть достаточно верен серверу, чтобы обученная в нём политика переносилась в реальную игру без существенной деградации.

## Источники правды

1. **Игровая документация** (Google Docs, сохранена в `.claude/game_doc.txt`) — авторитет по правилам.
2. **OpenAPI spec** (`.claude/openapi.yml`) — авторитет по форматам запросов/ответов.
3. **Дамп БД** (`dump/`, 85 раундов × ~600 ходов, ~51К переходов) — арбитр при расхождении с документацией, корпус для calibration и imitation learning.
4. **Существующий код** (`bot.py`, `strategies.py`, `game_logic.py`, `models.py`, `client.py`) — эталон форматов DTO и пример рабочих стратегий; **не переписывать**, использовать как baseline.

## Ключевые игровые параметры

- Карта: 278×278 (в рантайме брать `size` из `GET /api/arena`), клетки — пустыня или гора.
- Раунд: 600 ходов, ход ≈ 1 с.
- Стартовые параметры плантации: HP=MHP=50, TS=5, CS=5, RS=5, SE=5, BE=5, DS=10, AR=2, SR=3, VR=3.
- Лимит: 30 плантаций базово, до 40 с апгрейдом `settlement_limit`. Стройки с progress>0 входят в лимит. Превышение → гибель самой старой (возраст с момента завершения стройки). **ЦУ тоже может стать жертвой.**
- Радиусы AR/SR/VR — Chebyshev (квадрат). Буря — **Euclidean** (круг, подтверждено DB-доком дампа).
- Boosted-клетки: `x % 7 == 0 ∧ y % 7 == 0` → 75 очков/ход вместо 50.
- Бобры: 100 HP, регенерация 5/ход, атака AR=2, урон 15/ход минус `beaver_damage_mitigation*2`. Kill = **20× очки клетки**.
- Шторм: 10% шанс возникновения/ход, формирование 5 ходов, урон 2 HP/ход, HP-floor 1. Скорость 5–15 клеток/ход по диагонали через центр.
- Землетрясение: 5% шанс/ход, мгновенный -10 HP всем (-`earthquake_mitigation*2`).
- Очки апгрейда: +1 каждые 30 ходов, макс. 15.

## Область применения

- **In scope:** движок, Gym-среда, observation/action-кодирование, reward, curriculum, parity-валидация на дампе, BC-warmstart, PPO-обучение, интеграционный адаптер на реальный сервер через существующий `client.py`.
- **Out of scope (этого спека):** multi-agent self-play, distributed training, Numba/JAX-акселерация, визуализация. Задел для self-play оставляем (отдельный слой `opponents/`).

## Архитектура

```
simulator/
├── engine/                 # чистый движок, без Gym/RL
│   ├── state.py              # EngineState, Plantation, Beaver, Storm, UpgradesState
│   ├── config.py             # GameConfig (все константы)
│   ├── spatial.py            # Chebyshev/Euclidean, BFS от ЦУ
│   ├── events.py             # typed Event union для reward
│   └── rules.py              # tick() с 15 подэтапами
├── env/                    # Gymnasium-обёртка
│   ├── datssol_env.py        # reset/step, autoregressive цикл
│   ├── obs_encoder.py        # entities + local grid
│   ├── action_space.py       # spec + mask builder
│   └── reward.py             # Δscore + штрафы
├── parity/                 # валидация движка против дампа
│   ├── db_loader.py          # CSV → EngineState
│   ├── action_loader.py      # plantation_actions → CommandSet
│   ├── replay.py             # прогнать все ходы, собрать расхождения
│   ├── diff_report.py        # JSON отчёт по полям
│   └── calibrator.py         # grid-search неявных констант
├── imitation/              # BC pretrain
│   ├── bc_dataset.py         # (obs, action_sequence) из дампа
│   └── bc_pretrain.py        # supervised policy init
├── opponents/              # сейчас: stub solo; задел под rule-based противников
│   └── solo.py
├── train/
│   ├── configs/              # YAML per-stage
│   ├── curriculum.py         # 4 стадии + промоушн по reward
│   ├── pretrain_bc.py
│   ├── train_ppo.py
│   └── eval.py               # против rule-based baseline
├── inference/              # для реальной игры
│   └── live_agent.py         # client.py ↔ obs_encoder ↔ policy ↔ post_command
└── tests/                  # pytest, unit + parity
```

**Принцип изоляции:** `engine/` не знает про RL/Numpy-тензоры/HTTP. `env/` — единственный адаптер к Gymnasium. `inference/` — единственный адаптер к реальному серверу. Это:
- даёт тестируемость движка вне RL;
- позволяет запускать parity-проверку независимо;
- upgrade-путь к self-play меняет только `env/` и добавляет `opponents/`.

## Поток данных

**Один игровой ход** = несколько `env.step` (autoregressive микро-действия + `END_TURN`). `engine.tick()` вызывается только на `END_TURN`.

```
env.reset() → encode_obs(engine_state) → obs
 ┌── autoregressive loop (в пределах одного игрового хода) ────┐
 │  policy(obs, mask) → micro_action                           │
 │  if micro_action == END_TURN:                               │
 │      engine.tick(accumulated_commands) → new_engine_state   │
 │      reward = Δscore - штраф_за_потери                       │
 │      break                                                   │
 │  else:                                                       │
 │      accumulate command, update mask, obs                    │
 └─────────────────────────────────────────────────────────────┘
```

## Engine — порядок обработки хода

Порядок взят из основной документации (раздел «Фаза обработки», п.1–11) и дополнен недостающими системными подэтапами (cleanup, спавн катаклизмов, пересчёт связности). Парити-валидация на дампе может скорректировать порядок.

1. `apply_upgrade_points` — +1 очко каждые `intervalTurns`, до `maxPoints`.
2. `apply_upgrade(actions)` — максимум один апгрейд за ход.
3. `resolve_repair(actions)` — +`RS_effective` HP валидным target'ам (автор, реле связны, целевая в AR реле, автор не использует себя как цель).
4. `resolve_build(actions)` — прогресс +`CS_effective`. **Сразу после первого прогресса** проверка лимита: если переполнение — удалить самую старую плантацию (возможно ЦУ).
5. `resolve_sabotage(actions)` — -`SE_effective` HP; tiebreak: при одновременном уничтожении очки идут тому, кто в последний ход нанёс больше урона; при равенстве — делятся поровну.
6. `resolve_beaver_attack(actions)` — -`BE_effective`; kill → +20× очки клетки (с теми же tiebreak-правилами).
7. `resolve_relocate(actions)` — перенос ЦУ на ортогонально соседнюю свою плантацию. HP плантаций не меняются.
8. `resolve_beaver_damage` — бобры бьют все плантации в Chebyshev≤2 на 15-`mitigation`, регенерируют +5.
9. `resolve_degradation` — изолированные плантации теряют `DS` HP/ход; стройки без прогресса 1 ход — то же.
10. `resolve_meteo` — урон активного шторма (Euclidean disc), урон землетрясения (если turnsUntil=0).
11. `resolve_terraform_and_score` — +TS к клеткам под плантациями. При 100% — плантация исчезает, клетка «живёт» `cell_degradation_delay = 20` ходов, затем -10%/ход. Начисление очков пропорционально Δprogress в этом ходу (не накопленному).
12. `resolve_respawn` — если ЦУ уничтожено — штраф 5% текущего score (не менее 1), все плантации удалены, новый ЦУ на окраине (в обучении используем фиксированную процедуру спавна + детерминистичный RNG).
13. `spawn_new_meteo(rng)` — 10% буря (если нет активной), 5% землетрясение.
14. `cleanup_dead` — удалить плантации с HP≤0.
15. `recompute_connectivity` — BFS от ЦУ, пометить `is_isolated`.

**Детерминизм:** всё случайное идёт через `state.rng: np.random.Generator`. Никакого системного времени, `random.random()` или незатравленного numpy.

## Observation

```python
spaces.Dict({
  "self":     Box(shape=(MAX_PLANT=40, F_self)),
  "self_mask": Box(shape=(MAX_PLANT,), dtype=bool),
  "enemy":    Box(shape=(MAX_ENEMY=40, F_enemy)),
  "enemy_mask": ...,
  "beaver":   Box(shape=(MAX_BEAVER=30, F_beaver)),
  "beaver_mask": ...,
  "storm":    Box(shape=(MAX_STORM=2, F_storm)),
  "storm_mask": ...,
  "global":   Box(shape=(F_global,)),
  "grid":     Box(shape=(C=10, 31, 31)),     # локальное окно ±15 вокруг ЦУ
  "action_mask": MultiBinary(sum_of_discrete_sizes_in_action_dict),
})
```

**Признаки `self`** (нормализованные): `dx/50`, `dy/50`, `hp/MHP`, `is_main`, `is_isolated`, `immunity_turns_left/3`, `on_boosted_cell`, `cell_progress/100`, `turns_since_built/100`, `is_oldest` (помечен в encoder для защиты от оверлимита), `n_connected_neighbors/4`, `in_storm`, `available_this_turn`.

**Признаки `enemy`**: `dx/50`, `dy/50`, `hp/50`, `reachable_from_any_of_ours`.

**Признаки `beaver`**: `dx/50`, `dy/50`, `hp/100`, `n_ours_in_atk_range/10`.

**Признаки `storm`**: `dx_center/50`, `dy_center/50`, `dx_next/50`, `dy_next/50`, `radius/50`, `forming` (bool), `turns_to_impact/20` (0 если уже бьёт).

**Признаки `global`**: `turn_no/600`, `upgrade_points/15`, `RS`, `SE`, `BE`, `MHP/100`, `AR`, `SR`, `earthquake_in/20` (0 если нет), `storm_active`, `hq_hp/MHP`, `n_plant/40`, `score/1e5`.

**Grid-каналы** (31×31 вокруг ЦУ, за пределами — нули): mountains, own_plant, own_construction, enemy_plant, beaver, cell_progress, boosted_flag, storm_damage, reachable_from_network, free_cell.

**Позиции** храним относительно ЦУ (translation invariance). На smena ЦУ (`relocate`) окно переносится.

## Action

Autoregressive: один `env.step` = одно микро-действие.

```python
spaces.Dict({
  "kind":         Discrete(5),   # 0=END_TURN, 1=SET_UPGRADE, 2=SET_RELOCATE, 3=COMMAND, 4=NOOP
  "author_idx":   Discrete(MAX_PLANT=40),
  "relay_idx":    Discrete(MAX_PLANT=40),
  "target_offset": Discrete((2*AR+1)**2 = 25),  # AR зафиксирован = 2, апгрейда AR нет
  "upgrade_id":   Discrete(8),
})
```

`target_offset` — индекс клетки в квадрате AR вокруг relay. Абсолютная позиция вычисляется как `relay_pos + (dx, dy) - (AR, AR)`. Это корректно покрывает все достижимые цели из любого relay, независимо от его положения относительно ЦУ.

Тип команды (build/repair/sabotage/beaver) определяется целью автоматически:
- пустая клетка → build
- своя плантация (не сам автор) → repair
- видимая вражеская плантация → sabotage
- бобёр → beaver attack

**Action mask** пересчитывается перед каждым микро-шагом:
- `author_idx`: наша, связна с ЦУ, не использовалась как автор в этом ходу.
- `relay_idx`: наша связная, в Chebyshev≤SR от выбранного автора.
- `target_offset`: клетка в радиусе AR от relay, НЕ гора, не вне карты, НЕ строящаяся иммунная плантация. Для build — пусто; для repair — своя не-ЦУ; для sabotage — вражеская; для beaver — бобёр.
- `SET_UPGRADE` (`upgrade_id`): купленных<max, `upgrade_points>0`, апгрейд ещё не куплен в этом ходу.
- `SET_RELOCATE`: у ЦУ есть ортогонально соседняя своя плантация.
- `END_TURN` всегда валиден (fallback: пустой `plantationUpgrade=""` — сервер примет).
- `BUILD` дополнительно валиден только если либо `total_owned < limit`, либо `total_owned == limit ∧ oldest != ЦУ`.

**Бюджет автогрессии на ход:** `n_connected + 3` (макс. 43 микро-шага для 40 плантаций). На 600 ходов ≈ 26К env.step. Приемлемо для PPO.

**Ограничение числа микро-шагов в env:** `MAX_MICRO_PER_TURN = 45`. При достижении — принудительный `END_TURN`. Защита от зависания политики до того, как она выучит END_TURN.

## Reward

```
r_step = Δscore / 1000
       - penalty_for_lost_plantations
       - 0.05 * prev_score  (при потере ЦУ)
```

- `Δscore` — как считает engine (terraform + sabotage kills + beaver kills).
- `penalty_for_lost_plantations`: если плантация P уничтожена в этом ходу, вычитаем потенциальную ценность её клетки (`1000` для обычной, `1500` для boosted) × оставшаяся_фракция `(100 - cell_progress)/100`. Это выражает упущенную будущую прибыль.
- Никаких дополнительных shape-бонусов. Нет терминального reward — `score` уже накоплен через delta, сумма даёт финальный result (telescoping).

`VecNormalize` в SB3 для reward и obs. γ=0.99.

## Curriculum

| Stage | Карта | Ходов | Бобры | Катаклизмы | Промоушн |
|-------|-------|-------|-------|------------|----------|
| 0 | 30×30 | 100 | нет | off | mean_ep_reward > 100 за 100 эпизодов |
| 1 | 60×60 | 300 | редкие (0.5× серверной плотности) | off | > 300 |
| 2 | 100×100 | 600 | серверная плотность | on | > 600 |
| 3 | 378×378 | 600 | из дампа | on | финальная цель |

Stage 3 использует реальные стартовые состояния из дампа (случайный `round_id, turn=0`) для разнообразия.

## Imitation warm-start (BC)

- Из `plantation_actions.csv` (289К записей) восстановить (`state`, `actions`) на каждый ход из `turns.csv` + snapshot-таблицы.
- Для каждого хода упорядочить команды по позиции автора (lexicographic) → целевая последовательность микро-действий.
- Cross-entropy по `kind`, `author_idx`, `relay_idx`, `target_local` с уважением маски (обнуляем loss на padding).
- 2–3 эпохи, батч 256, Adam 1e-4.
- Веса → initialization для MaskablePPO.

## Parity-валидация

CLI:
```
python -m simulator.parity.replay --dump dump/ --split val --report out.json
```

- `train` split: 20% раундов для калибровки.
- `val` split: 80% hold-out.

Алгоритм: для каждого раунда и хода `T` — `engine.from_dump(round, T).tick(actions_T)` vs `from_dump(round, T+1)`. Сравнение:

- `plantations`: множество (id, x, y, hp, is_main, is_isolated)
- `beavers`: (id, x, y, hp)
- `cells`: (x, y, progress, turns_until_deg)
- `constructions`: (x, y, progress)
- `upgrades`: (points, tiers[name].current)
- `score` (из `score_log`)

Gate: **`exact_state_match_rate ≥ 99%` на val** — обязательное условие для старта PPO-тренировки. До этого порога итерируем: diff-отчёт → правка `rules.py` → повтор.

Калибратор: если систематическое расхождение в одном поле (например, beaver regen шаг) — параметризуем константу в `GameConfig`, grid-search минимизируя суммарный diff на train split.

## Inference интеграция

`inference/live_agent.py`:
```
while round_active:
    raw = client.get_arena()
    engine_state = EngineState.from_api(raw)  # тот же адаптер, что и dump → state
    obs = encode_obs(engine_state)
    commands = []
    while True:
        action, _ = policy.predict(obs, action_masks=mask)
        if action == END_TURN: break
        commands.append(...)
    post_command(to_api_payload(commands, upgrade, relocate))
```

Политика работает на абсолютно том же obs-кодировании, что и в тренировке.

## Тестирование

- `tests/engine/` — по unit-тесту на каждый подэтап `tick` на синтетических мирах 10×10.
- `tests/obs_action/` — тесты, что кодирование обратимо и маска корректна.
- `tests/parity/` — интеграционный (помечен `@pytest.mark.slow`), прогоняет 3 раунда из dump'а и ожидает match-rate ≥ 99%.

## Зависимости

Ключевые (добавить в `requirements.txt`):
- `gymnasium>=0.29`
- `stable-baselines3>=2.3`
- `sb3-contrib>=2.3` (MaskablePPO)
- `numpy>=1.26`
- `torch>=2.2`
- `pandas>=2.2` (для загрузки CSV дампа)
- `pytest`, `pytest-xdist`

## Метрики успеха

1. **Engine fidelity:** `exact_state_match_rate ≥ 99%` на val split дампа (blocker gate).
2. **Training speed:** ≥2000 env.step/sec/core на stage 1.
3. **BC baseline:** скопировать поведение rule-based бота с cross-entropy validation loss ниже шумового уровня.
4. **RL baseline:** PPO на stage 2 достигает mean-reward равного BC в течение 10М steps.
5. **Deployment:** обученная политика на реальном сервере даёт score ≥ 80% от score существующего rule-based бота на первом запуске (sim-to-real gap измерен).

## Открытые вопросы (для плана реализации)

1. Нужен ли GPU для тренировки? На таком observation (≈2К-мерный) CPU-PPO хватит — оценка ~24ч на M4-уровне CPU для stage 2.
2. Будем ли держать существующий rule-based бот живым как scripted opponent для stage 4 (после вывода PPO)? Сейчас — да, задел `opponents/scripted.py` оставлен.
3. Какой источник тайминга в инференсе? У нас `next_turn_in` от сервера — политика должна финишировать autoregressive цикл за ≤0.8 с (POLL_INTERVAL). Профилируем после обучения.

## Приложение: не-очевидные нюансы из документации

- «Усиленная клетка» = `x%7==0 ∧ y%7==0` — буквально оба модуля 0.
- После 100% терраформации клетка живёт 20 ходов (40 → 30 → 20 по последовательным ребалансам 2026-04-18), потом -10%/ход. Клетка без плантации не приносит очков, но может снова быть достроена — учитывать в reward-потенциале.
- При одновременной достройке несколькими игроками весь прогресс обнуляется (достижение «Будем соседями»). Engine должен реализовать это даже в solo-режиме (для parity).
- Диверсии на недостроенных плантациях невозможны — вместо этого конкурирующий билд на той же клетке.
- Иммунитет после постройки 3 хода от любого негатива (диверсии, катаклизмы, бобры).
- `decay_mitigation` упомянут в доке и swagger, но в списке апгрейдов раздела «Апгрейд плантаций» его нет. Наличие в данных уточняет parity.
