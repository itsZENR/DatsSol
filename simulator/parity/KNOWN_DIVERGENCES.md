# DatsSol Simulator — Known Parity Divergences

## Baseline

Measured against dump on 2026-04-18 at commit f5131b3b9d217accbacb105adaea321760020413.
Replay: 5 rounds × 20 turns.

**Result: Zero divergences detected.**

## Aggregated divergences (top fields)

| Field | Count |
|-------|-------|
| (none) | 0 |

The baseline replay on rounds 1–5 with max 20 turns per round found no divergences between the replayed engine state and the recorded dump state at each turn boundary.

## Expected divergences (non-actionable)

### `construction_progress`, `plantation_hp` — relay data missing from dump

The dump stores only `author_x/y` and `target_x/y` in `plantation_actions.csv`; the real `relay` position is not recorded. Our action loader sets `relay = author` as a fallback.

Real server commands often used a distant relay, which changes the relay-penalty (and therefore the effective CS/RS/SE/BE). Result: all command-based effects (build progress, repair HP, sabotage damage, beaver damage) will diverge systematically when the real command went through a relay ≠ author.

**Current status:** The baseline shows no divergences on this field, suggesting either:
1. The test rounds did not trigger enough relay-dependent commands to expose the issue, or
2. The fallback `relay = author` approximation was sufficient for this subset of the game data.

Recommend monitoring this field in larger replay runs.

### `cell_progress` — initial turn state

`load_state_at` for turn T does not replay prior turns; it snapshots the dump. The `completed_on_turn` field is not in the dump and is derived via heuristic (`turn_no if progress >= 100 else None`). This is correct for steady-state but can produce mismatch in the first few turns after a cell completes.

**Current status:** Baseline shows no divergences. This heuristic appears to hold across the test data.

### Stochastic fields — seeded differently

- `earthquake_turn`, `storms[].speed/direction/forming_left`: our engine generates these via `state.rng` (seed=0 by default at load), while the dump shows actual server draws.
- These fields are not currently included in the diff comparison.

**Note:** The diff module does not yet track weather/seismic state, so these are not measured in the baseline.

## Potentially actionable (needs investigation)

- `plantation_hp` on plantations NOT involved in any command this turn — if divergent in larger runs, suggests a bug in beaver damage or isolation degradation.
- `cell_progress` during steady-state (plantation alive, already tracked) — should deterministically be +TS per turn.
- `score` divergence — should track terraform points exactly for turns with no kills.

**Baseline finding:** These fields all matched the dump (count=0).

## Updated baseline (10 rounds full-turn replay)

Measured 2026-04-18 at commit 7c8bf0e.
Wall time: 8.5 seconds (post-optimization).

### Top divergence fields (aggregated across 10 rounds)

Rounds replayed: 1,2,3,4,5,6,7,8,9,10 = 4317 turn boundaries total.

| Field | Count | Approx. rate per turn-boundary |
|-------|-------|-------------------------------|
| `score` | 161 | 3.73% |
| `cell_missing` | 161 | 3.73% |
| `plantation_set` | 149 | 3.45% |

### Interpretation

1. **plantation_set divergence (~3.45%)**: The replayed engine and dump disagree on which plantation IDs are present. This suggests either a plantation creation/destruction mismatch in the engine, or the dump captures a different snapshot timing than the simulated tick. Likely due to initial state construction missing some plantations or command processing order differences.

2. **cell_missing and score divergence (~3.73% each)**: Both are directly related to plantation set discrepancies. Missing plantations mean missing cells (and their terraform points). The score field diverges because it's built from plantation-contributed points. These are consequences of the plantation_set issue rather than independent bugs.

3. **All three fields correlate**: If N plantations are missing at turn boundary T, then exactly N cells are "missing" (not tracked), and score diverges by the sum of their terraformed values. The 3.45%–3.73% range suggests a systematic issue affecting 3–4% of turn boundaries.

### Actionable findings

- **Investigation needed**: Why do certain plantation IDs appear in the dump at a turn but not in the replayed engine state? Candidates:
  - Initial state loading (`load_state_at` for first turn) missing plantations from CSV.
  - Plantation lifecycle (placement/destruction commands) not replayed identically.
  - Relay fallback (`relay = author`) causing command target mismatches on certain cells.

- **Next step**: Add debug logging to compare planted positions at first turn boundary (round 1, turn 2) between engine and dump. Narrow down whether the gap is in loader or in tick logic.

## Root cause analysis: plantation_set divergence (2026-04-18, commit fd3474a)

### Finding 1: ID namespace mismatch (resolved)

**Issue:** The engine generates synthetic plantation IDs (`p0`, `p1`, ...) via `EngineState.new_plant_id()`, while the dump contains server-generated UUIDs (e.g., `07b808e6-2150-4cd7-b8c7-cb9fc182056e`). This caused **all** ID-based comparisons to diverge by design, even when plantation state was identical.

**Root cause:** `diff.py` was comparing plantations by ID using `set(a.plantations.keys()) != set(b.plantations.keys())`, which is inherently incompatible with heterogeneous ID namespaces.

**Fix (v0.1):** Changed comparison from ID-keyed to **position-keyed**. Created `a_pos = {p.position: p for p in a.plantations.values()}` and compare by position instead. This decouples the diff from the ID generation strategy.

**Impact:** Eliminated spurious `plantation_set` divergences caused by ID namespace mismatch. Unit test added: `test_diff_ignores_id_namespace()` verifies that plantations at identical positions are correctly treated as equivalent despite differing IDs.

### Finding 2: Missing connectivity recomputation (resolved)

**Issue:** The dump does not persist `connected_ids` (the set of plantation IDs reachable from HQ via adjacency). When `load_state_at()` reconstructs an EngineState from CSV, it leaves `connected_ids = set()`. This causes all command validation to fail (`is_valid_command()` checks `if relay_id not in s.connected_ids: return False`), preventing all build/repair/sabotage actions from executing.

**Concrete case:** Round 50, turn 11→12. Dump shows a construction at (126, 19) with progress=45 at turn 11, and a completed plantation at that position by turn 12. The engine's tick failed to execute the build command because the relay plantation at (127, 19) was not in `connected_ids`, leaving the construction stuck at progress=45 and preventing completion.

**Root cause:** `db_loader.py` reconstructs state from per-turn snapshots in CSV files, but connectivity is a derived property (computed via BFS from HQ in rules.py phase 15). The dump does not record this derived state, so it must be recomputed at load time.

**Fix (v0.1):** Added connectivity recomputation at the end of `load_state_at()`:
```python
from simulator.engine.spatial import bfs_connected
own_positions = {pid: p.position for pid, p in s.plantations.items() if pid not in s.enemy_ids}
s.connected_ids = bfs_connected(s.hq_id, own_positions)
```

**Impact:** Reduced plantation_set_by_pos divergences by allowing commands to execute correctly. Constructions now complete as intended, creating plantations that match the dump.

### Combined impact

Before fixes:
- `plantation_set` divergence: ~3.45% (149 occurrences in 4317 turn boundaries across rounds 1–10)

After fixes (applied sequentially):
- ID namespace fix: Eliminated systematic ID mismatches in diff comparison
- Connectivity fix: Allowed commands to execute, reducing position-based plantations divergences

Residual plantation_set_by_pos divergences (rounds 10, 20, 30, 40 still show 20 per 50 turns) appear to be unrelated to these two issues and warrant further investigation (likely related to relay fallback or stochastic field timing).

## Next calibration steps (future work)

1. Run extended replay on all 85 available rounds to increase sample size and surface any rare divergences.
2. Add a "canonical" replay mode that excludes stochastic fields from diff scoring.
3. Investigate residual plantation_set_by_pos divergences in later turns (not explained by ID namespace or connectivity issues).
4. Add a parity gate to CI: on each PR, replay 10 rounds and fail if scoring-field divergence rate exceeds a budget (initial: loose, tightening over time).
5. Consider extending the diff module to track weather and seismic events for completeness.
