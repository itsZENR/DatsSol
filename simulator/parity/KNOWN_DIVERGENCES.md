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

## Next calibration steps (future work)

1. Run extended replay on all 85 available rounds to increase sample size and surface any rare divergences.
2. Add a "canonical" replay mode that excludes stochastic fields from diff scoring.
3. Investigate any actionable divergences found in step 1; add minimal-reproduction unit tests (`tests/engine/test_parity_<field>.py`) and fix the underlying engine behavior.
4. Add a parity gate to CI: on each PR, replay 10 rounds and fail if scoring-field divergence rate exceeds a budget (initial: loose, tightening over time).
5. Consider extending the diff module to track weather and seismic events for completeness.
