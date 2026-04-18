"""
Replay dump rounds through engine and collect diff against recorded states.

Usage:
  python -m simulator.parity.replay --dump dump --rounds 1,2,3 --report out.json
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
