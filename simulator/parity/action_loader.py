from pathlib import Path
import pandas as pd
from simulator.engine.commands import Command, CommandSet


_actions_cache: dict = {}


def _load_actions_by_round(dump_dir: Path, round_id: int) -> dict:
    key = (str(dump_dir.resolve()), round_id)
    if key not in _actions_cache:
        df = pd.read_csv(dump_dir / "plantation_actions.csv")
        df = df[df.round_id == round_id]
        _actions_cache[key] = {
            tno: grp for tno, grp in df.groupby("turn_no")
        }
    return _actions_cache[key]


def load_actions_for_turn(dump_dir: Path, round_id: int, turn_no: int) -> CommandSet:
    by_turn = _load_actions_by_round(dump_dir, round_id)
    rows = by_turn.get(turn_no)
    commands: list[Command] = []
    if rows is not None:
        for _, r in rows.iterrows():
            commands.append(Command(
                author=(int(r["author_x"]), int(r["author_y"])),
                relay=(int(r["author_x"]), int(r["author_y"])),
                target=(int(r["target_x"]), int(r["target_y"])),
            ))
    return CommandSet(commands=commands)
