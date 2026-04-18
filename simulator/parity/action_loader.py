from pathlib import Path
import pandas as pd
from simulator.engine.commands import Command, CommandSet


def load_actions_for_turn(dump_dir: Path, round_id: int, turn_no: int) -> CommandSet:
    df = pd.read_csv(dump_dir / "plantation_actions.csv")
    rows = df[(df.round_id == round_id) & (df.turn_no == turn_no)]
    commands: list[Command] = []
    for _, r in rows.iterrows():
        # Dump stores only author and target; relay = author as fallback.
        commands.append(Command(
            author=(int(r["author_x"]), int(r["author_y"])),
            relay=(int(r["author_x"]), int(r["author_y"])),
            target=(int(r["target_x"]), int(r["target_y"])),
        ))
    return CommandSet(commands=commands)
