from dataclasses import dataclass, field
from simulator.engine.state import EngineState


@dataclass
class StateDiff:
    field_counts: dict[str, int] = field(default_factory=dict)
    sample_diffs: list[str] = field(default_factory=list)

    def add(self, field_name: str, sample: str = "") -> None:
        self.field_counts[field_name] = self.field_counts.get(field_name, 0) + 1
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
