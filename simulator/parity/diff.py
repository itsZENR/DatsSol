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

    # Compare plantations by POSITION, not by ID.
    # Rationale: dump uses server-generated UUIDs, our engine uses synthetic IDs (p0, p1, ...).
    # The ID namespace differs by design; position-based comparison avoids spurious divergences.
    a_pos = {p.position: p for p in a.plantations.values()}
    b_pos = {p.position: p for p in b.plantations.values()}

    if set(a_pos.keys()) != set(b_pos.keys()):
        d.add("plantation_set_by_pos",
              f"engine {sorted(a_pos.keys())} vs dump {sorted(b_pos.keys())}")

    for pos, p in a_pos.items():
        if pos not in b_pos:
            continue
        q = b_pos[pos]
        if p.hp != q.hp:
            d.add("plantation_hp", f"{pos}: {p.hp} vs {q.hp}")
        if p.is_main != q.is_main:
            d.add("plantation_is_main", f"{pos}: {p.is_main} vs {q.is_main}")

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
