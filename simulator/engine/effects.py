from dataclasses import dataclass, field


@dataclass
class RelayUsage:
    count: dict[str, int] = field(default_factory=dict)


def apply_relay_penalty(base: int, relay_id: str, usage: RelayUsage) -> int:
    """Effective value considering relay reuse penalty: first use = base,
    each subsequent use = base - n (min 0)."""
    n_before = usage.count.get(relay_id, 0)
    usage.count[relay_id] = n_before + 1
    return max(0, base - n_before)
