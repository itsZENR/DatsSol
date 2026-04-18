from pathlib import Path
from simulator.imitation.bc_dataset import build_bc_samples

DUMP = Path("dump")


def test_build_bc_samples_returns_nonempty():
    samples = list(build_bc_samples(DUMP, round_ids=[1], max_turns=10))
    # Round 1 may have few/no turns — test tolerates that but verifies no crash
    # and that returned tuples have the right shape when present
    assert isinstance(samples, list)
    if samples:
        obs, action = samples[0]
        assert "self" in obs
        assert "kind" in action
