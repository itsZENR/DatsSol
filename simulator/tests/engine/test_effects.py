from simulator.engine.effects import RelayUsage, apply_relay_penalty


def test_first_use_is_base():
    u = RelayUsage()
    assert apply_relay_penalty(5, "r1", u) == 5
    assert u.count == {"r1": 1}


def test_subsequent_uses_penalty():
    u = RelayUsage()
    apply_relay_penalty(5, "r1", u)
    assert apply_relay_penalty(5, "r1", u) == 4
    assert apply_relay_penalty(5, "r1", u) == 3


def test_penalty_clipped_at_zero():
    u = RelayUsage()
    for _ in range(10):
        apply_relay_penalty(3, "r1", u)
    assert apply_relay_penalty(3, "r1", u) == 0


def test_different_relays_independent():
    u = RelayUsage()
    assert apply_relay_penalty(5, "r1", u) == 5
    assert apply_relay_penalty(5, "r2", u) == 5
