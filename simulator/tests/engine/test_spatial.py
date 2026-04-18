import math
from simulator.engine.spatial import (
    chebyshev, manhattan, euclidean,
    in_square_range, in_disc,
    iter_square_range, bfs_connected,
)


def test_chebyshev():
    assert chebyshev((0, 0), (3, 2)) == 3
    assert chebyshev((5, 5), (2, 1)) == 4
    assert chebyshev((0, 0), (0, 0)) == 0


def test_manhattan():
    assert manhattan((0, 0), (3, 2)) == 5
    assert manhattan((5, 5), (2, 1)) == 7


def test_euclidean():
    assert euclidean((0, 0), (3, 4)) == 5.0
    assert math.isclose(euclidean((0, 0), (1, 1)), math.sqrt(2))


def test_in_square_range():
    assert in_square_range((0, 0), (2, 2), 2) is True
    assert in_square_range((0, 0), (3, 0), 2) is False
    assert in_square_range((0, 0), (2, -2), 2) is True


def test_in_disc_euclidean():
    assert in_disc((0, 0), (3, 4), 5) is True
    assert in_disc((0, 0), (3, 4), 4) is False
    assert in_disc((0, 0), (0, 0), 0) is True


def test_iter_square_range_bounds():
    cells = list(iter_square_range((5, 5), 1, (10, 10)))
    assert set(cells) == {(4, 4), (4, 5), (4, 6),
                          (5, 4), (5, 5), (5, 6),
                          (6, 4), (6, 5), (6, 6)}


def test_iter_square_range_clamped_to_map():
    cells = list(iter_square_range((0, 0), 1, (3, 3)))
    for (x, y) in cells:
        assert 0 <= x < 3 and 0 <= y < 3
    # При center=(0,0), radius=1 ожидаем 4 клетки: (0,0),(0,1),(1,0),(1,1)
    assert set(cells) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_bfs_connected_simple_line():
    plants = {
        "p0": (0, 0),  # ЦУ
        "p1": (1, 0),
        "p2": (2, 0),
        "p3": (5, 0),  # изолирована
    }
    connected = bfs_connected("p0", plants)
    assert connected == {"p0", "p1", "p2"}


def test_bfs_connected_empty_hq():
    connected = bfs_connected(None, {})
    assert connected == set()


def test_bfs_only_cardinal_not_diagonal():
    plants = {
        "hq": (0, 0),
        "p1": (1, 1),  # диагональ
    }
    connected = bfs_connected("hq", plants)
    assert connected == {"hq"}
