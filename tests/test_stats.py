import pytest

from adaptive_iteration.core._stats import t_cdf, t_ppf


@pytest.mark.parametrize("df,expected", [
    (1, 12.706205), (2, 4.302653), (5, 2.570582), (10, 2.228139), (30, 2.042272),
])
def test_t_ppf_matches_tables(df, expected):
    assert t_ppf(0.975, df) == pytest.approx(expected, abs=1e-6)


def test_t_ppf_symmetry_and_inverse():
    assert t_ppf(0.5, 7) == 0.0
    assert t_ppf(0.1, 7) == pytest.approx(-t_ppf(0.9, 7))
    assert t_cdf(t_ppf(0.99, 4.3), 4.3) == pytest.approx(0.99, abs=1e-10)


def test_t_ppf_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for p in (0.6, 0.9, 0.975, 0.99375, 0.999):
        for df in (1, 1.5, 3, 4.7, 10, 38.2, 200):
            assert t_ppf(p, df) == pytest.approx(stats.t.ppf(p, df), abs=1e-8)
