"""Unit tests for metrics.py — synthetic D values, no model needed."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from pilot.metrics import faith_s, jaccard, pairwise_jaccards, spearman_rho
import numpy as np


class TestFaithS:
    def test_full_destruction(self):
        # D_S == D_corrupt → ablating S fully destroys behaviour → Faith = 1.0
        assert faith_s(d_full=2.0, d_s=0.0, d_corrupt=0.0) == pytest.approx(1.0)

    def test_no_effect(self):
        # D_S == D_full → ablating S has no effect → Faith = 0.0
        assert faith_s(d_full=2.0, d_s=2.0, d_corrupt=0.0) == pytest.approx(0.0)

    def test_partial(self):
        # Half destruction → Faith = 0.5
        assert faith_s(d_full=2.0, d_s=1.0, d_corrupt=0.0) == pytest.approx(0.5)

    def test_zero_denominator(self):
        import math
        assert math.isnan(faith_s(d_full=1.0, d_s=0.5, d_corrupt=1.0))

    def test_negative_d(self):
        # D_full negative, D_corrupt=0: (−1 − (−0.5)) / (−1 − 0) = −0.5/−1 = 0.5
        assert faith_s(d_full=-1.0, d_s=-0.5, d_corrupt=0.0) == pytest.approx(0.5)

    def test_overcorrection(self):
        # Ablation overshoots: D_S < D_corrupt → Faith > 1
        result = faith_s(d_full=2.0, d_s=-0.5, d_corrupt=0.0)
        assert result == pytest.approx(1.25)


class TestJaccard:
    def test_identical(self):
        s = {1, 2, 3}
        assert jaccard(s, s) == pytest.approx(1.0)

    def test_disjoint(self):
        assert jaccard({1, 2}, {3, 4}) == pytest.approx(0.0)

    def test_partial(self):
        assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)

    def test_empty_both(self):
        assert jaccard(set(), set()) == pytest.approx(1.0)

    def test_one_empty(self):
        assert jaccard({1}, set()) == pytest.approx(0.0)


class TestPairwiseJaccards:
    def test_shape(self):
        sets = [{1, 2}, {2, 3}, {3, 4}]
        mat = pairwise_jaccards(sets)
        assert mat.shape == (3, 3)

    def test_diagonal(self):
        sets = [{1, 2}, {3, 4}]
        mat = pairwise_jaccards(sets)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[1, 1] == pytest.approx(1.0)

    def test_symmetric(self):
        sets = [{1, 2}, {2, 3, 4}]
        mat = pairwise_jaccards(sets)
        assert mat[0, 1] == pytest.approx(mat[1, 0])


class TestSpearman:
    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        rho, _ = spearman_rho(x, x)
        assert rho == pytest.approx(1.0)

    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        rho, _ = spearman_rho(x, y)
        assert rho == pytest.approx(-1.0)

    def test_uncorrelated(self):
        rho, _ = spearman_rho([1, 2, 3, 4], [3, 1, 4, 2])
        assert abs(rho) < 0.6
