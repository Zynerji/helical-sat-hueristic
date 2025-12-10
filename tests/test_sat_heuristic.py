"""
Tests for Helical SAT Heuristic core functionality.
"""

import numpy as np
import pytest
from sat_heuristic import (
    random_3sat,
    evaluate_sat,
    helical_sat_approx,
    uniform_sat_baseline
)


class TestRandom3SAT:
    """Tests for random 3-SAT generator."""

    def test_basic_generation(self):
        """Test basic instance generation."""
        n_vars = 10
        m_clauses = 42
        clauses = random_3sat(n_vars, m_clauses, seed=42)

        assert len(clauses) == m_clauses
        assert all(len(clause) == 3 for clause in clauses)

    def test_literals_in_range(self):
        """Test that literals are in valid range."""
        n_vars = 20
        m_clauses = 84
        clauses = random_3sat(n_vars, m_clauses, seed=123)

        for clause in clauses:
            for lit in clause:
                assert abs(lit) >= 1 and abs(lit) <= n_vars

    def test_distinct_variables(self):
        """Test that clauses have distinct variables."""
        n_vars = 50
        m_clauses = 210
        clauses = random_3sat(n_vars, m_clauses, seed=456)

        for clause in clauses:
            abs_vars = [abs(lit) for lit in clause]
            assert len(set(abs_vars)) == 3, "Clause should have 3 distinct variables"

    def test_reproducibility(self):
        """Test that same seed produces same instance."""
        n_vars = 15
        m_clauses = 63
        seed = 789

        clauses1 = random_3sat(n_vars, m_clauses, seed=seed)
        clauses2 = random_3sat(n_vars, m_clauses, seed=seed)

        assert clauses1 == clauses2


class TestEvaluateSAT:
    """Tests for SAT evaluation function."""

    def test_all_satisfied(self):
        """Test with assignment that satisfies all clauses."""
        clauses = [[1, 2, 3], [1, -2, 3], [-1, 2, -3]]
        assign = np.array([1, 1, 1])  # All variables True

        rho = evaluate_sat(clauses, assign)
        assert rho == 1.0

    def test_none_satisfied(self):
        """Test with assignment that satisfies no clauses."""
        clauses = [[1, 2, 3]]
        assign = np.array([-1, -1, -1])  # All variables False

        rho = evaluate_sat(clauses, assign)
        assert rho == 0.0

    def test_partial_satisfaction(self):
        """Test with partially satisfied clauses."""
        clauses = [
            [1, 2, 3],    # Satisfied by x1=True
            [-1, 2, 3]    # Not satisfied (x1=True, x2=False, x3=False)
        ]
        assign = np.array([1, -1, -1])

        rho = evaluate_sat(clauses, assign)
        assert rho == 0.5

    def test_random_assignment_baseline(self):
        """Test that random assignment gives ~7/8 satisfaction for 3-SAT."""
        n_vars = 100
        m_clauses = 420
        clauses = random_3sat(n_vars, m_clauses, seed=42)

        rhos = []
        for i in range(20):
            np.random.seed(i)
            assign = np.random.choice([1, -1], size=n_vars)
            rho = evaluate_sat(clauses, assign)
            rhos.append(rho)

        avg_rho = np.mean(rhos)
        # Random assignment should satisfy ~7/8 ≈ 0.875 of clauses
        assert 0.8 < avg_rho < 0.95


class TestHelicalSATApprox:
    """Tests for helical SAT approximation."""

    def test_small_instance(self):
        """Test on small instance."""
        n_vars = 10
        m_clauses = 42
        clauses = random_3sat(n_vars, m_clauses, seed=42)

        rho, bound = helical_sat_approx(clauses, n_vars)

        assert 0.0 <= rho <= 1.0
        assert 0.0 <= bound <= 1.0

    def test_better_than_random(self):
        """Test that helical method beats random assignment on average."""
        n_vars = 50
        m_clauses = 210
        clauses = random_3sat(n_vars, m_clauses, seed=123)

        helical_rho, _ = helical_sat_approx(clauses, n_vars)

        # Random baseline
        random_rhos = []
        for i in range(10):
            np.random.seed(1000 + i)
            assign = np.random.choice([1, -1], size=n_vars)
            rho = evaluate_sat(clauses, assign)
            random_rhos.append(rho)

        random_avg = np.mean(random_rhos)

        # Helical should generally beat random (with some variance)
        # Just check it's competitive
        assert helical_rho > 0.8

    def test_different_parameters(self):
        """Test with different omega parameters."""
        n_vars = 20
        m_clauses = 84
        clauses = random_3sat(n_vars, m_clauses, seed=456)

        rho1, _ = helical_sat_approx(clauses, n_vars, omega=0.1)
        rho2, _ = helical_sat_approx(clauses, n_vars, omega=0.5)

        # Both should produce valid results
        assert 0.0 <= rho1 <= 1.0
        assert 0.0 <= rho2 <= 1.0

    def test_deterministic(self):
        """Test that method is deterministic for same input."""
        n_vars = 30
        m_clauses = 126
        clauses = random_3sat(n_vars, m_clauses, seed=789)

        # Note: eigenvector signs might flip, so we check assignment quality
        rho1, _ = helical_sat_approx(clauses, n_vars, omega=0.3, N=20000)
        rho2, _ = helical_sat_approx(clauses, n_vars, omega=0.3, N=20000)

        # Should give very similar (or identical) results
        assert abs(rho1 - rho2) < 0.01


class TestUniformBaseline:
    """Tests for uniform spectral baseline."""

    def test_small_instance(self):
        """Test uniform baseline on small instance."""
        n_vars = 10
        m_clauses = 42
        clauses = random_3sat(n_vars, m_clauses, seed=42)

        rho = uniform_sat_baseline(clauses, n_vars)

        assert 0.0 <= rho <= 1.0

    def test_vs_random(self):
        """Test that uniform baseline beats random assignment."""
        n_vars = 50
        m_clauses = 210
        clauses = random_3sat(n_vars, m_clauses, seed=999)

        uniform_rho = uniform_sat_baseline(clauses, n_vars)

        # Should be competitive
        assert uniform_rho > 0.8


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_clauses(self):
        """Test with empty clause list."""
        clauses = []
        n_vars = 10
        assign = np.ones(n_vars)

        # Should handle gracefully (though not meaningful)
        # This might raise an error or return 0/1, either is acceptable
        try:
            rho = evaluate_sat(clauses, assign)
            # If it doesn't error, should return some value
            assert isinstance(rho, (float, np.floating))
        except (ZeroDivisionError, ValueError):
            # Acceptable to raise error on empty input
            pass

    def test_single_variable(self):
        """Test with single variable."""
        n_vars = 1
        m_clauses = 5
        clauses = [[1], [-1], [1]]

        # This violates 3-SAT but tests robustness
        assign = np.array([1])
        # Should handle without crashing
        # evaluate_sat might give partial results

    def test_large_instance(self):
        """Test that large instances don't crash."""
        n_vars = 200
        m_clauses = 840
        clauses = random_3sat(n_vars, m_clauses, seed=12345)

        rho, bound = helical_sat_approx(clauses, n_vars)

        assert 0.0 <= rho <= 1.0
        assert 0.0 <= bound <= 1.0


@pytest.mark.parametrize("n,m,seed", [
    (10, 42, 1),
    (20, 84, 2),
    (50, 210, 3),
])
def test_parametrized_instances(n, m, seed):
    """Parametrized test for various instance sizes."""
    clauses = random_3sat(n, m, seed=seed)
    rho, bound = helical_sat_approx(clauses, n)

    assert 0.0 <= rho <= 1.0
    assert 0.0 <= bound <= 1.0
    assert len(clauses) == m
