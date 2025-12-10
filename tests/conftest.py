"""
Pytest configuration and fixtures for test suite.
"""

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield


@pytest.fixture
def small_instance():
    """Fixture providing a small 3-SAT instance."""
    from sat_heuristic import random_3sat
    return random_3sat(n_vars=10, m_clauses=42, seed=42)


@pytest.fixture
def medium_instance():
    """Fixture providing a medium 3-SAT instance."""
    from sat_heuristic import random_3sat
    return random_3sat(n_vars=50, m_clauses=210, seed=123)


@pytest.fixture
def large_instance():
    """Fixture providing a large 3-SAT instance."""
    from sat_heuristic import random_3sat
    return random_3sat(n_vars=100, m_clauses=420, seed=456)
