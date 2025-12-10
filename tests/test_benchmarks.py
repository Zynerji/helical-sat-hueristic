"""
Tests for benchmarking module.
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from benchmarks import (
    load_dimacs_cnf,
    random_assignment_baseline,
    run_single_benchmark,
    aggregate_results
)
from sat_heuristic import random_3sat


class TestDIMACSParser:
    """Tests for DIMACS CNF parser."""

    def test_parse_simple_cnf(self):
        """Test parsing a simple DIMACS CNF file."""
        cnf_content = """c Simple test CNF
c Comment line
p cnf 3 2
1 2 3 0
-1 -2 -3 0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.cnf') as f:
            f.write(cnf_content)
            temp_file = f.name

        try:
            clauses, n_vars = load_dimacs_cnf(temp_file)

            assert n_vars == 3
            assert len(clauses) == 2
            assert [1, 2, 3] in clauses
            assert [-1, -2, -3] in clauses
        finally:
            os.unlink(temp_file)

    def test_parse_with_comments(self):
        """Test parsing CNF with many comment lines."""
        cnf_content = """c This is a comment
c Another comment
c Yet another
p cnf 2 1
c Comment in the middle
1 -2 0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.cnf') as f:
            f.write(cnf_content)
            temp_file = f.name

        try:
            clauses, n_vars = load_dimacs_cnf(temp_file)

            assert n_vars == 2
            assert len(clauses) == 1
            assert clauses[0] == [1, -2]
        finally:
            os.unlink(temp_file)

    def test_parse_uf20_format(self):
        """Test parsing typical SATLIB uf20 format."""
        cnf_content = """c This is a random 3-SAT instance
c Generated for testing
p cnf 5 3
1 2 3 0
-1 4 5 0
2 -4 -5 0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.cnf') as f:
            f.write(cnf_content)
            temp_file = f.name

        try:
            clauses, n_vars = load_dimacs_cnf(temp_file)

            assert n_vars == 5
            assert len(clauses) == 3
            # Check each clause
            assert all(len(clause) == 3 for clause in clauses)
        finally:
            os.unlink(temp_file)


class TestRandomBaseline:
    """Tests for random assignment baseline."""

    def test_random_baseline_bounds(self):
        """Test that random baseline returns valid probabilities."""
        n_vars = 20
        m_clauses = 84
        clauses = random_3sat(n_vars, m_clauses, seed=42)

        rho = random_assignment_baseline(clauses, n_vars, num_trials=10)

        assert 0.0 <= rho <= 1.0

    def test_random_baseline_approximate_seven_eighths(self):
        """Test that random baseline approximates 7/8 for 3-SAT."""
        n_vars = 100
        m_clauses = 420
        clauses = random_3sat(n_vars, m_clauses, seed=123)

        rho = random_assignment_baseline(clauses, n_vars, num_trials=50)

        # Should be around 0.875 for 3-SAT
        assert 0.82 < rho < 0.92


class TestSingleBenchmark:
    """Tests for single benchmark runner."""

    def test_run_single_benchmark(self):
        """Test running benchmark on single instance."""
        n_vars = 20
        m_clauses = 84
        clauses = random_3sat(n_vars, m_clauses, seed=42)

        results = run_single_benchmark(clauses, n_vars, num_runs=3, include_walksat=False)

        # Check structure
        assert 'helical_rho' in results
        assert 'uniform_rho' in results
        assert 'random_rho' in results
        assert 'helical_time' in results

        # Check counts
        assert len(results['helical_rho']) == 3
        assert len(results['uniform_rho']) == 3

        # Check validity
        assert all(0.0 <= rho <= 1.0 for rho in results['helical_rho'])
        assert all(0.0 <= rho <= 1.0 for rho in results['uniform_rho'])
        assert 0.0 <= results['random_rho'] <= 1.0

    def test_benchmark_metadata(self):
        """Test that benchmark includes correct metadata."""
        n_vars = 50
        m_clauses = 210
        clauses = random_3sat(n_vars, m_clauses, seed=999)

        results = run_single_benchmark(clauses, n_vars, num_runs=2)

        assert results['n_vars'] == n_vars
        assert results['m_clauses'] == m_clauses


class TestAggregateResults:
    """Tests for result aggregation."""

    def test_aggregate_single_instance(self):
        """Test aggregation with single instance."""
        n_vars = 20
        m_clauses = 84
        clauses = random_3sat(n_vars, m_clauses, seed=42)

        results = run_single_benchmark(clauses, n_vars, num_runs=5)
        aggregated = aggregate_results([results])

        assert 'helical_mean' in aggregated
        assert 'helical_ci' in aggregated
        assert 'uniform_mean' in aggregated
        assert 'uniform_ci' in aggregated
        assert aggregated['n'] == n_vars
        assert aggregated['m'] == m_clauses

    def test_aggregate_multiple_instances(self):
        """Test aggregation with multiple instances."""
        n_vars = 30
        m_clauses = 126

        results_list = []
        for seed in range(3):
            clauses = random_3sat(n_vars, m_clauses, seed=seed)
            results = run_single_benchmark(clauses, n_vars, num_runs=3)
            results_list.append(results)

        aggregated = aggregate_results(results_list)

        # Should have statistics from all runs
        assert 0.0 <= aggregated['helical_mean'] <= 1.0
        assert aggregated['helical_ci'] >= 0.0
        assert 0.0 <= aggregated['uniform_mean'] <= 1.0


class TestIntegration:
    """Integration tests for full benchmark pipeline."""

    def test_full_pipeline_small(self):
        """Test complete benchmark pipeline with small instance."""
        from benchmarks import run_benchmarks

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'test_results.md')

            results = run_benchmarks(
                suite='random',
                suite_size='small',
                num_instances=2,
                num_runs=2,
                output=output_file,
                include_walksat=False
            )

            # Check results generated
            assert len(results) > 0

            # Check output file created
            assert os.path.exists(output_file)

            # Check file has content
            with open(output_file, 'r') as f:
                content = f.read()
                assert 'Helical SAT Heuristic' in content
                assert 'Benchmark Results' in content
