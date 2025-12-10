#!/usr/bin/env python3
"""
Helical SAT Heuristic - A spectral graph approach for Max-3-SAT approximation.

This module implements a one-shot spectral graph heuristic for approximating
Max-3-SAT satisfaction ratio (ρ). It builds a clause-literal graph with edges
weighted by cosine phases on logarithmic variable indices.

Inspired by xAI Grok conversation. Tag @grok for feedback.
"""

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from itertools import combinations
from sklearn.feature_selection import mutual_info_regression
import time
from typing import List, Tuple


def random_3sat(n_vars: int, m_clauses: int, seed: int = 42) -> List[List[int]]:
    """
    Generate a random 3-SAT instance.

    Args:
        n_vars: Number of variables
        m_clauses: Number of clauses
        seed: Random seed for reproducibility

    Returns:
        List of clauses, where each clause is a list of literals.
        Literals are represented as integers from -n_vars to n_vars (excluding 0),
        where positive values indicate the variable and negative values indicate negation.
    """
    np.random.seed(seed)
    clauses = []

    for _ in range(m_clauses):
        # Select 3 distinct variables randomly
        vars_selected = np.random.choice(n_vars, size=3, replace=False)
        # Randomly negate each variable (True = positive, False = negative)
        signs = np.random.choice([1, -1], size=3)
        # Create clause with literals (1-indexed)
        clause = [(vars_selected[i] + 1) * signs[i] for i in range(3)]
        clauses.append(clause)

    return clauses


def evaluate_sat(clauses: List[List[int]], assign: np.ndarray) -> float:
    """
    Evaluate satisfaction ratio for a given assignment.

    Args:
        clauses: List of clauses (each clause is a list of literals)
        assign: Assignment vector where assign[i] is the sign for variable i

    Returns:
        Satisfaction ratio (fraction of satisfied clauses)
    """
    sat_count = 0

    for clause in clauses:
        # A clause is satisfied if at least one literal is true
        clause_satisfied = any(
            (lit > 0 and assign[abs(lit) - 1] > 0) or
            (lit < 0 and assign[abs(lit) - 1] < 0)
            for lit in clause
        )
        if clause_satisfied:
            sat_count += 1

    return sat_count / len(clauses)


def helical_sat_approx(
    clauses: List[List[int]],
    n_vars: int,
    omega: float = 0.3,
    N: int = 20000
) -> Tuple[float, float]:
    """
    Helical SAT approximation using spectral graph method.

    Builds a clause-literal graph with edges weighted by cosine phases on
    logarithmic variable indices: w_uv = cos(ω (θ_u - θ_v)), where
    θ ∝ log(var+1). Uses the lowest Laplacian eigenvector to assign variables.

    Args:
        clauses: List of clauses (each clause is a list of literals)
        n_vars: Number of variables
        omega: Frequency parameter for helical weighting (default: 0.3)
        N: Normalization constant for theta calculation (default: 20000)

    Returns:
        Tuple of (satisfaction_ratio, mutual_info_bound)
    """
    # Build graph with variable nodes
    G = nx.Graph()
    for i in range(n_vars):
        G.add_node(i)

    # Add edges between variables in same clauses with helical weights
    for clause in clauses:
        vars_in_clause = [abs(lit) - 1 for lit in clause]

        # Add edges for all pairs of variables in the clause
        for u, v in combinations(set(vars_in_clause), 2):
            # Compute helical phase angles based on logarithmic indices
            theta_u = 2 * np.pi * np.log(u + 1) / N
            theta_v = 2 * np.pi * np.log(v + 1) / N

            # Edge weight is cosine of phase difference scaled by omega
            w = np.cos(omega * (theta_u - theta_v))

            # Add or update edge weight
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)

    # Compute Laplacian matrix
    L = nx.laplacian_matrix(G, weight='weight').tocsc().astype(float)

    # Find smallest eigenvector (Fiedler vector)
    _, vec = eigsh(L, k=1, which='SM', maxiter=200)

    # Assignment based on eigenvector signs
    assign = np.sign(vec[:, 0])

    # Handle zero assignments (assign randomly)
    zero_indices = np.where(assign == 0)[0]
    if len(zero_indices) > 0:
        assign[zero_indices] = np.random.choice([1, -1], size=len(zero_indices))

    # Evaluate satisfaction ratio
    rho = evaluate_sat(clauses, assign)

    # Compute mutual information bound
    # For each clause, compute average edge weight of edges within the clause
    clause_avg_weights = []
    clause_sizes = []

    for clause in clauses:
        vars_in_clause = list(set([abs(lit) - 1 for lit in clause]))
        clause_sizes.append(len(vars_in_clause))

        # Get edge weights for this clause
        edge_weights = []
        for u, v in combinations(vars_in_clause, 2):
            if G.has_edge(u, v):
                edge_weights.append(G[u][v]['weight'])

        if edge_weights:
            clause_avg_weights.append(np.mean(edge_weights))
        else:
            clause_avg_weights.append(0.0)

    clause_avg_weights = np.array(clause_avg_weights)
    clause_sizes = np.array(clause_sizes)

    # Compute mutual information between clause average weights and sizes
    if len(clause_avg_weights) > 1:
        I = mutual_info_regression(clause_avg_weights.reshape(-1, 1), clause_sizes)
        bound = 1 - np.exp(-np.mean(I) / np.log(len(clauses)))
    else:
        bound = 0.0

    return rho, bound


def uniform_sat_baseline(clauses: List[List[int]], n_vars: int) -> float:
    """
    Baseline uniform spectral method with constant edge weight (-1).

    Args:
        clauses: List of clauses (each clause is a list of literals)
        n_vars: Number of variables

    Returns:
        Satisfaction ratio
    """
    # Build graph with variable nodes
    G = nx.Graph()
    for i in range(n_vars):
        G.add_node(i)

    # Add edges with uniform weight
    for clause in clauses:
        vars_in_clause = [abs(lit) - 1 for lit in clause]

        for u, v in combinations(set(vars_in_clause), 2):
            w = -1.0

            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)

    # Compute Laplacian matrix
    L = nx.laplacian_matrix(G, weight='weight').tocsc().astype(float)

    # Find smallest eigenvector
    _, vec = eigsh(L, k=1, which='SM', maxiter=200)

    # Assignment based on eigenvector signs
    assign = np.sign(vec[:, 0])

    # Handle zero assignments
    zero_indices = np.where(assign == 0)[0]
    if len(zero_indices) > 0:
        assign[zero_indices] = np.random.choice([1, -1], size=len(zero_indices))

    # Evaluate satisfaction ratio
    rho = evaluate_sat(clauses, assign)

    return rho


def run_benchmark(
    n_vars: int,
    m_clauses: int,
    num_seeds: int = 5
) -> dict:
    """
    Run benchmark for given problem size over multiple random seeds.

    Args:
        n_vars: Number of variables
        m_clauses: Number of clauses
        num_seeds: Number of random seeds to average over

    Returns:
        Dictionary with benchmark results including means, CIs, and runtime
    """
    helical_rhos = []
    uniform_rhos = []
    runtimes = []

    for seed in range(42, 42 + num_seeds):
        # Generate random 3-SAT instance
        clauses = random_3sat(n_vars, m_clauses, seed=seed)

        # Run helical method
        start_time = time.time()
        helical_rho, _ = helical_sat_approx(clauses, n_vars)
        runtime_ms = (time.time() - start_time) * 1000

        helical_rhos.append(helical_rho)
        runtimes.append(runtime_ms)

        # Run uniform baseline
        uniform_rho = uniform_sat_baseline(clauses, n_vars)
        uniform_rhos.append(uniform_rho)

    # Compute statistics
    helical_mean = np.mean(helical_rhos)
    helical_std = np.std(helical_rhos, ddof=1)
    helical_ci = 1.96 * helical_std / np.sqrt(num_seeds)  # 95% CI

    uniform_mean = np.mean(uniform_rhos)
    uniform_std = np.std(uniform_rhos, ddof=1)
    uniform_ci = 1.96 * uniform_std / np.sqrt(num_seeds)

    runtime_mean = np.mean(runtimes)

    return {
        'n': n_vars,
        'm': m_clauses,
        'helical_mean': helical_mean,
        'helical_ci': helical_ci,
        'uniform_mean': uniform_mean,
        'uniform_ci': uniform_ci,
        'runtime_ms': runtime_mean
    }


def main():
    """Run benchmarks and display results."""
    print("=" * 80)
    print("Helical SAT Heuristic - Benchmarking")
    print("=" * 80)
    print()

    # Benchmark configurations (m ≈ 4.2n for phase transition)
    benchmarks = [
        (20, 84),
        (100, 420),
        (200, 840)
    ]

    results = []

    for n, m in benchmarks:
        print(f"Running benchmark: n={n}, m={m}...")
        result = run_benchmark(n, m, num_seeds=5)
        results.append(result)
        print(f"  Helical ρ: {result['helical_mean']:.4f} ± {result['helical_ci']:.4f}")
        print(f"  Uniform ρ: {result['uniform_mean']:.4f} ± {result['uniform_ci']:.4f}")
        print(f"  Runtime: {result['runtime_ms']:.2f} ms")
        print()

    # Display results table
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    print("| {:>4} | {:>4} | {:>13} | {:>8} | {:>13} | {:>8} | {:>11} |".format(
        "n", "m", "Avg ρ Helical", "CI", "Avg ρ Uniform", "CI", "Runtime ms"
    ))
    print("|" + "-" * 78 + "|")

    for r in results:
        print("| {:>4} | {:>4} | {:>13.4f} | {:>8.4f} | {:>13.4f} | {:>8.4f} | {:>11.2f} |".format(
            r['n'], r['m'], r['helical_mean'], r['helical_ci'],
            r['uniform_mean'], r['uniform_ci'], r['runtime_ms']
        ))

    print()
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
