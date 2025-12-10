# SATLIB Data Directory

This directory contains benchmark instances for testing the Helical SAT Heuristic.

## SATLIB Benchmarks

The SATLIB library contains standard benchmark instances for SAT solvers. To download SATLIB instances:

### Manual Download

Visit [SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html) and download benchmark suites:

1. **uf20-91** (20 variables, 91 clauses) - Small instances
   - Download: `uf20-91.tar.gz`
   - Extract to: `data/satlib/uf20-91/`

2. **uf50-218** (50 variables, 218 clauses) - Medium instances
   - Download: `uf50-218.tar.gz`
   - Extract to: `data/satlib/uf50-218/`

3. **uf100-430** (100 variables, 430 clauses) - Large instances
   - Download: `uf100-430.tar.gz`
   - Extract to: `data/satlib/uf100-430/`

### Automated Download (Unix/Linux/Mac)

Use the provided download script:

```bash
python download_satlib.py
```

This will download and extract common SATLIB benchmarks to the appropriate directories.

## Directory Structure

```
data/
├── README.md
├── download_satlib.py
├── satlib/
│   ├── uf20-91/       # 1000 instances, 20 vars, 91 clauses
│   ├── uf50-218/      # 1000 instances, 50 vars, 218 clauses
│   ├── uf100-430/     # 1000 instances, 100 vars, 430 clauses
│   ├── uf150-645/     # 100 instances, 150 vars, 645 clauses
│   └── uf200-860/     # 100 instances, 200 vars, 860 clauses
└── hamlib/
    └── (HamLib instances)
```

## Sample Instance

A sample instance is provided in `satlib/sample-uf100-01.cnf` for testing.

## DIMACS CNF Format

SATLIB instances use the DIMACS CNF format:

```
c Comment lines start with 'c'
c
p cnf 100 430
1 2 3 0
-1 4 5 0
...
```

- Line starting with `p cnf`: Problem definition (vars, clauses)
- Other lines: Clauses (space-separated literals, ending with 0)
- Positive numbers: Variable is true
- Negative numbers: Variable is negated
- 0: End of clause marker
