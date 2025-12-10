#!/usr/bin/env python3
"""
Download SATLIB benchmark instances.

This script downloads common SATLIB benchmark suites for testing
the Helical SAT Heuristic.
"""

import urllib.request
import tarfile
import os
from pathlib import Path
import sys


SATLIB_BASE_URL = "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT"

BENCHMARKS = {
    'uf20-91': {
        'url': f'{SATLIB_BASE_URL}/uf20-91.tar.gz',
        'filename': 'uf20-91.tar.gz',
        'description': '1000 instances, 20 variables, 91 clauses'
    },
    'uf50-218': {
        'url': f'{SATLIB_BASE_URL}/uf50-218.tar.gz',
        'filename': 'uf50-218.tar.gz',
        'description': '1000 instances, 50 variables, 218 clauses'
    },
    'uf100-430': {
        'url': f'{SATLIB_BASE_URL}/uf100-430.tar.gz',
        'filename': 'uf100-430.tar.gz',
        'description': '1000 instances, 100 variables, 430 clauses'
    },
    'uf150-645': {
        'url': f'{SATLIB_BASE_URL}/uf150-645.tar.gz',
        'filename': 'uf150-645.tar.gz',
        'description': '100 instances, 150 variables, 645 clauses'
    },
    'uf200-860': {
        'url': f'{SATLIB_BASE_URL}/uf200-860.tar.gz',
        'filename': 'uf200-860.tar.gz',
        'description': '100 instances, 200 variables, 860 clauses'
    }
}


def download_file(url, dest_path):
    """Download a file from URL to destination path."""
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def extract_tarball(tar_path, extract_dir):
    """Extract a tar.gz file to directory."""
    print(f"Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print(f"  Extracted to {extract_dir}")
        return True
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False


def download_benchmark(benchmark_name, data_dir='data/satlib'):
    """Download and extract a single benchmark suite."""
    if benchmark_name not in BENCHMARKS:
        print(f"Unknown benchmark: {benchmark_name}")
        print(f"Available: {', '.join(BENCHMARKS.keys())}")
        return False

    bench = BENCHMARKS[benchmark_name]
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    tar_path = data_path / bench['filename']

    # Download if not already present
    if not tar_path.exists():
        if not download_file(bench['url'], tar_path):
            return False
    else:
        print(f"{tar_path} already exists, skipping download")

    # Extract
    if not extract_tarball(tar_path, data_path):
        return False

    # Clean up tarball
    try:
        os.remove(tar_path)
        print(f"  Removed {tar_path}")
    except:
        pass

    print(f"✓ {benchmark_name}: {bench['description']}")
    return True


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download SATLIB benchmark instances'
    )
    parser.add_argument(
        '--suite',
        type=str,
        choices=['all', 'small', 'medium', 'large'] + list(BENCHMARKS.keys()),
        default='small',
        help='Which benchmark suite(s) to download'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/satlib',
        help='Directory to save benchmarks'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SATLIB Benchmark Downloader")
    print("=" * 70)
    print()

    if args.suite == 'all':
        benchmarks = list(BENCHMARKS.keys())
    elif args.suite == 'small':
        benchmarks = ['uf20-91', 'uf50-218']
    elif args.suite == 'medium':
        benchmarks = ['uf100-430']
    elif args.suite == 'large':
        benchmarks = ['uf150-645', 'uf200-860']
    else:
        benchmarks = [args.suite]

    success_count = 0
    for benchmark in benchmarks:
        print()
        if download_benchmark(benchmark, args.data_dir):
            success_count += 1
        print()

    print("=" * 70)
    print(f"Downloaded {success_count}/{len(benchmarks)} benchmark suites")
    print("=" * 70)

    if success_count > 0:
        print()
        print("You can now run benchmarks with:")
        print(f"  python benchmarks.py --suite satlib --data-dir {args.data_dir}")


if __name__ == '__main__':
    main()
