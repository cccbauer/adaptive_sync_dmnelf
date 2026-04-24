"""
fetch_results.py

Download all results from cluster (training, validation, full trajectories).

Usage:
    python deploy_scripts/fetch_results.py [--models] [--validation] [--full] [--all]

Author: Clemens Bauer
Date: April 2026
"""

import subprocess
from pathlib import Path
import argparse

CLUSTER = "cccbauer@explorer.northeastern.edu"
MODELS_DIR = "/projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/models"
VALIDATION_DIR = "/projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/results/validation"

def fetch_models():
    """Download trained RL agents"""
    print("\nFetching trained models...")
    Path("results/rl_agents").mkdir(exist_ok=True, parents=True)
    
    files = ["qlearning_feedback", "qlearning_shortrest", "qlearning_rest",
             "results_feedback", "results_shortrest", "results_rest"]
    
    for f in files:
        cmd = f'scp "{CLUSTER}:{MODELS_DIR}/{f}.pkl" "results/rl_agents/" 2>&1 | grep -v "Loading"'
        result = subprocess.run(cmd, shell=True, capture_output=True)
        print(f"  {'✓' if result.returncode == 0 else '⚠'} {f}.pkl")

def fetch_validation():
    """Download validation results (summary only)"""
    print("\nFetching validation results...")
    Path("results/validation").mkdir(exist_ok=True, parents=True)
    
    for cond in ["feedback", "shortrest", "rest"]:
        f = f"sub-dmnelf001_{cond}_validation"
        cmd = f'scp "{CLUSTER}:{VALIDATION_DIR}/{f}.pkl" "results/validation/" 2>&1 | grep -v "Loading"'
        result = subprocess.run(cmd, shell=True, capture_output=True)
        print(f"  {'✓' if result.returncode == 0 else '⚠'} {f}.pkl")

def fetch_full_trajectories():
    """Download full validation with time series"""
    print("\nFetching full trajectories...")
    Path("results/validation").mkdir(exist_ok=True, parents=True)
    
    for cond in ["feedback", "shortrest", "rest"]:
        f = f"sub-dmnelf001_{cond}_full_validation"
        cmd = f'scp "{CLUSTER}:{VALIDATION_DIR}/{f}.pkl" "results/validation/" 2>&1 | grep -v "Loading"'
        result = subprocess.run(cmd, shell=True, capture_output=True)
        print(f"  {'✓' if result.returncode == 0 else '⚠'} {f}.pkl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--full", action="store_true", help="Fetch full trajectories")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Fetching Results from Cluster")
    print("=" * 70)
    
    if args.all or not any([args.models, args.validation, args.full]):
        fetch_models()
        fetch_validation()
        fetch_full_trajectories()
    else:
        if args.models:
            fetch_models()
        if args.validation:
            fetch_validation()
        if args.full:
            fetch_full_trajectories()
    
    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
