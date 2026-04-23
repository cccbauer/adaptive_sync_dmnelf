"""
fetch_results.py

Download trained models and results from Explorer cluster.
Matches microstate_pda fetch pattern.

Usage:
    python deploy_scripts/fetch_results.py [--models] [--plots] [--all]

Author: Clemens Bauer
Date: April 2026
"""

import sys
from pathlib import Path
import subprocess
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def fetch_models():
    """Download trained RL agents"""
    print("\nFetching trained models...")
    
    local_dir = config.LOCAL_RESULTS / "rl_agents"
    local_dir.mkdir(exist_ok=True, parents=True)
    
    # Download specific model files directly
    model_files = [
        'qlearning_feedback.pkl',
        'qlearning_shortrest.pkl', 
        'qlearning_rest.pkl',
        'results_feedback.pkl',
        'results_shortrest.pkl',
        'results_rest.pkl'
    ]
    
    for filename in model_files:
        cluster_file = config.CLUSTER_MODELS / filename
        local_file = local_dir / filename
        
        cmd = config.scp_from_cluster(cluster_file, local_file)
        result = subprocess.run(cmd, shell=True, capture_output=True)
        
        if result.returncode == 0:
            print(f"  ✓ Downloaded: {filename}")
        else:
            print(f"  ⚠ Not found: {filename}")
            
def fetch_training_curves():
    """Download training results and plots"""
    print("\nFetching training curves...")
    
    local_dir = config.LOCAL_RESULTS / "training_curves"
    local_dir.mkdir(exist_ok=True, parents=True)
    
    # Training results pickle
    cluster_file = config.CLUSTER_MODELS / "training_results.pkl"
    local_file = local_dir / "training_results.pkl"
    
    try:
        cmd = config.scp_from_cluster(cluster_file, local_file)
        subprocess.run(cmd, shell=True, check=True)
        print("  ✓ Downloaded: training_results.pkl")
    except subprocess.CalledProcessError:
        print("  ⚠ training_results.pkl not found")

def fetch_validation_results():
    """Download validation results"""
    print("\nFetching validation results...")
    
    local_dir = config.LOCAL_RESULTS / "validation"
    local_dir.mkdir(exist_ok=True, parents=True)
    
    # Fetch specific files directly
    subjects = ["sub-dmnelf001"]  # Expand later
    conditions = ["feedback", "shortrest", "rest"]
    
    for subj in subjects:
        for cond in conditions:
            filename = f"{subj}_{cond}_validation.pkl"
            cluster_file = config.CLUSTER_RESULTS / "validation" / filename
            local_file = local_dir / filename
            
            cmd = config.scp_from_cluster(cluster_file, local_file)
            result = subprocess.run(cmd, shell=True, capture_output=True)
            
            if result.returncode == 0:
                print(f"  ✓ Downloaded: {filename}")
            else:
                print(f"  ⚠ Not found: {filename}")

def fetch_validation_plots():
    """Download validation plots comparing real vs simulated"""
    print("\nFetching validation plots...")
    
    local_dir = config.LOCAL_RESULTS / "validation_plots"
    local_dir.mkdir(exist_ok=True, parents=True)
    
    # List available plots
    cluster_plots = config.CLUSTER_RESULTS / "validation"
    
    cmd = f"ls {cluster_plots}/*.png 2>/dev/null || echo 'No plots found'"
    result = subprocess.run(
        config.ssh_command(cmd),
        shell=True,
        capture_output=True,
        text=True
    )
    
    if "No plots found" in result.stdout:
        print("  No plots found on cluster")
        return
    
    plot_files = [f.strip() for f in result.stdout.strip().split('\n')]
    
    for plot_file in plot_files:
        filename = Path(plot_file).name
        local_file = local_dir / filename
        
        cmd = config.scp_from_cluster(plot_file, local_file)
        subprocess.run(cmd, shell=True, check=True)
        print(f"  ✓ Downloaded: {filename}")

def fetch_logs(job_id=None):
    """Download SLURM logs"""
    print("\nFetching logs...")
    
    local_dir = config.LOCAL_BASE / "logs"
    local_dir.mkdir(exist_ok=True, parents=True)
    
    if job_id:
        # Fetch specific job log
        patterns = [
            f"rl_train_{job_id}.out",
            f"rl_train_{job_id}.err"
        ]
    else:
        # Fetch latest logs
        cmd = f"ls -t {config.CLUSTER_LOGS}/rl_train_*.out 2>/dev/null | head -5"
        result = subprocess.run(
            config.ssh_command(cmd),
            shell=True,
            capture_output=True,
            text=True
        )
        
        if not result.stdout.strip():
            print("  No logs found")
            return
        
        patterns = [Path(f.strip()).name for f in result.stdout.strip().split('\n')]
    
    for pattern in patterns:
        cluster_file = config.CLUSTER_LOGS / pattern
        local_file = local_dir / pattern
        
        try:
            cmd = config.scp_from_cluster(cluster_file, local_file)
            subprocess.run(cmd, shell=True, check=True)
            print(f"  ✓ Downloaded: {pattern}")
        except subprocess.CalledProcessError:
            print(f"  ⚠ {pattern} not found")

def fetch_all():
    """Download everything"""
    fetch_models()
    fetch_training_curves()
    fetch_validation_results()  # Changed from fetch_validation_plots
    fetch_logs()

def main():
    parser = argparse.ArgumentParser(description="Fetch results from cluster")
    parser.add_argument("--models", action="store_true", help="Fetch trained models")
    parser.add_argument("--plots", action="store_true", help="Fetch validation plots")
    parser.add_argument("--logs", action="store_true", help="Fetch latest logs")
    parser.add_argument("--job-id", type=str, help="Fetch logs for specific job")
    parser.add_argument("--all", action="store_true", help="Fetch everything")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Fetching Results from Cluster")
    print("=" * 70)
    
    if args.all or (not any([args.models, args.plots, args.logs])):
        fetch_all()
    else:
        if args.models:
            fetch_models()
        if args.plots:
            fetch_validation_plots()
        if args.logs:
            fetch_logs(args.job_id)
    
    print("\n" + "=" * 70)
    print("Fetch complete!")
    print("=" * 70)
    print(f"\nResults saved to: {config.LOCAL_RESULTS}")

if __name__ == "__main__":
    main()
