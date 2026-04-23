"""
plot_training_results.py

Plot training curves for all three conditions.
Run after fetch_results.py downloads the .pkl files.

Usage:
    python deploy_scripts/plot_training_results.py

Author: Clemens Bauer
Date: April 2026
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def load_results(condition):
    """Load training results for a condition."""
    results_file = config.LOCAL_RESULTS / "rl_agents" / f"results_{condition}.pkl"
    
    if not results_file.exists():
        print(f"Warning: {results_file} not found")
        return None
    
    with open(results_file, 'rb') as f:
        return pickle.load(f)

def plot_all_conditions():
    """Plot comparison across all three conditions."""
    
    conditions = ['feedback', 'shortrest', 'rest']
    colors = {'feedback': 'red', 'shortrest': 'orange', 'rest': 'blue'}
    targets = {'feedback': 0.8, 'shortrest': 0.5, 'rest': 0.3}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Load all results
    all_results = {}
    for cond in conditions:
        results = load_results(cond)
        if results:
            all_results[cond] = results
    
    if not all_results:
        print("No results found! Run fetch_results.py first.")
        return
    
    # Plot 1: Cumulative Reward
    ax = axes[0, 0]
    for cond, results in all_results.items():
        ax.plot(results['episode_rewards'], label=cond.capitalize(), 
                color=colors[cond], linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Learning Curves - Reward')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Mean Synchronization
    ax = axes[0, 1]
    for cond, results in all_results.items():
        ax.plot(results['episode_sync_means'], label=cond.capitalize(),
                color=colors[cond], linewidth=2)
        ax.axhline(targets[cond], color=colors[cond], linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean R(t)')
    ax.set_title('Synchronization Level (dashed = target)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Mean Energy
    ax = axes[1, 0]
    for cond, results in all_results.items():
        ax.plot(results['episode_energy_means'], label=cond.capitalize(),
                color=colors[cond], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean E(t)')
    ax.set_title('Energy Consumption')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Final Stats Comparison
    ax = axes[1, 1]
    cond_names = []
    final_sync = []
    final_energy = []
    
    for cond in conditions:
        if cond in all_results:
            cond_names.append(cond.capitalize())
            final_sync.append(all_results[cond]['episode_sync_means'][-1])
            final_energy.append(all_results[cond]['episode_energy_means'][-1])
    
    x = np.arange(len(cond_names))
    width = 0.35
    
    ax.bar(x - width/2, final_sync, width, label='Final R(t)', color='steelblue')
    ax.bar(x + width/2, [e/10 for e in final_energy], width, 
           label='Final E(t)/10', color='coral')
    
    ax.set_ylabel('Value')
    ax.set_title('Final Episode Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = config.LOCAL_RESULTS / "training_curves" / "all_conditions_comparison.png"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    print("=" * 70)
    print("Training Results Visualization")
    print("=" * 70)
    plot_all_conditions()