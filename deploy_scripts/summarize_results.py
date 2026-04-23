"""
summarize_results.py

Print summary statistics for all trained conditions.

Usage:
    python deploy_scripts/summarize_results.py

Author: Clemens Bauer
Date: April 2026
"""

import sys
from pathlib import Path
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def summarize_condition(condition):
    """Print summary for one condition."""
    results_file = config.LOCAL_RESULTS / "rl_agents" / f"results_{condition}.pkl"
    
    if not results_file.exists():
        print(f"  ⚠ Results not found: {results_file}")
        return None
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print(f"\n{condition.upper()}:")
    print(f"  Episodes: {len(results['episode_rewards'])}")
    
    # Sync progression
    print(f"  R(t) progression:")
    print(f"    Episode 1:   {results['episode_sync_means'][0]:.3f}")
    print(f"    Episode 10:  {results['episode_sync_means'][9]:.3f}")
    print(f"    Episode 50:  {results['episode_sync_means'][49]:.3f}")
    print(f"    Episode 100: {results['episode_sync_means'][99]:.3f}")
    
    # Target comparison
    targets = {'feedback': 0.8, 'shortrest': 0.5, 'rest': 0.3}
    target = targets.get(condition, 0.0)
    final_r = results['episode_sync_means'][-1]
    error = abs(target - final_r)
    print(f"  Target R: {target:.1f}, Final R: {final_r:.3f}, Error: {error:.3f}")
    
    # Reward improvement
    reward_start = results['episode_rewards'][0]
    reward_end = results['episode_rewards'][-1]
    reward_improve = reward_end - reward_start
    print(f"  Reward: {reward_start:.1f} → {reward_end:.1f} ({reward_improve:+.1f})")
    
    # Energy
    energy_start = results['episode_energy_means'][0]
    energy_end = results['episode_energy_means'][-1]
    print(f"  Energy: {energy_start:.2f} → {energy_end:.2f}")
    
    return results

def main():
    print("=" * 70)
    print("Training Results Summary")
    print("=" * 70)
    
    conditions = ['feedback', 'shortrest', 'rest']
    all_results = {}
    
    for cond in conditions:
        results = summarize_condition(cond)
        if results:
            all_results[cond] = results
    
    # Overall comparison
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("Comparison to Hall et al. (2025)")
        print("=" * 70)
        print(f"{'Condition':<12} {'Hall R':<8} {'Our R':<8} {'Match':<8}")
        print("-" * 70)
        
        hall_results = {'feedback': 0.8, 'shortrest': 0.5, 'rest': 0.3}
        
        for cond in conditions:
            if cond in all_results:
                hall_r = hall_results[cond]
                our_r = all_results[cond]['episode_sync_means'][-1]
                match = "✓" if abs(hall_r - our_r) < 0.2 else "✗"
                print(f"{cond:<12} {hall_r:<8.1f} {our_r:<8.3f} {match:<8}")

if __name__ == "__main__":
    main()
