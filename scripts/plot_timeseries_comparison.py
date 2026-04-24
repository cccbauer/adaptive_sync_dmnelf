"""
plot_timeseries_comparison.py

Visualize simulated BOLD vs real PDA time series.
Verifies the r=-0.76 correlation.

Usage:
    python scripts/plot_timeseries_comparison.py --condition feedback

Author: Clemens Bauer
Date: April 2026
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

CONDITION = sys.argv[1] if len(sys.argv) > 1 else "feedback"
SUBJECT = "sub-dmnelf001"

# Load full validation
val_file = Path("results/validation") / f"{SUBJECT}_{CONDITION}_full_validation.pkl"

if not val_file.exists():
    print(f"ERROR: {val_file} not found")
    sys.exit(1)

with open(val_file, 'rb') as f:
    data = pickle.load(f)

print(f"Loaded {CONDITION} validation data")
print(f"Keys: {list(data.keys())}")

traj = data['trajectories']
corr = data['correlations']

# Extract time series
sync_sim = np.array(traj['sync_sim'])
bold_sim = np.array(traj['bold_sim'])
pda_difumo = np.array(traj['pda_difumo'])
bold_resampled = np.array(traj['bold_resampled'])

print(f"\nTime series lengths:")
print(f"  Sync (Kuramoto): {len(sync_sim)} points (0.01s steps = {len(sync_sim)*0.01:.1f}s)")
print(f"  BOLD (simulated): {len(bold_sim)} points")
print(f"  PDA (real fMRI): {len(pda_difumo)} volumes (TR=2s = {len(pda_difumo)*2:.1f}s)")
print(f"  BOLD resampled: {len(bold_resampled)} points")

print(f"\nCorrelation: r={corr['difumo']:.3f}")

# Normalize both for visualization
pda_norm = (pda_difumo - np.mean(pda_difumo)) / np.std(pda_difumo)
bold_norm = (bold_resampled - np.mean(bold_resampled)) / np.std(bold_resampled)

# Recompute to verify
r_verify, p = pearsonr(pda_difumo, bold_resampled)
print(f"Verified correlation: r={r_verify:.3f}, p={p:.4f}")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Subplot 1: Simulated sync
time_sim = np.arange(len(sync_sim)) * 0.01
axes[0].plot(time_sim, sync_sim, linewidth=2, color='#E63946', alpha=0.8)
axes[0].set_ylabel('R(t)', fontsize=12)
axes[0].set_title(f'Simulated Synchronization (Kuramoto, K={data.get("coupling_k", 10)})', 
                  fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 1)

# Subplot 2: Simulated BOLD vs Real PDA (normalized overlay)
time_fmri = np.arange(len(pda_difumo)) * 2.0
axes[1].plot(time_fmri, pda_norm, linewidth=2.5, color='#028090', 
             label='Real PDA (CEN-DMN)', alpha=0.9)
axes[1].plot(time_fmri, bold_norm, linewidth=2, color='#E63946', 
             linestyle='--', label='Simulated BOLD', alpha=0.8)
axes[1].set_ylabel('Normalized Signal', fontsize=12)
axes[1].set_title(f'BOLD vs PDA Comparison (r={r_verify:.3f})', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11, loc='upper right')
axes[1].grid(alpha=0.3)

# Subplot 3: Scatter plot
axes[2].scatter(pda_difumo, bold_resampled, alpha=0.6, s=60, color='#028090', edgecolors='black', linewidth=0.5)
z = np.polyfit(pda_difumo, bold_resampled, 1)
p = np.poly1d(z)
axes[2].plot(pda_difumo, p(pda_difumo), "r--", linewidth=2, alpha=0.8, 
             label=f'Linear fit: r={r_verify:.3f}')
axes[2].set_xlabel('Real PDA (CEN-DMN)', fontsize=12)
axes[2].set_ylabel('Simulated BOLD', fontsize=12)
axes[2].set_title('Point-by-Point Correlation', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=11)
axes[2].grid(alpha=0.3)

plt.suptitle(f'{SUBJECT} - {CONDITION.capitalize()} Run: Simulated vs Real fMRI', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()

output = Path("results/figures") / f"timeseries_{CONDITION}_comparison.png"
output.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output}")
plt.show()
