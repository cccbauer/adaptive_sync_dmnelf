#!/usr/bin/env python
import mne
import numpy as np
from pathlib import Path

SUBJECT = "sub-dmnelf001"

print("Analyzing channel variance: " + SUBJECT)
print("=" * 70)

# Load all feedback runs
eeg_base = Path("/projects/swglab/data/DMNELF/analysis/MNE/bids/derivatives/preprocessed") / SUBJECT / "ses-dmnelf" / "eeg"
eeg_files = sorted(eeg_base.glob("*task-feedback*_eeg.fif"))

all_variances = []
channel_names = None

for eeg_file in eeg_files:
    raw = mne.io.read_raw_fif(str(eeg_file), preload=True, verbose=False)
    if channel_names is None:
        channel_names = raw.ch_names
    
    data = raw.get_data()
    variances = np.var(data, axis=1)
    all_variances.append(variances)
    print("  " + str(eeg_file.name) + ": " + str(data.shape))

# Mean variance across runs
mean_var = np.mean(all_variances, axis=0)

# Sort by variance (highest = noisiest)
sorted_idx = np.argsort(mean_var)[::-1]

print("\n" + "=" * 70)
print("Channel Variance Ranking (noisiest first)")
print("=" * 70)
print("Rank  Channel    Variance      Relative")
print("-" * 70)

for rank, idx in enumerate(sorted_idx[:15]):  # Top 15
    ch_name = channel_names[idx]
    var = mean_var[idx]
    rel_var = var / np.median(mean_var)
    print(f"{rank+1:2d}    {ch_name:8s}   {var:.2e}    {rel_var:.2f}x")

print("\n" + "=" * 70)
print("Specific Channels of Interest")
print("=" * 70)

check_channels = ["TP9", "TP10", "ECG", "Fp1", "Fp2", "T7", "T8"]
for ch in check_channels:
    if ch in channel_names:
        idx = channel_names.index(ch)
        var = mean_var[idx]
        rank = np.where(sorted_idx == idx)[0][0] + 1
        rel_var = var / np.median(mean_var)
        print(f"{ch:8s}: rank {rank:2d}/32, variance={var:.2e} ({rel_var:.1f}x median)")