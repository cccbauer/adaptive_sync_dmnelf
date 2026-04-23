#!/usr/bin/env python
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
import mne

SUBJECT = "sub-dmnelf001"

# Load feedback run
eeg_file = Path("/projects/swglab/data/DMNELF/analysis/MNE/bids/derivatives/preprocessed") / SUBJECT / "ses-dmnelf" / "eeg" / (SUBJECT + "_ses-dmnelf_task-feedback_run-01_desc-preproc_eeg.fif")
raw = mne.io.read_raw_fif(str(eeg_file), preload=True, verbose=False)

# Auto-detect and interpolate bad channels
data = raw.get_data()
variances = np.var(data, axis=1)
median_var = np.median(variances)
bad_channels = [raw.ch_names[i] for i, v in enumerate(variances) if v > 3*median_var or raw.ch_names[i] == "ECG"]
raw.info["bads"] = bad_channels
raw.interpolate_bads(reset_bads=True)

picks = mne.pick_types(raw.info, eeg=True, exclude=["ECG"])
eeg_clean = raw.get_data(picks=picks)[:, :15000]  # 60s

# Compute beta-band envelope
sos = butter(4, [13, 30], btype="bandpass", fs=250, output="sos")
beta_filt = sosfiltfilt(sos, eeg_clean, axis=1)
analytic = hilbert(beta_filt, axis=1)
envelope = np.abs(analytic)
mean_envelope = np.mean(envelope, axis=0)

# Normalize
mean_envelope = (mean_envelope - np.mean(mean_envelope)) / np.std(mean_envelope)

# Simulate Kuramoto
sys.path.insert(0, "/projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/src")
from models.kuramoto import KuramotoModel, KuramotoParams

model = KuramotoModel(KuramotoParams(
    n_oscillators=eeg_clean.shape[0],
    coupling_strength=10.0,
    dt=0.01
))

def input_func(t):
    return np.ones(eeg_clean.shape[0]) * 5.0

sync_sim, _ = model.simulate(60.0, input_func)

# Downsample for plotting
time_eeg = np.arange(len(mean_envelope)) / 250.0
time_sim = np.arange(len(sync_sim)) * 0.01

# Resample sync to EEG rate
sync_resampled = np.interp(time_eeg, time_sim, sync_sim)
sync_norm = (sync_resampled - np.mean(sync_resampled)) / np.std(sync_resampled)

# Plot
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(time_eeg, mean_envelope, linewidth=2, alpha=0.7, 
        label='Real Beta Envelope (EEG)', color='#028090')
ax.plot(time_eeg, sync_norm, linewidth=2, alpha=0.8,
        label='Simulated R(t) (Kuramoto)', color='#E63946', linestyle='--')

ax.set_xlabel('Time (seconds)', fontsize=14)
ax.set_ylabel('Normalized Amplitude', fontsize=14)
ax.set_title('Feedback Run: Simulated vs Real Dynamics (sub-dmnelf001)', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(0, 60)

plt.tight_layout()

output_file = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/results") / "figures" / "timeseries_feedback_comparison.png"
output_file.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
