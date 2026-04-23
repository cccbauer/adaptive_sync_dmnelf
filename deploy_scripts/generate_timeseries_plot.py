"""
generate_timeseries_plot.py

Generate time series comparison plot for feedback run.
Creates overlay of simulated R(t) and real beta-band envelope.

Usage:
    python deploy_scripts/generate_timeseries_plot.py --subject sub-dmnelf001

Author: Clemens Bauer
Date: April 2026
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, default="sub-dmnelf001")
args = parser.parse_args()

SUBJECT = args.subject

print("This script needs to run on cluster where EEG data is located.")
print("Creating deployment version...")

# Cluster script to generate the plot
CLUSTER_SCRIPT = '''#!/usr/bin/env python
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfiltfilt
import mne

SUBJECT = "''' + SUBJECT + '''"

# Load feedback run
eeg_file = Path("''' + str(config.CLUSTER_PREPROCESSED) + '''") / SUBJECT / "ses-dmnelf" / "eeg" / (SUBJECT + "_ses-dmnelf_task-feedback_run-01_desc-preproc_eeg.fif")
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
sys.path.insert(0, "''' + str(config.CLUSTER_BASE) + '''/src")
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

output_file = Path("''' + str(config.CLUSTER_RESULTS) + '''") / "figures" / "timeseries_feedback_comparison.png"
output_file.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
'''

# Write and deploy
local_file = config.LOCAL_SCRIPTS / "generate_timeseries_cluster.py"
with open(local_file, 'w') as f:
    f.write(CLUSTER_SCRIPT)

import subprocess
cmd = config.scp_to_cluster(local_file, config.CLUSTER_SCRIPTS / "generate_timeseries_cluster.py")
subprocess.run(cmd, shell=True, check=True)
print("✓ Deployed to cluster")

# Run it
cmd = f"python {config.CLUSTER_SCRIPTS / 'generate_timeseries_cluster.py'}"
print("Running on cluster...")
result = subprocess.run(config.ssh_command(cmd), shell=True, capture_output=True, text=True)
print(result.stdout)

if "Saved:" in result.stdout:
    # Fetch the plot
    cluster_plot = config.CLUSTER_RESULTS / "figures" / "timeseries_feedback_comparison.png"
    local_plot = config.LOCAL_RESULTS / "figures" / "timeseries_feedback_comparison.png"
    cmd = config.scp_from_cluster(cluster_plot, local_plot)
    subprocess.run(cmd, shell=True, check=True)
    print(f"✓ Downloaded to: {local_plot}")
    print("\nNow manually insert this image into Slide 6 of the PowerPoint!")