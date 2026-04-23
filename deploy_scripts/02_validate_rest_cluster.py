#!/usr/bin/env python
import sys
from pathlib import Path
import numpy as np
import mne
import pickle
from scipy.signal import hilbert, butter, sosfiltfilt

SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from models.kuramoto import KuramotoModel, KuramotoParams
from models.energy import EnergyFunction

SUBJECT = "sub-dmnelf001"
CONDITION = "rest"
PRIMARY_BAND = "alpha"

print("=" * 70)
print("Validation: " + SUBJECT + " - " + CONDITION)
print("=" * 70)

# ========== Load EEG ==========
print("\n1. Loading EEG...")
eeg_base = Path("/projects/swglab/data/DMNELF/analysis/MNE/bids/derivatives/preprocessed") / SUBJECT / "ses-dmnelf" / "eeg"
eeg_files = sorted(eeg_base.glob("*task-" + CONDITION + "*_eeg.fif"))

if not eeg_files:
    print("ERROR: No files found")
    sys.exit(1)

# Load all runs for variance analysis
all_data = []
for f in eeg_files:
    raw = mne.io.read_raw_fif(str(f), preload=True, verbose=False)
    all_data.append(raw.get_data())

combined = np.concatenate(all_data, axis=1)
channel_names = raw.ch_names

# Auto-detect bad channels
variances = np.var(combined, axis=1)
median_var = np.median(variances)
bad_channels = []

for ch, var in zip(channel_names, variances):
    if ch == "ECG" or var > 3.0 * median_var:
        bad_channels.append(ch)

print(f"  Detected {len(bad_channels)} bad channels: {bad_channels}")

# Reload and interpolate
raw = mne.io.read_raw_fif(str(eeg_files[0]), preload=True, verbose=False)
raw.info["bads"] = bad_channels
raw.interpolate_bads(reset_bads=True)

# Get clean EEG (exclude ECG)
picks = mne.pick_types(raw.info, eeg=True, exclude=["ECG"])
eeg_clean = raw.get_data(picks=picks)[:, :15000]
print(f"  Clean data: {eeg_clean.shape}")

# ========== Simulate Kuramoto ==========
print("\n2. Simulating Kuramoto (K=1.0)...")
n_ch = eeg_clean.shape[0]
params = KuramotoParams(n_oscillators=n_ch, coupling_strength=1.0, dt=0.01)
model = KuramotoModel(params)

if CONDITION == "feedback":
    input_func = lambda t: np.ones(n_ch) * 0.0
elif CONDITION == "shortrest":
    def input_func(t):
        inp = np.zeros(n_ch)
        inp[:n_ch//2] = 0.0
        inp[n_ch//2:] = -0.0
        return inp
else:
    input_func = None

sync_sim, _ = model.simulate(60.0, input_func)
print(f"  Sim R(t): mean={np.mean(sync_sim):.3f}")

# ========== Multi-band PLV ==========
print("\n3. Computing multi-band PLV...")

bands = {"alpha": (8, 12), "beta_low": (13, 20), "beta_high": (20, 30)}
plv_results = {}

for band_name, (fmin, fmax) in bands.items():
    sos = butter(4, [fmin, fmax], btype="bandpass", fs=250, output="sos")
    filt = sosfiltfilt(sos, eeg_clean, axis=1)
    analytic = hilbert(filt, axis=1)
    phases = np.angle(analytic)
    
    plv_vals = []
    for i in range(n_ch):
        for j in range(i+1, n_ch):
            phase_diff = phases[i] - phases[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_vals.append(plv)
    
    mean_plv = np.mean(plv_vals)
    plv_results[band_name] = mean_plv
    marker = " ← PRIMARY" if band_name == PRIMARY_BAND.replace("_full", "") or (PRIMARY_BAND == "beta" and "beta" in band_name) else ""
    print(f"  {band_name:10s}: PLV={mean_plv:.3f}{marker}")

# ========== Load fMRI Networks ==========
print("\n4. Loading fMRI networks...")

# DiFuMo group
difumo_file = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/difumo_timeseries") / (SUBJECT + "_ses-dmnelf_task-" + CONDITION + "_run-01_desc-difumo64_timeseries.tsv")
print(f"  Looking for: {difumo_file}")
if difumo_file.exists():
    import pandas as pd
    df = pd.read_csv(difumo_file, sep="\t")
    parcels = df.values.T  # Transpose to [64 parcels, n_volumes]
    print(f"  Loaded DiFuMo: {parcels.shape}")
    
    dmn_idx = [2, 3, 9, 13, 20, 25, 28, 32, 38, 41, 52, 55]
    cen_idx = [0, 5, 11, 14, 16, 22, 29, 35, 39, 44, 47, 50, 57, 62]
    dmn_difumo = np.mean(parcels[dmn_idx, :], axis=0)
    cen_difumo = np.mean(parcels[cen_idx, :], axis=0)
    pda_difumo = cen_difumo - dmn_difumo
    print(f"  DiFuMo PDA: mean={np.mean(pda_difumo):.3f}, std={np.std(pda_difumo):.3f}")
    has_difumo = True
else:
    print("  DiFuMo: not found")
    has_difumo = False

# Personal parcels
personal_dir = Path("/projects/swglab/data/DMNELF/derivatives/difumo_personal")
dmn_personal_file = personal_dir / (SUBJECT + "_personal_dmn.npy")
cen_personal_file = personal_dir / (SUBJECT + "_personal_cen.npy")

if dmn_personal_file.exists() and cen_personal_file.exists():
    dmn_personal_idx = np.load(dmn_personal_file)
    cen_personal_idx = np.load(cen_personal_file)
    if has_difumo:
        dmn_personal = np.mean(parcels[dmn_personal_idx, :], axis=0)
        cen_personal = np.mean(parcels[cen_personal_idx, :], axis=0)
        pda_personal = cen_personal - dmn_personal
        print(f"  Personal PDA: mean={np.mean(pda_personal):.3f}")
        has_personal = True
    else:
        has_personal = False
else:
    print("  Personal: not found")
    has_personal = False

# ========== Simulate BOLD from sync ==========
if has_difumo:
    print("\n5. Simulating BOLD from R(t)...")
    energy_fn = EnergyFunction()
    bold_sim = energy_fn.simulate_bold_from_sync(sync_sim, dt=0.01)
    
    from scipy.stats import pearsonr
    
    # Resample to TR=2s
    bold_resampled = np.interp(
        np.linspace(0, len(bold_sim)-1, len(pda_difumo)),
        np.arange(len(bold_sim)),
        bold_sim
    )
    
    # Correlations
    corr_difumo, _ = pearsonr(pda_difumo, bold_resampled)
    print(f"  DiFuMo PDA correlation: r={corr_difumo:.3f}")
    
    if has_personal:
        corr_personal, _ = pearsonr(pda_personal, bold_resampled)
        print(f"  Personal PDA correlation: r={corr_personal:.3f}")
else:
    corr_difumo = None
    corr_personal = None

# ========== Summary ==========
primary_plv = plv_results.get(PRIMARY_BAND.replace("_full", "_low") if "beta" in PRIMARY_BAND else PRIMARY_BAND, 0)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Simulated R(t): {np.mean(sync_sim):.3f} (target: 0.39)")
print(f"Real PLV ({PRIMARY_BAND}): {primary_plv:.3f}")
print(f"Difference: {abs(np.mean(sync_sim) - primary_plv):.3f}")
print("\nAll bands:")
for band, plv in plv_results.items():
    print(f"  {band:10s}: {plv:.3f}")
if has_difumo:
    print(f"\nBOLD-PDA correlation:")
    print(f"  DiFuMo: r={corr_difumo:.3f}")
    if has_personal:
        print(f"  Personal: r={corr_personal:.3f}")

# Save
output_dir = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/results") / "validation"
output_dir.mkdir(exist_ok=True, parents=True)
output_file = output_dir / (SUBJECT + "_" + CONDITION + "_validation.pkl")

with open(output_file, "wb") as f:
    pickle.dump({
        "subject": SUBJECT,
        "condition": CONDITION,
        "bad_channels": bad_channels,
        "n_clean": n_ch,
        "sim_sync_mean": float(np.mean(sync_sim)),
        "plv_alpha": float(plv_results["alpha"]),
        "plv_beta_low": float(plv_results["beta_low"]),
        "plv_beta_high": float(plv_results["beta_high"]),
        "primary_band": PRIMARY_BAND,
        "primary_plv": float(primary_plv),
        "bold_pda_corr_difumo": float(corr_difumo) if corr_difumo else None,
        "bold_pda_corr_personal": float(corr_personal) if has_personal else None
    }, f)

print("\nSaved: " + str(output_file))