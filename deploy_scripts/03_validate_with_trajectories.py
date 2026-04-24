"""
03_validate_with_trajectories.py

Full validation with:
- Complete time series saved (for visualization)
- Personal network masks + DiFuMo group
- Multi-band PLV

Usage:
    python deploy_scripts/03_validate_with_trajectories.py --subject sub-dmnelf001 --condition feedback

Author: Clemens Bauer
Date: April 2026
"""

import sys
from pathlib import Path
import subprocess
import py_compile
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, default="sub-dmnelf001")
parser.add_argument("--condition", type=str, default="feedback")
args = parser.parse_args()

SUBJECT = args.subject
CONDITION = args.condition
cond_params = config.get_condition_params(CONDITION)

k_coupling = str(cond_params['kuramoto']['coupling_strength'])
input_mag = str(cond_params['input_magnitude'])

CLUSTER_SCRIPT = [
    '#!/usr/bin/env python',
    'import sys',
    'from pathlib import Path',
    'import numpy as np',
    'import pandas as pd',
    'import mne',
    'import pickle',
    'from scipy.stats import pearsonr',
    'from scipy.signal import hilbert, butter, sosfiltfilt',
    '',
    'SCRIPT_DIR = Path(__file__).parent',
    'SRC_DIR = SCRIPT_DIR.parent / "src"',
    'sys.path.insert(0, str(SRC_DIR))',
    '',
    'from models.kuramoto import KuramotoModel, KuramotoParams',
    'from models.energy import EnergyFunction',
    '',
    'SUBJECT = "' + SUBJECT + '"',
    'CONDITION = "' + CONDITION + '"',
    '',
    'print("=" * 70)',
    'print("Full Validation: " + SUBJECT + " - " + CONDITION)',
    'print("=" * 70)',
    '',
    '# Load EEG',
    'print("\\n1. Loading EEG...")',
    'eeg_file = Path("' + str(config.CLUSTER_PREPROCESSED) + '") / SUBJECT / "ses-dmnelf" / "eeg" / (SUBJECT + "_ses-dmnelf_task-' + CONDITION + '_run-01_desc-preproc_eeg.fif")',
    'raw = mne.io.read_raw_fif(str(eeg_file), preload=True, verbose=False)',
    '',
    '# Auto-detect bad channels',
    'data = raw.get_data()',
    'variances = np.var(data, axis=1)',
    'bad_channels = [raw.ch_names[i] for i, v in enumerate(variances) if v > 3*np.median(variances) or raw.ch_names[i] == "ECG"]',
    'raw.info["bads"] = bad_channels',
    'raw.interpolate_bads(reset_bads=True)',
    '',
    'picks = mne.pick_types(raw.info, eeg=True, exclude=["ECG"])',
    'eeg_clean = raw.get_data(picks=picks)[:, :15000]',
    'print(f"  Clean: {eeg_clean.shape}, bad channels: {len(bad_channels)}")',
    '',
    '# Simulate Kuramoto',
    'print("\\n2. Simulating Kuramoto...")',
    'n_ch = eeg_clean.shape[0]',
    'params = KuramotoParams(n_oscillators=n_ch, coupling_strength=' + k_coupling + ', dt=0.01)',
    'model = KuramotoModel(params)',
    '',
    'if CONDITION == "feedback":',
    '    input_func = lambda t: np.ones(n_ch) * ' + input_mag,
    'elif CONDITION == "shortrest":',
    '    def input_func(t):',
    '        inp = np.zeros(n_ch)',
    '        inp[:n_ch//2] = ' + input_mag,
    '        inp[n_ch//2:] = -' + input_mag,
    '        return inp',
    'else:',
    '    input_func = None',
    '',
    'sync_sim, phases_sim = model.simulate(60.0, input_func)',
    'print(f"  Simulated R(t): mean={np.mean(sync_sim):.3f}, len={len(sync_sim)}")',
    '',
    '# Multi-band PLV',
    'print("\\n3. Computing multi-band PLV...")',
    'bands = {"alpha": (8, 12), "beta_low": (13, 20), "beta_high": (20, 30)}',
    'plv_results = {}',
    '',
    'for band_name, (fmin, fmax) in bands.items():',
    '    sos = butter(4, [fmin, fmax], btype="bandpass", fs=250, output="sos")',
    '    filt = sosfiltfilt(sos, eeg_clean, axis=1)',
    '    analytic = hilbert(filt, axis=1)',
    '    phases = np.angle(analytic)',
    '    plv_vals = []',
    '    for i in range(n_ch):',
    '        for j in range(i+1, n_ch):',
    '            phase_diff = phases[i] - phases[j]',
    '            plv = np.abs(np.mean(np.exp(1j * phase_diff)))',
    '            plv_vals.append(plv)',
    '    plv_results[band_name] = np.mean(plv_vals)',
    '    print(f"  {band_name}: {plv_results[band_name]:.3f}")',
    '',
    '# Load fMRI - DiFuMo group',
    'print("\\n4. Loading fMRI networks...")',
    'difumo_file = Path("' + str(config.CLUSTER_DIFUMO) + '") / (SUBJECT + "_ses-dmnelf_task-' + CONDITION + '_run-01_desc-difumo64_timeseries.tsv")',
    '',
    'if difumo_file.exists():',
    '    df = pd.read_csv(difumo_file, sep="\\t")',
    '    parcels_difumo = df.values.T',
    '    dmn_idx = ' + str(config.DIFUMO_DMN_INDICES),
    '    cen_idx = ' + str(config.DIFUMO_CEN_INDICES),
    '    dmn_difumo = np.mean(parcels_difumo[dmn_idx, :], axis=0)',
    '    cen_difumo = np.mean(parcels_difumo[cen_idx, :], axis=0)',
    '    pda_difumo = cen_difumo - dmn_difumo',
    '    print(f"  DiFuMo: PDA shape={pda_difumo.shape}, mean={np.mean(pda_difumo):.3f}")',
    'else:',
    '    print("  DiFuMo: not found")',
    '    pda_difumo = None',
    '',
    '# Load personal masks',
    'personal_dir = Path("' + str(config.CLUSTER_DIFUMO) + '")',
    'personal_file = personal_dir / (SUBJECT + "_ses-dmnelf_task-' + CONDITION + '_run-01_desc-personalized_timeseries.tsv")',
    '',
    'if personal_file.exists():',
    '    df_pers = pd.read_csv(personal_file, sep="\\t")',
    '    parcels_pers = df_pers.values.T',
    '    ',
    '    # Try to find CEN/DMN columns',
    '    cols = df_pers.columns.tolist()',
    '    dmn_cols = [c for c in cols if "dmn" in c.lower() or "default" in c.lower()]',
    '    cen_cols = [c for c in cols if "cen" in c.lower() or "executive" in c.lower() or "fronto" in c.lower()]',
    '    ',
    '    if dmn_cols and cen_cols:',
    '        dmn_personal = df_pers[dmn_cols].values.mean(axis=1)',
    '        cen_personal = df_pers[cen_cols].values.mean(axis=1)',
    '        pda_personal = cen_personal - dmn_personal',
    '        print(f"  Personal: PDA shape={pda_personal.shape}, mean={np.mean(pda_personal):.3f}")',
    '        print(f"    DMN cols: {dmn_cols}")',
    '        print(f"    CEN cols: {cen_cols}")',
    '    else:',
    '        pda_personal = None',
    '        print("  Personal: No CEN/DMN columns found")',
    'else:',
    '    print(f"  Personal: not found at {personal_file}")',
    '    pda_personal = None',
    '',
    '# Simulate BOLD from sync',
    'print("\\n5. Simulating BOLD...")',
    'energy_fn = EnergyFunction()',
    'bold_sim = energy_fn.simulate_bold_from_sync(sync_sim, dt=0.01)',
    'print(f"  Simulated BOLD: len={len(bold_sim)}")',
    '',
    '# Correlations',
    'correlations = {}',
    'trajectories = {"sync_sim": sync_sim.tolist(), "bold_sim": bold_sim.tolist()}',
    '',
    'if pda_difumo is not None:',
    '    bold_resampled = np.interp(',
    '        np.linspace(0, len(bold_sim)-1, len(pda_difumo)),',
    '        np.arange(len(bold_sim)),',
    '        bold_sim',
    '    )',
    '    corr, pval = pearsonr(pda_difumo, bold_resampled)',
    '    print(f"  DiFuMo: r={corr:.3f}, p={pval:.4f}")',
    '    correlations["difumo"] = float(corr)',
    '    trajectories["pda_difumo"] = pda_difumo.tolist()',
    '    trajectories["bold_resampled"] = bold_resampled.tolist()',
    '',
    'if pda_personal is not None:',
    '    bold_resampled_p = np.interp(',
    '        np.linspace(0, len(bold_sim)-1, len(pda_personal)),',
    '        np.arange(len(bold_sim)),',
    '        bold_sim',
    '    )',
    '    corr_p, pval_p = pearsonr(pda_personal, bold_resampled_p)',
    '    print(f"  Personal: r={corr_p:.3f}, p={pval_p:.4f}")',
    '    correlations["personal"] = float(corr_p)',
    '    trajectories["pda_personal"] = pda_personal.tolist()',
    '',
    '# Save with trajectories',
    'output_dir = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/adaptive_sync_dmnelf/results/validation")',
    'output_file = output_dir / (SUBJECT + "_" + CONDITION + "_full_validation.pkl")',
    '',
    'with open(output_file, "wb") as f:',
    '    pickle.dump({',
    '        "subject": SUBJECT,',
    '        "condition": CONDITION,',
    '        "plv_alpha": float(plv_results["alpha"]),',
    '        "plv_beta_low": float(plv_results["beta_low"]),',
    '        "plv_beta_high": float(plv_results["beta_high"]),',
    '        "correlations": correlations,',
    '        "trajectories": trajectories,',
    '        "bad_channels": bad_channels',
    '    }, f)',
    '',
    'print(f"\\nSaved with trajectories: {output_file}")',
]

SLURM_SCRIPT = [
    '#!/bin/bash',
    '#SBATCH --job-name=val_full',
    '#SBATCH --partition=short',
    '#SBATCH --time=00:30:00',
    '#SBATCH --cpus-per-task=2',
    '#SBATCH --mem=8G',
    '#SBATCH --output=' + str(config.CLUSTER_LOGS / f'val_full_{SUBJECT}_{CONDITION}_%j.out'),
    '',
    'source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh',
    'python ' + str(config.CLUSTER_SCRIPTS / f'03_validate_full_{CONDITION}_cluster.py'),
]

def deploy():
    local_file = Path("deploy_scripts") / f"03_validate_full_{CONDITION}_cluster.py"
    with open(local_file, 'w') as f:
        f.write('\n'.join(CLUSTER_SCRIPT))
    
    py_compile.compile(str(local_file), doraise=True)
    print("✓ Compiled")
    
    subprocess.run(config.scp_to_cluster(local_file, config.CLUSTER_SCRIPTS / f"03_validate_full_{CONDITION}_cluster.py"), shell=True, check=True)
    
    slurm_file = Path("deploy_scripts") / f"03_validate_full_{CONDITION}.sh"
    with open(slurm_file, 'w') as f:
        f.write('\n'.join(SLURM_SCRIPT))
    subprocess.run(config.scp_to_cluster(slurm_file, config.CLUSTER_SCRIPTS / f"03_validate_full_{CONDITION}.sh"), shell=True, check=True)
    print("✓ Deployed")
    
    result = subprocess.run(
        config.ssh_command(f"sbatch {config.CLUSTER_SCRIPTS / f'03_validate_full_{CONDITION}.sh'}"),
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"✓ Job: {job_id}")

if __name__ == "__main__":
    print("=" * 70)
    print(f"Deploy Full Validation: {SUBJECT} - {CONDITION}")
    print("=" * 70)
    deploy()
