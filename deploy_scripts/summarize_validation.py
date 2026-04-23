"""
summarize_validation.py

Multi-band validation summary with network-specific analysis.

Usage:
    python deploy_scripts/summarize_validation.py --subject sub-dmnelf001

Author: Clemens Bauer
Date: April 2026
"""

import sys
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, default="sub-dmnelf001")
args = parser.parse_args()

SUBJECT = args.subject

print("=" * 70)
print(f"Validation Summary: {SUBJECT}")
print("=" * 70)

conditions = ["feedback", "shortrest", "rest"]
results = {}

for cond in conditions:
    val_file = config.LOCAL_RESULTS / "validation" / f"{SUBJECT}_{cond}_validation.pkl"
    
    if val_file.exists():
        with open(val_file, 'rb') as f:
            results[cond] = pickle.load(f)
    else:
        print(f"\n⚠ {cond}: Not found")

if not results:
    print("\nNo validation results found!")
    sys.exit(1)

# Multi-band table
print("\n" + "=" * 70)
print("Multi-Band PLV Analysis")
print("=" * 70)
print(f"{'Condition':<12} {'Sim R':<10} {'Alpha':<10} {'Beta_Low':<10} {'Beta_High':<10} {'Primary'}")
print("-" * 70)

for cond in conditions:
    if cond in results:
        r = results[cond]
        sim = r.get('sim_sync_mean', 0)
        alpha = r.get('plv_alpha', 0)
        beta_low = r.get('plv_beta_low', 0)
        beta_high = r.get('plv_beta_high', 0)
        primary_band = r.get('primary_band', 'N/A')
        print(f"{cond:<12} {sim:<10.3f} {alpha:<10.3f} {beta_low:<10.3f} {beta_high:<10.3f} {primary_band}")

# Band-specific ordering tests
print("\n" + "=" * 70)
print("Ordering Tests (DMNELF-specific)")
print("=" * 70)

if len(results) == 3:
    # Simulated ordering
    sim_order = sorted(results.keys(), key=lambda c: results[c].get('sim_sync_mean', 0))
    print(f"Simulated R(t): {' < '.join([c.upper() for c in sim_order])}")
    
    # Real alpha ordering
    alpha_order = sorted(results.keys(), key=lambda c: results[c].get('plv_alpha', 0))
    print(f"Real Alpha PLV: {' < '.join([c.upper() for c in alpha_order])}")
    
    # Real beta_low ordering
    beta_low_order = sorted(results.keys(), key=lambda c: results[c].get('plv_beta_low', 0))
    print(f"Real Beta-Low:  {' < '.join([c.upper() for c in beta_low_order])}")
    
    # Real beta_high ordering
    beta_high_order = sorted(results.keys(), key=lambda c: results[c].get('plv_beta_high', 0))
    print(f"Real Beta-High: {' < '.join([c.upper() for c in beta_high_order])}")
    
    # Expected DMNELF ordering: feedback < rest < shortrest
    expected = ["feedback", "rest", "shortrest"]
    
    print("\n" + "=" * 70)
    print("Which band matches neurofeedback effect?")
    print("=" * 70)
    
    for band_name, order in [('Alpha', alpha_order), ('Beta-Low', beta_low_order), ('Beta-High', beta_high_order)]:
        match = "✓" if order == expected else "✗"
        print(f"{band_name:12s}: {match} {' < '.join(order)}")

# BOLD correlations
print("\n" + "=" * 70)
print("BOLD-PDA Correlations")
print("=" * 70)

for cond in conditions:
    if cond in results:
        r = results[cond]
        difumo_corr = r.get('bold_pda_corr_difumo')
        personal_corr = r.get('bold_pda_corr_personal')
        
        print(f"\n{cond.upper()}:")
        if difumo_corr is not None:
            print(f"  DiFuMo:   r={difumo_corr:.3f}")
        if personal_corr is not None:
            print(f"  Personal: r={personal_corr:.3f}")
