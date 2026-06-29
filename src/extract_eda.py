"""
extract_eda.py — full per-subject EDA extraction for the PhysioNet BCI2000 MI dataset.

Run locally in the fl_eeg env. Produces two SMALL files you upload back to Claude:
  - per_subject_summary.csv   (one row per subject, human-readable diagnostics)
  - features.npz              (raw 128-dim log-variance features + labels + subject ids)
  - meta.json                 (bad-record log, config, channel info)

Run:
  C:\\Users\\minha\\.conda\\envs\\fl_eeg\\python.exe D:\\FL-new\\code\\extract_eda.py

Expected runtime: a few minutes (loading + filtering 109 subjects x 6 runs).
"""

import os
import glob
import json
import warnings
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import mne
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# ----------------------------- CONFIG -----------------------------
DATASET_DIR = os.path.join(_BASE, "data", "dataset")  # raw PhysioNet EDFs (see data/README.md)
OUT_DIR     = os.path.join(_BASE, "results", "eda")   # features.npz, summary, meta.json
RUNS        = ["R03", "R04", "R07", "R08", "R11", "R12"]  # motor imagery runs
MU_BAND     = (8.0, 13.0)
BETA_BAND   = (13.0, 30.0)
SFREQ_EXP   = 160                      # expected sampling rate
N_CH_EXP    = 64                       # expected channel count
TMIN, TMAX  = 0.0, 4.0                 # epoch window (s) relative to event onset
MOTOR_CH    = ["C3", "CZ", "C4"]       # channels used for ERD diagnostics
SUBJECTS    = [f"S{idx:03d}" for idx in range(1, 110)]   # S001..S109
# ------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


def clean_name(ch):
    return ch.replace(".", "").strip().upper()


def find_run_file(subject, run):
    pats = [
        os.path.join(DATASET_DIR, subject, f"{subject}{run}.edf"),
        os.path.join(DATASET_DIR, f"{subject}{run}.edf"),
        os.path.join(DATASET_DIR, "**", f"{subject}{run}.edf"),
    ]
    for p in pats:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None


def logvar_features(raw, events, event_id, band):
    """Bandpass -> epoch -> log-variance per channel. Returns (n_epochs, n_ch)."""
    rb = raw.copy().filter(band[0], band[1], verbose="ERROR")
    ep = mne.Epochs(rb, events, event_id=event_id, tmin=TMIN, tmax=TMAX,
                    baseline=None, preload=True, verbose="ERROR")
    data = ep.get_data()                 # (n_ep, n_ch, n_times)
    var = data.var(axis=2)               # (n_ep, n_ch)
    var = np.maximum(var, 1e-12)
    return np.log(var), ep.events[:, 2]  # log-var, raw event codes


all_X, all_y, all_subj = [], [], []
summary_rows = []
bad_records = []
chan_names_ref = None

for subj in SUBJECTS:
    run_files = [(r, find_run_file(subj, r)) for r in RUNS]
    found = [(r, f) for r, f in run_files if f]
    if not found:
        bad_records.append({"subject": subj, "reason": "no run files found"})
        continue

    raws = []
    for r, f in found:
        try:
            raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
            if abs(raw.info["sfreq"] - SFREQ_EXP) > 1:
                bad_records.append({"subject": subj, "run": r,
                                    "reason": f"sfreq={raw.info['sfreq']}"})
                continue
            if len(raw.ch_names) != N_CH_EXP:
                bad_records.append({"subject": subj, "run": r,
                                    "reason": f"n_ch={len(raw.ch_names)}"})
            raws.append(raw)
        except Exception as e:
            bad_records.append({"subject": subj, "run": r, "reason": str(e)[:120]})

    if not raws:
        continue

    raw = mne.concatenate_raws(raws, verbose="ERROR")
    if chan_names_ref is None:
        chan_names_ref = list(raw.ch_names)

    # Events from annotations: T0=rest, T1/T2=movement
    try:
        events, ev_id = mne.events_from_annotations(raw, verbose="ERROR")
    except Exception as e:
        bad_records.append({"subject": subj, "reason": f"events: {str(e)[:80]}"})
        continue

    name_by_code = {v: k for k, v in ev_id.items()}
    rest_codes = [c for c, n in name_by_code.items() if n.upper().endswith("T0")]
    move_codes = [c for c, n in name_by_code.items()
                  if n.upper().endswith("T1") or n.upper().endswith("T2")]
    if not rest_codes or not move_codes:
        bad_records.append({"subject": subj, "reason": f"bad annot {ev_id}"})
        continue

    mu_feat, codes = logvar_features(raw, events, ev_id, MU_BAND)
    beta_feat, _ = logvar_features(raw, events, ev_id, BETA_BAND)
    X = np.concatenate([mu_feat, beta_feat], axis=1)        # (n_ep, 128)
    y = np.array([0 if c in rest_codes else 1 for c in codes])

    if len(np.unique(y)) < 2 or len(y) < 12:
        bad_records.append({"subject": subj, "reason": f"too few epochs n={len(y)}"})
        continue

    # Within-subject linear-SVM separability (5-fold, z-scored inside CV)
    try:
        clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000, dual="auto"))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        svm_acc = float(cross_val_score(clf, X, y, cv=skf).mean())
    except Exception:
        svm_acc = float("nan")

    # ERD at motor channels (raw mu/beta power, move vs rest)
    cleaned = [clean_name(c) for c in raw.ch_names]
    motor_idx = [cleaned.index(m) for m in MOTOR_CH if m in cleaned]
    erd_mu = erd_beta = float("nan")
    if motor_idx:
        rest_m = y == 0
        move_m = y == 1
        mu_pow = np.exp(mu_feat[:, motor_idx]).mean(axis=1)    # var ~ power
        beta_pow = np.exp(beta_feat[:, motor_idx]).mean(axis=1)
        if rest_m.any() and move_m.any():
            erd_mu = float((mu_pow[move_m].mean() - mu_pow[rest_m].mean())
                           / (mu_pow[rest_m].mean() + 1e-12))
            erd_beta = float((beta_pow[move_m].mean() - beta_pow[rest_m].mean())
                             / (beta_pow[rest_m].mean() + 1e-12))

    # Artifact / signal-quality proxies
    raw_data = raw.get_data()
    max_amp = float(np.abs(raw_data).max())
    n_flat = int((raw_data.std(axis=1) < 1e-9).sum())

    summary_rows.append({
        "subject": subj,
        "n_epochs": int(len(y)),
        "n_rest": int((y == 0).sum()),
        "n_move": int((y == 1).sum()),
        "class_balance": float((y == 1).mean()),
        "svm_acc": round(svm_acc, 4),
        "erd_mu": round(erd_mu, 4),
        "erd_beta": round(erd_beta, 4),
        "mean_mu_logvar": round(float(mu_feat.mean()), 4),
        "mean_beta_logvar": round(float(beta_feat.mean()), 4),
        "feat_std": round(float(X.std()), 4),
        "max_amp_uV": round(max_amp * 1e6, 1),
        "n_flat_ch": n_flat,
        "n_runs_used": len(raws),
    })

    all_X.append(X.astype(np.float32))
    all_y.append(y.astype(np.int8))
    all_subj.append(np.array([subj] * len(y)))
    print(f"{subj}: epochs={len(y):3d}  svm={svm_acc:.3f}  erd_mu={erd_mu:+.3f}")

# ----------------------------- SAVE -----------------------------
import csv
csv_path = os.path.join(OUT_DIR, "per_subject_summary.csv")
if summary_rows:
    keys = list(summary_rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(summary_rows)

if all_X:
    np.savez_compressed(
        os.path.join(OUT_DIR, "features.npz"),
        X=np.concatenate(all_X, axis=0),
        y=np.concatenate(all_y, axis=0),
        subject=np.concatenate(all_subj, axis=0),
    )

with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
    json.dump({
        "n_subjects_ok": len(summary_rows),
        "n_bad_records": len(bad_records),
        "bad_records": bad_records,
        "runs": RUNS, "mu_band": MU_BAND, "beta_band": BETA_BAND,
        "tmin": TMIN, "tmax": TMAX,
        "channels": chan_names_ref,
    }, fh, indent=2)

print(f"\nDone. {len(summary_rows)} subjects OK, {len(bad_records)} bad records.")
print(f"Wrote: {OUT_DIR}\\per_subject_summary.csv, features.npz, meta.json")
