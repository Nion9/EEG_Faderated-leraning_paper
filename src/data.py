"""
PhysioNet EEG Motor Imagery Loader — paper exact spec.
=======================================================
Paper: "1-40 Hz bandpass IIR, average re-referencing, z-score normalisation"
Dataset: D:/FL/dataset/S001/S001R03.edf
Runs: R03,R04,R07,R08,R11,R12
Labels: T1=left fist (0), T2=right fist (1)
Epoch: 0-4s post-stimulus, 640 samples at 160 Hz
"""

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    raise ImportError("\nInstall MNE: pip install mne\n")

DATASET_PATH = r"D:\FL\dataset"
SFREQ        = 160
N_CHANNELS   = 64
N_SAMPLES    = 640
TMIN         = 0.0
TMAX         = 3.9375
MOTOR_RUNS   = [3, 4, 7, 8, 11, 12]


def load_subject(subject_id, dataset_path=DATASET_PATH):
    sid  = f"S{subject_id:03d}"
    sdir = os.path.join(dataset_path, sid)
    X_all, y_all = [], []

    for run in MOTOR_RUNS:
        edf_path = os.path.join(sdir, f"{sid}R{run:02d}.edf")
        if not os.path.exists(edf_path):
            continue
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception:
            continue

        try:
            raw.pick_types(eeg=True, verbose=False)
        except Exception:
            pass
        if raw.info['nchan'] != N_CHANNELS:
            continue

        if abs(raw.info['sfreq'] - SFREQ) > 1:
            raw.resample(SFREQ, verbose=False)

        # Paper: 1-40 Hz IIR bandpass
        raw.filter(1.0, 40.0, method='iir', verbose=False)

        # Paper: average re-reference
        raw.set_eeg_reference('average', projection=False, verbose=False)

        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
        except Exception:
            continue

        # str() handles np.str_ type from MNE
        label_map = {}
        for key, val in event_id.items():
            ks = str(key)
            if 'T1' in ks:
                label_map[val] = 0
            elif 'T2' in ks:
                label_map[val] = 1

        if not label_map:
            continue

        mask       = np.isin(events[:, 2], list(label_map.keys()))
        events_use = events[mask]
        if len(events_use) == 0:
            continue

        try:
            epochs = mne.Epochs(
                raw, events_use,
                event_id = {str(k): k for k in label_map.keys()},
                tmin     = TMIN,
                tmax     = TMAX,
                baseline = None,   # MUST be None when tmin=0
                preload  = True,
                verbose  = False,
            )
        except Exception:
            continue

        data   = epochs.get_data()
        labels = np.array([label_map[ev] for ev in epochs.events[:, 2]])

        if len(labels) == 0:
            continue

        # Fix length — trim or pad to exactly 640 samples
        T = data.shape[2]
        if T > N_SAMPLES:
            data = data[:, :, :N_SAMPLES]
        elif T < N_SAMPLES:
            pad  = np.zeros((data.shape[0], N_CHANNELS, N_SAMPLES - T),
                            dtype=data.dtype)
            data = np.concatenate([data, pad], axis=2)
        # Now data.shape[2] == 640

        # Paper: z-score per channel per epoch
        mu        = data.mean(axis=2, keepdims=True)
        std       = data.std(axis=2,  keepdims=True) + 1e-8
        data_norm = ((data - mu) / std).astype(np.float32)

        X_all.append(data_norm[:, np.newaxis, :, :])
        y_all.append(labels)

    if not X_all:
        return None, None

    X    = np.concatenate(X_all, axis=0)
    y    = np.concatenate(y_all, axis=0).astype(np.int64)
    rng  = np.random.RandomState(subject_id * 31 + 17)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def load_all_subjects(n_subjects=40, subject_ids=None,
                      dataset_path=DATASET_PATH,
                      seed=42, verbose=True,
                      n_epochs_per_class=None):
    rng = np.random.RandomState(seed)

    if subject_ids is None:
        subject_ids = list(range(1, n_subjects + 1))

    clients = []
    skipped = []

    for sid in subject_ids:
        if verbose:
            print(f"  S{sid:03d} ...", end=' ', flush=True)

        X, y = load_subject(sid, dataset_path=dataset_path)

        if X is None or len(y) < 10:
            n = len(y) if y is not None else 0
            if verbose:
                print(f"SKIP ({n} epochs)")
            skipped.append(sid)
            continue

        n    = len(y)
        idx  = rng.permutation(n)
        n_tr = int(0.70 * n)
        n_v  = int(0.10 * n)
        n_te = n - n_tr - n_v
        n0   = int((y == 0).sum())
        n1   = int((y == 1).sum())

        clients.append({
            'id'   : sid,
            'train': (X[idx[:n_tr]],           y[idx[:n_tr]]),
            'val'  : (X[idx[n_tr:n_tr+n_v]],   y[idx[n_tr:n_tr+n_v]]),
            'test' : (X[idx[n_tr+n_v:]],        y[idx[n_tr+n_v:]]),
            'snr'  : 1.0,
        })

        if verbose:
            print(f"OK | {n} epochs | "
                  f"train={n_tr} val={n_v} test={n_te} | "
                  f"L={n0} R={n1}")

    if not clients:
        raise RuntimeError(
            f"\nNo subjects loaded from: {dataset_path}\n"
            f"Expected: {dataset_path}\\S001\\S001R03.edf\n"
        )

    if skipped and verbose:
        print(f"\n  Skipped {len(skipped)}: {skipped}")
    if verbose:
        print(f"\n  Total: {len(clients)} FL clients loaded")
    return clients


def build_shared_val(clients, n_subjects=5, seed=42):
    rng  = np.random.RandomState(seed)
    Xs   = [c['val'][0] for c in clients[:n_subjects]]
    ys   = [c['val'][1] for c in clients[:n_subjects]]
    Xv   = np.concatenate(Xs)
    yv   = np.concatenate(ys)
    perm = rng.permutation(len(yv))
    return Xv[perm], yv[perm]


if __name__ == '__main__':
    print("Testing PhysioNet loader (paper spec)...")
    print(f"Dataset: {DATASET_PATH}\n")
    clients = load_all_subjects(n_subjects=5, verbose=True)
    print(f"\nLoaded {len(clients)} subjects OK")
    for c in clients:
        X, y = c['train']
        bc   = np.bincount(y, minlength=2)
        print(f"  S{c['id']:03d}: X={X.shape}  L={bc[0]} R={bc[1]}")
