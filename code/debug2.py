"""Debug script v2 - prints every step with full errors."""
import os, numpy as np, warnings
warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

DATASET_PATH = r"D:\FL\dataset"
sid  = "S001"
sdir = os.path.join(DATASET_PATH, sid)
edf  = os.path.join(sdir, "S001R03.edf")

print(f"File exists: {os.path.exists(edf)}")

print("\nStep 1: Reading EDF...")
raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
print(f"  nchan={raw.info['nchan']}  sfreq={raw.info['sfreq']}")

print("\nStep 2: Pick EEG...")
try:
    raw.pick_types(eeg=True, verbose=False)
    print(f"  nchan after pick_types={raw.info['nchan']}")
except Exception as e:
    print(f"  pick_types FAILED: {e}")
    print("  Trying pick(['eeg'])...")
    try:
        raw.pick(['eeg'], verbose=False)
        print(f"  nchan after pick=['eeg']={raw.info['nchan']}")
    except Exception as e2:
        print(f"  Also failed: {e2}")
        print(f"  Channel types: {set(raw.get_channel_types())}")

print(f"\nStep 3: Filter 1-40 Hz IIR...")
try:
    raw.filter(1.0, 40.0, method='iir', verbose=False)
    print("  Filter OK")
except Exception as e:
    print(f"  Filter FAILED: {e}")

print("\nStep 4: Average reference...")
try:
    raw.set_eeg_reference('average', projection=False, verbose=False)
    print("  Reference OK")
except Exception as e:
    print(f"  Reference FAILED: {e}")

print("\nStep 5: Events from annotations...")
try:
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    print(f"  event_id={event_id}")
    print(f"  events shape={events.shape}")
except Exception as e:
    print(f"  Events FAILED: {e}")
    import sys; sys.exit(1)

print("\nStep 6: Build label map...")
label_map = {}
for key, val in event_id.items():
    ks = str(key)
    if 'T1' in ks: label_map[val] = 0
    elif 'T2' in ks: label_map[val] = 1
print(f"  label_map={label_map}")

mask = np.isin(events[:,2], list(label_map.keys()))
ev   = events[mask]
print(f"  filtered events: {len(ev)}")

print("\nStep 7: Create Epochs (baseline=None)...")
try:
    epochs = mne.Epochs(raw, ev,
                        event_id={str(k):k for k in label_map.keys()},
                        tmin=0.0, tmax=3.9375,
                        baseline=None,
                        preload=True, verbose=False)
    data   = epochs.get_data()
    labels = np.array([label_map[e] for e in epochs.events[:,2]])
    print(f"  SUCCESS: data={data.shape}  labels={labels.shape}")
    print(f"  min={data.min():.4f}  max={data.max():.4f}")
    print(f"  L={int((labels==0).sum())}  R={int((labels==1).sum())}")
except Exception as e:
    print(f"  Epochs FAILED: {e}")
    import traceback; traceback.print_exc()
