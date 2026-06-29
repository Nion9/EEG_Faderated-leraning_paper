# Dataset

This project uses the **PhysioNet EEG Motor Movement/Imagery Dataset** (`eegmmidb`),
recorded with the BCI2000 system: 109 subjects, 64 EEG channels, 160 Hz, EDF+ format.

The raw data is **not committed** to this repository (it is ~1.8 GB and is freely
redistributable from PhysioNet directly).

## Download

Get the dataset from PhysioNet:

> https://physionet.org/content/eegmmidb/1.0.0/

Place it so the layout is:

```
data/
└── dataset/
    ├── S001/
    │   ├── S001R01.edf
    │   ├── S001R03.edf
    │   └── ...
    ├── S002/
    └── ... S109/
```

`src/extract_eda.py` reads from `data/dataset/`.

## Runs and task used

This project uses the left/right-fist runs in both executed and imagined conditions:
**R03, R04, R07, R08, R11, R12**. The classification target is binary **Rest vs Movement**
(`T0` vs `T1` ∪ `T2`).

## Excluded subjects

S088, S092, and S100 are excluded: all of their runs were recorded at 128 Hz rather than
the standard 160 Hz. This leaves 106 usable subjects.
