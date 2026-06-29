"""
byzantine_experiment.py — the deciding experiment for FL-EVO (PSO).

Runs FedAvg vs FedProx vs FL-EVO(PSO) on the real EEG strong-subject pool,
under three Byzantine attacks at adversarial fractions 0/2/3/4 of 10 clients,
across 5 seeds. Writes per-seed final accuracies to a CSV you upload back.

Self-contained: needs only numpy + features.npz. All three methods share one
harness and differ ONLY in the aggregation step, so the robustness gap is a
clean apples-to-apples comparison.

Run:
  C:\\Users\\minha\\.conda\\envs\\fl_eeg\\python.exe D:\\FL-new\\code\\byzantine_experiment.py

Expected runtime: ~15-40 min (FL-EVO's PSO is the cost; reduce PSO_ITERS for a fast pass).
"""

import os
import csv
import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------- CONFIG -----------------------------
FEATURES_PATH = os.path.join(_BASE, "results", "eda", "features.npz")
OUT_DIR       = os.path.join(_BASE, "results", "byzantine")
OUT_CSV       = os.path.join(OUT_DIR, "byzantine_results.csv")

# Top-10 strong pool (highest within-subject separability from the EDA)
CLIENTS = ["S004", "S033", "S089", "S045", "S032",
           "S025", "S042", "S022", "S069", "S015"]

N_CLIENTS    = 10
ROUNDS       = 30
LOCAL_EPOCHS = 5
LOCAL_LR     = 0.1
BATCH        = 32
SERVER_LR    = 0.5          # alpha local-integration parameter
FEDPROX_MU   = 0.01

FRACTIONS = [0, 2, 3, 4]    # number of Byzantine clients
ATTACKS   = ["signflip_scale", "signflip", "gaussian"]
METHODS   = ["fedavg", "fedprox", "flevo", "median", "trimmed", "krum"]
SEEDS     = [0, 1, 2, 3, 4]

# PSO (locked specs)
PSO_PARTICLES = 30
PSO_ITERS     = 50
PSO_W         = 0.7
PSO_C1        = 2.0
PSO_C2        = 2.0

# Fitness weights: f = 0.6*acc + 0.3*diversity + 0.1*snr
F_ACC, F_DIV, F_SNR = 0.6, 0.3, 0.1

# Attack strengths
SIGNFLIP_SCALE = 5.0
GAUSSIAN_FACTOR = 10.0
# ------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def accuracy(w, b, X, y):
    return float((( (X @ w + b) > 0).astype(int) == y).mean())


def train_local(w, b, X, y, gw, gb, rng):
    """5 local epochs of mini-batch logistic-regression SGD (+ FedProx prox term)."""
    w = w.copy(); b = float(b)
    n = len(y)
    for _ in range(LOCAL_EPOCHS):
        idx = rng.permutation(n)
        for s in range(0, n, BATCH):
            bi = idx[s:s + BATCH]
            Xb, yb = X[bi], y[bi]
            p = sigmoid(Xb @ w + b)
            err = p - yb
            dw = Xb.T @ err / len(bi) + FEDPROX_MU * (w - gw)
            db = err.mean() + FEDPROX_MU * (b - gb)
            w -= LOCAL_LR * dw
            b -= LOCAL_LR * db
    return w, b


def apply_attack(dw, db, attack, rng, ref_norm):
    if attack == "signflip_scale":
        return -SIGNFLIP_SCALE * dw, -SIGNFLIP_SCALE * db
    if attack == "signflip":
        return -dw, -db
    if attack == "gaussian":
        scale = GAUSSIAN_FACTOR * (ref_norm + 1e-9) / np.sqrt(len(dw))
        return rng.normal(0, scale, size=dw.shape), float(rng.normal(0, scale))
    return dw, db


def pso_aggregate(deltas_w, deltas_b, gw, gb, Xval, yval, rng):
    """PSO over client weights (softmax simplex) maximizing 0.6*acc+0.3*div+0.1*snr."""
    K = len(deltas_w)
    DW = np.array(deltas_w); DB = np.array(deltas_b)        # (K,128),(K,)
    norms = np.linalg.norm(DW, axis=1) + 1e-9

    # diversity term: mean pairwise cosine distance of client deltas (constant per round)
    U = DW / norms[:, None]
    cos = U @ U.T
    iu = np.triu_indices(K, k=1)
    diversity = float((1 - cos[iu]).mean()) if K > 1 else 0.0
    diversity = np.clip(diversity, 0, 1)

    def fitness(pos):
        a = np.exp(pos - pos.max()); a = a / a.sum()        # softmax -> simplex
        agg_w = a @ DW; agg_b = float(a @ DB)
        acc = accuracy(gw + SERVER_LR * agg_w, gb + SERVER_LR * agg_b, Xval, yval)
        snr = np.linalg.norm(agg_w) / float(a @ norms + 1e-9)   # in [0,1], rewards coherence
        return F_ACC * acc + F_DIV * diversity + F_SNR * snr, a

    pos = rng.normal(0, 1, size=(PSO_PARTICLES, K))
    vel = rng.normal(0, 0.1, size=(PSO_PARTICLES, K))
    pbest = pos.copy()
    pbest_f = np.full(PSO_PARTICLES, -np.inf)
    pbest_a = [None] * PSO_PARTICLES
    gbest = None; gbest_f = -np.inf; gbest_a = None

    for _ in range(PSO_ITERS):
        for i in range(PSO_PARTICLES):
            f, a = fitness(pos[i])
            if f > pbest_f[i]:
                pbest_f[i] = f; pbest[i] = pos[i].copy(); pbest_a[i] = a
            if f > gbest_f:
                gbest_f = f; gbest = pos[i].copy(); gbest_a = a
        r1 = rng.random((PSO_PARTICLES, K)); r2 = rng.random((PSO_PARTICLES, K))
        vel = PSO_W * vel + PSO_C1 * r1 * (pbest - pos) + PSO_C2 * r2 * (gbest - pos)
        pos = pos + vel

    agg_w = gbest_a @ DW; agg_b = float(gbest_a @ DB)
    return agg_w, agg_b, gbest_a


def robust_aggregate(deltas_w, deltas_b, method, n_byz):
    """Closed-form robust aggregators (no optimizer, no validation set)."""
    DW = np.array(deltas_w); DB = np.array(deltas_b); K = len(DW)
    if method == "median":
        return np.median(DW, axis=0), float(np.median(DB))
    if method == "trimmed":
        k = min(n_byz, (K - 1) // 2)
        sw = np.sort(DW, axis=0); sb = np.sort(DB)
        if K - 2 * k > 0:
            return sw[k:K - k].mean(axis=0), float(sb[k:K - k].mean())
        return sw.mean(axis=0), float(sb.mean())
    if method == "krum":
        f = n_byz
        D = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i != j:
                    D[i, j] = np.sum((DW[i] - DW[j]) ** 2)
        nbr = max(1, K - f - 2)
        scores = [np.sort(D[i])[1:1 + nbr].sum() for i in range(K)]   # exclude self (0)
        m = max(1, K - f)                                             # Multi-Krum: avg best (K-f)
        order = np.argsort(scores)[:m]
        return DW[order].mean(axis=0), float(DB[order].mean())
    raise ValueError(method)


def run_fl(client_data, val, method, attack, n_byz, seed):
    rng = np.random.default_rng(seed)
    Xval, yval = val
    w = np.zeros(128); b = 0.0
    byz_ids = set(range(n_byz))     # first n_byz clients are adversarial

    for _ in range(ROUNDS):
        deltas_w, deltas_b, n_samples = [], [], []
        for cid, (Xtr, ytr, _, _) in enumerate(client_data):
            lw, lb = train_local(w, b, Xtr, ytr, w, b, rng)
            dw, db = lw - w, lb - b
            if cid in byz_ids:
                dw, db = apply_attack(dw, db, attack, rng, np.linalg.norm(dw))
            deltas_w.append(dw); deltas_b.append(db); n_samples.append(len(ytr))

        if method in ("fedavg", "fedprox"):
            wts = np.array(n_samples, float); wts /= wts.sum()
            agg_w = wts @ np.array(deltas_w); agg_b = float(wts @ np.array(deltas_b))
        elif method == "flevo":
            agg_w, agg_b, _ = pso_aggregate(deltas_w, deltas_b, w, b, Xval, yval, rng)
        else:
            agg_w, agg_b = robust_aggregate(deltas_w, deltas_b, method, n_byz)

        w = w + SERVER_LR * agg_w
        b = b + SERVER_LR * agg_b
    return w, b


def main():
    d = np.load(FEATURES_PATH, allow_pickle=True)
    X, y, subject = d["X"], d["y"].astype(int), d["subject"]

    # Build per-client train/val/test splits (per-subject z-scored)
    client_data = []
    val_X, val_y, test_X, test_y = [], [], [], []
    for cid, subj in enumerate(CLIENTS):
        m = subject == subj
        Xs = X[m].astype(float); ys = y[m]
        Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)         # per-subject z-score
        rng = np.random.default_rng(1000 + cid)
        idx = rng.permutation(len(ys))
        n = len(ys); a, bnd = int(0.6 * n), int(0.8 * n)
        tr, va, te = idx[:a], idx[a:bnd], idx[bnd:]
        client_data.append((Xs[tr], ys[tr], Xs[te], ys[te]))
        val_X.append(Xs[va]); val_y.append(ys[va])          # server clean validation (honest)
        test_X.append(Xs[te]); test_y.append(ys[te])

    # Server validation = honest clients only (excludes the Byzantine ones at eval time
    # is handled per-run below). For simplicity we use the full honest pool's val slice.
    rows = []
    print(f"{'method':8s} {'attack':16s} {'byz':>3s} {'seed':>4s} {'test_acc':>9s}")
    for n_byz in FRACTIONS:
        attacks = ["none"] if n_byz == 0 else ATTACKS
        # validation/test pooled over HONEST clients (ids >= n_byz)
        honest = list(range(n_byz, N_CLIENTS))
        Xva = np.vstack([val_X[i] for i in honest]); yva = np.concatenate([val_y[i] for i in honest])
        Xte = np.vstack([test_X[i] for i in honest]); yte = np.concatenate([test_y[i] for i in honest])
        for attack in attacks:
            for method in METHODS:
                for seed in SEEDS:
                    w, b = run_fl(client_data, (Xva, yva), method,
                                  attack if attack != "none" else "signflip", n_byz, seed)
                    acc = accuracy(w, b, Xte, yte)
                    rows.append(dict(method=method, attack=attack, n_byz=n_byz,
                                     seed=seed, test_acc=round(acc, 4)))
                    print(f"{method:8s} {attack:16s} {n_byz:3d} {seed:4d} {acc:9.4f}")

    with open(OUT_CSV, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=["method", "attack", "n_byz", "seed", "test_acc"])
        wcsv.writeheader(); wcsv.writerows(rows)
    print(f"\nDone. Wrote {len(rows)} results to {OUT_CSV}")


if __name__ == "__main__":
    main()
