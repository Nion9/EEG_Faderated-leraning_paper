"""
PSO Aggregation — exactly as paper specifies.
=============================================
Paper fitness function:
    f(w) = 0.6·accuracy + 0.3·diversity + 0.1·SNR

PSO parameters (from paper):
    w  = 0.7  (fixed inertia)
    c1 = 2.0  (cognitive)
    c2 = 2.0  (social)
    N  = 30   particles
    G  = 50   iterations

Aggregation (delta-based, from paper Algorithm 1):
    θ_new = θ_old + Σ_k w_k · Δθ_k

Diversity (from paper):
    C_ij = cos(Δθ_i, Δθ_j) = <Δθ_i, Δθ_j> / (||Δθ_i|| · ||Δθ_j||)

SNR (Fisher score, from paper):
    SNR = (μ_c1 - μ_c2)² / (σ²_c1 + σ²_c2)
"""

import numpy as np


# ── Simplex projection (Duchi et al. 2008) ────────────────────────────────────

def project_simplex(v):
    K   = len(v)
    u   = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, K+1, dtype=float)
    rho = int(np.where(u > css/ind)[0][-1]) + 1
    theta = css[rho-1] / float(rho)
    return np.maximum(v - theta, 0.0)


# ── Delta-based aggregation (paper Algorithm 1, line 23) ─────────────────────

def delta_aggregate(global_params, delta_list, w):
    """
    θ_new = θ_old + Σ_k w_k · Δθ_k

    global_params : list of ndarray  (current global model weights)
    delta_list    : list of lists    (K client deltas)
    w             : [K]              convex weights
    Returns new global params list.
    """
    n_layers = len(global_params)
    result   = []
    for li in range(n_layers):
        update = sum(w[k] * delta_list[k][li] for k in range(len(w)))
        result.append(global_params[li] + update)
    return result


def compute_deltas(global_params, local_params_list):
    """
    Δθ_k = θ_k^local - θ_global  (paper Algorithm 1, line 13)
    """
    deltas = []
    for lp in local_params_list:
        delta = [lp[i] - global_params[i] for i in range(len(global_params))]
        deltas.append(delta)
    return deltas


# ── Pairwise cosine distance ──────────────────────────────────────────────────

def pairwise_cosine_matrix(vecs):
    """vecs: list of flat arrays -> [K,K] cosine-distance matrix."""
    K     = len(vecs)
    mat   = np.zeros((K, K))
    norms = np.array([np.linalg.norm(v)+1e-12 for v in vecs])
    for i in range(K):
        for j in range(i+1, K):
            cd = 1.0 - np.dot(vecs[i],vecs[j])/(norms[i]*norms[j])
            mat[i,j] = mat[j,i] = cd
    return mat


def mean_pairwise_cosine(vecs):
    """Mean pairwise cosine distance (diversity metric, paper eq.)"""
    K = len(vecs)
    if K < 2: return 0.0
    norms = [np.linalg.norm(v)+1e-12 for v in vecs]
    dists = [1.0-np.dot(vecs[i],vecs[j])/(norms[i]*norms[j])
             for i in range(K) for j in range(i+1,K)]
    return float(np.mean(dists))


# ── Fisher SNR (paper equation) ───────────────────────────────────────────────

def fisher_score(Xf, y):
    """
    SNR_class = (μ_c1 - μ_c2)² / (σ²_c1 + σ²_c2)
    Averaged over all features.
    """
    c0, c1 = Xf[y==0], Xf[y==1]
    if len(c0) < 2 or len(c1) < 2:
        return 0.0
    num = (c0.mean(0) - c1.mean(0))**2
    den = c0.var(0) + c1.var(0) + 1e-10
    return float((num/den).mean())


# ── PSO fitness (paper exact) ─────────────────────────────────────────────────

def eval_fitness_batch(pos_batch, global_params, delta_list,
                       model, Xfv, yv, cos_mat, fisher_norm,
                       alpha=0.6, beta_d=0.3, beta_s=0.1):
    """
    Evaluate fitness for P candidate weight vectors.
    f(w) = 0.6·accuracy + 0.3·diversity + 0.1·SNR  (paper exact)

    pos_batch : [P, K]
    Returns fits [P], accs [P]
    """
    P    = pos_batch.shape[0]
    fits = np.zeros(P)
    accs = np.zeros(P)
    for i in range(P):
        # Delta-based aggregation (paper Algorithm 1)
        cand_params = delta_aggregate(global_params, delta_list, pos_batch[i])
        model.set_weights(cand_params)
        model.training = False
        acc = model.evaluate(Xfv, yv)

        # Diversity: mean pairwise cosine distance weighted by w
        div = float(0.5 * pos_batch[i] @ cos_mat @ pos_batch[i])

        # SNR: Fisher score weighted by client weights
        snr = np.tanh(float(np.dot(pos_batch[i], fisher_norm)))

        fits[i] = alpha*acc + beta_d*div + beta_s*snr
        accs[i] = acc
    return fits, accs


# ── Main PSO routine (paper Algorithm 2) ─────────────────────────────────────

def pso_aggregate(global_params, local_params_list,
                  val_accs, fisher_scores,
                  model, Xfv, yv, seed,
                  n_particles=30, n_iter=50,
                  w_inertia=0.7,   # paper: fixed w=0.7
                  c1=2.0, c2=2.0): # paper: c1=c2=2.0
    """
    PSO weight optimisation (paper Algorithm 2).

    global_params      : current global model weights (list of ndarray)
    local_params_list  : K client local model weights
    val_accs           : K local validation accuracies
    fisher_scores      : K Fisher SNR scores

    Returns best_w [K], best_fit float, best_acc float
    """
    K   = len(local_params_list)
    rng = np.random.RandomState(seed)

    # Compute deltas: Δθ_k = θ_k - θ_global
    delta_list = compute_deltas(global_params, local_params_list)

    # Flat delta vectors for diversity computation
    dvecs   = [np.concatenate([d.ravel() for d in dl]) for dl in delta_list]
    cos_mat = pairwise_cosine_matrix(dvecs)

    # Fisher scores normalised
    fs   = np.array(fisher_scores, dtype=float)
    fs_n = (fs - fs.min()) / (fs.max()-fs.min()+1e-10)

    # ── Initialise swarm ──────────────────────────────────────────────
    acc_arr = np.maximum(np.array(val_accs, dtype=float), 0)
    w_acc   = acc_arr / (acc_arr.sum()+1e-10)
    w_snr   = fs_n   / (fs_n.sum()+1e-10)
    w_comb  = 0.6*w_acc + 0.4*w_snr
    w_comb  = w_comb / w_comb.sum()

    pos = np.zeros((n_particles, K))
    pos[0] = project_simplex(w_acc)
    pos[1] = project_simplex(w_snr)
    pos[2] = project_simplex(w_comb)
    pos[3] = np.ones(K)/K
    for i in range(4, n_particles):
        r = rng.exponential(1, K); pos[i] = r/r.sum()
    for i in range(n_particles):
        pos[i] = project_simplex(pos[i])

    vel = rng.uniform(-0.05, 0.05, (n_particles, K))

    # Initial fitness
    fits, accs = eval_fitness_batch(
        pos, global_params, delta_list, model,
        Xfv, yv, cos_mat, fs_n)
    pbest     = pos.copy()
    pbest_fit = fits.copy()
    gi        = int(np.argmax(fits))
    gbest     = pos[gi].copy()
    gbest_fit = fits[gi]
    gbest_acc = accs[gi]

    vmax = 0.4

    # ── PSO main loop (paper Algorithm 2) ────────────────────────────
    for g in range(n_iter):
        # Paper: w=0.7 fixed (not decaying)
        r1 = rng.rand(n_particles, K)
        r2 = rng.rand(n_particles, K)
        vel = (w_inertia*vel
               + c1*r1*(pbest-pos)
               + c2*r2*(gbest-pos))
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos+vel, 0, None)
        for i in range(n_particles):
            pos[i] = project_simplex(pos[i])

        fits, accs = eval_fitness_batch(
            pos, global_params, delta_list, model,
            Xfv, yv, cos_mat, fs_n)

        improve         = fits > pbest_fit
        pbest[improve]     = pos[improve].copy()
        pbest_fit[improve] = fits[improve]

        gi = int(np.argmax(fits))
        if fits[gi] > gbest_fit:
            gbest_fit = fits[gi]
            gbest     = pos[gi].copy()
            gbest_acc = accs[gi]

    return gbest, gbest_fit, gbest_acc
