"""
FL-EVO BCI — Main Experiment (Paper-Exact Implementation)
==========================================================
Paper specification:
  - 40 subjects (S001-S040) as FL clients
  - EEGNet-inspired CNN
  - 10 federated rounds, 5 local epochs
  - PSO: w=0.7, c1=c2=2.0, 30 particles, 50 iter
  - Fitness: 0.6·acc + 0.3·diversity + 0.1·SNR
  - 5 seeds, paired t-test for significance
  - Baselines: Baseline, FedAvg, FedProx

Run: python experiment.py
Requirements: pip install flwr mne numpy scipy matplotlib
"""

import os, sys, time, json, csv, warnings
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
import logging
logging.getLogger("flwr").setLevel(logging.WARNING)

import flwr as fl

from data       import load_all_subjects, build_shared_val
from model      import EEGMLP
from client     import make_client_fn
from strategies import (BaselineStrategy, FedAvgStrategy,
                         FedProxStrategy, FLEvoStrategy)
from pso        import fisher_score, mean_pairwise_cosine

# ── Config (paper exact) ──────────────────────────────────────────────────────
CFG = dict(
    n_subjects  = 40,    # 40 FL clients
    n_rounds    = 10,    # paper: 10 rounds
    n_epochs    = 5,     # paper: 5 local epochs
    n_seeds     = 5,     # paper: 5 seeds
    seeds       = [42, 123, 2024, 7, 99],
    n_particles = 30,    # paper: 30 particles
    n_iter      = 50,    # paper: 50 iterations
    fedprox_mu  = 0.01,  # paper: μ=0.01
    noise_frac  = 0.20,  # paper: 20% Gaussian noise
)

METHODS     = ['Baseline', 'FedAvg', 'FedProx', 'FL-EVO (PSO)']
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Feature flattening for Fisher score (uses raw EEG power) ─────────────────

def flatten_for_fisher(X_raw):
    """
    Flatten EEG epochs to feature vectors for Fisher SNR computation.
    X_raw: [N, 1, 64, 640] -> [N, 64] (RMS per channel)
    """
    return np.sqrt((X_raw[:,0,:,:]**2).mean(axis=2))


def prepare_client_data(clients):
    """Build client data list for Flower + pre-compute Fisher features."""
    client_data   = []
    client_fisher = []

    for c in clients:
        Xtr, ytr = c['train']
        Xv,  yv  = c['val']
        Xte, yte = c['test']

        # Flatten for Fisher score
        Xf_tr = flatten_for_fisher(Xtr)

        client_data.append({
            'X_train': Xtr, 'y_train': ytr,
            'X_val':   Xv,  'y_val':   yv,
            'X_test':  Xte, 'y_test':  yte,
        })
        client_fisher.append({
            'X_flat':  Xf_tr,
            'y_train': ytr,
        })

    return client_data, client_fisher


# ── Shared validation features (for PSO fitness) ──────────────────────────────

def prepare_shared_val(clients, n_subjects=5, seed=42):
    """Pool val sets from first n_subjects, flatten for PSO evaluation."""
    rng  = np.random.RandomState(seed)
    Xs   = [c['val'][0] for c in clients[:n_subjects]]
    ys   = [c['val'][1] for c in clients[:n_subjects]]
    Xv   = np.concatenate(Xs)
    yv   = np.concatenate(ys)
    perm = rng.permutation(len(yv))
    return Xv[perm], yv[perm]


# ── Robustness evaluation (paper: 20% Gaussian noise) ────────────────────────

def evaluate_robustness(parameters, client_data, noise_frac=0.20, seed=42):
    from flwr.common import parameters_to_ndarrays
    model = EEGMLP(seed=seed)
    model.set_weights(parameters_to_ndarrays(parameters))
    model.training = False
    rng     = np.random.RandomState(seed)
    total_n = sum(len(c['y_test']) for c in client_data)
    acc     = 0.0
    for c in client_data:
        X     = c['X_test'].astype(np.float64)
        noise = rng.randn(*X.shape) * X.std() * noise_frac
        a     = model.evaluate(X + noise, c['y_test'])
        acc  += a * len(c['y_test']) / total_n
    return float(acc)


# ── Run one method / seed ─────────────────────────────────────────────────────

def run_method(method_name, seed, client_data, client_fisher,
               shared_val, n_rounds, n_epochs):
    from flwr.common import ndarrays_to_parameters

    init_model  = EEGMLP(seed=seed)
    init_params = ndarrays_to_parameters(init_model.get_weights())

    if method_name == 'Baseline':
        strategy = BaselineStrategy(init_params, n_epochs=n_epochs)

    elif method_name == 'FedAvg':
        strategy = FedAvgStrategy(init_params, n_epochs=n_epochs)

    elif method_name == 'FedProx':
        strategy = FedProxStrategy(init_params, n_epochs=n_epochs,
                                   prox_mu=CFG['fedprox_mu'])

    elif method_name == 'FL-EVO (PSO)':
        strategy = FLEvoStrategy(
            initial_parameters = init_params,
            client_features    = client_fisher,
            shared_val         = shared_val,
            n_epochs           = n_epochs,
            n_particles        = CFG['n_particles'],
            n_iter             = CFG['n_iter'],
            seed               = seed,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    client_fn = make_client_fn(client_data, seed=seed)

    fl_history = fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = len(client_data),
        config           = fl.server.ServerConfig(num_rounds=n_rounds),
        strategy         = strategy,
        client_resources = {"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args    = {"include_dashboard": False,
                            "ignore_reinit_error": True,
                            "log_to_driver": False},
    )

    round_accs = []
    if "accuracy" in fl_history.metrics_distributed:
        for _, acc in fl_history.metrics_distributed["accuracy"]:
            round_accs.append(float(acc))
    else:
        round_accs = [0.0] * n_rounds

    rounds_to_75 = n_rounds + 1
    for i, a in enumerate(round_accs):
        if a >= 0.75:
            rounds_to_75 = i + 1; break

    diversities = []
    if hasattr(strategy, 'round_metrics'):
        for rm in strategy.round_metrics:
            diversities.append(rm.get('diversity', 0.0))
    if not diversities:
        diversities = [0.0] * n_rounds

    return {
        'round_acc'    : round_accs,
        'diversity'    : diversities,
        'rounds_to_75' : rounds_to_75,
        'strategy'     : strategy,
    }


# ── Collector ─────────────────────────────────────────────────────────────────

class Collector:
    def __init__(self):
        self.data = {m: dict(final_acc=[], robustness=[],
                             rounds_to_75=[], mean_div=[],
                             round_accs=[]) for m in METHODS}

    def add(self, method, hist, rob):
        d = self.data[method]
        d['final_acc'].append(hist['round_acc'][-1]
                              if hist['round_acc'] else 0.0)
        d['robustness'].append(rob)
        d['rounds_to_75'].append(hist['rounds_to_75'])
        d['mean_div'].append(float(np.mean(hist['diversity']))
                             if hist['diversity'] else 0.0)
        d['round_accs'].append(list(hist['round_acc']))

    def ci(self, method, key):
        arr = np.asarray(self.data[method][key], float)
        return arr.mean(), arr.std(ddof=1)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(collector, n_rounds):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    COLORS = {'Baseline':'#95a5a6','FedAvg':'#e74c3c',
              'FedProx':'#e67e22','FL-EVO (PSO)':'#27ae60'}
    LSTYLE = {'Baseline':':','FedAvg':'--',
              'FedProx':'-.','FL-EVO (PSO)':'-'}
    rounds = np.arange(1, n_rounds+1)
    arrs   = {m: np.array(collector.data[m]['round_accs']) for m in METHODS}

    # Fig 1 — accuracy curves
    fig, ax = plt.subplots(figsize=(9,5))
    for m in METHODS:
        mn = arrs[m].mean(0); sd = arrs[m].std(0, ddof=1)
        ax.plot(rounds, mn, LSTYLE[m], color=COLORS[m], lw=2.5,
                label=m, marker='o', ms=4)
        ax.fill_between(rounds, mn-sd, mn+sd, alpha=0.15, color=COLORS[m])
    ax.axhline(0.75, color='gray', ls=':', lw=1, label='75% target')
    ax.set_xlabel('Federated Round', fontsize=13)
    ax.set_ylabel('Test Accuracy', fontsize=13)
    ax.set_title('FL-EVO vs Baselines — Test Accuracy over Rounds\n'
                 f'(PhysioNet EEG, {CFG["n_subjects"]} subjects, '
                 f'{CFG["n_seeds"]} seeds)', fontsize=12)
    ax.legend(fontsize=11); ax.set_ylim(0.4,1.02); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'accuracy_curves.png'), dpi=150)
    plt.close()

    # Fig 2 — bar chart
    x = np.arange(len(METHODS)); w = 0.35
    acc_m = [collector.ci(m,'final_acc')[0]  for m in METHODS]
    acc_e = [collector.ci(m,'final_acc')[1]  for m in METHODS]
    rob_m = [collector.ci(m,'robustness')[0] for m in METHODS]
    rob_e = [collector.ci(m,'robustness')[1] for m in METHODS]
    fig,ax = plt.subplots(figsize=(9,5))
    b1 = ax.bar(x-w/2, acc_m, w, yerr=acc_e, capsize=5,
                color=[COLORS[m] for m in METHODS], alpha=0.85, label='Clean')
    b2 = ax.bar(x+w/2, rob_m, w, yerr=rob_e, capsize=5,
                color=[COLORS[m] for m in METHODS], alpha=0.45,
                label='Noisy (20%)')
    ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title(f'Final Accuracy & Robustness '
                 f'(mean±std, {CFG["n_seeds"]} seeds)', fontsize=12)
    ax.set_ylim(0.4,1.07); ax.legend(fontsize=11); ax.grid(axis='y',alpha=0.3)
    for bar,val in zip(b1,acc_m):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'accuracy_bar.png'), dpi=150)
    plt.close()

    # Fig 3 — convergence
    r75_m = [collector.ci(m,'rounds_to_75')[0] for m in METHODS]
    r75_e = [collector.ci(m,'rounds_to_75')[1] for m in METHODS]
    fig,ax = plt.subplots(figsize=(7,4))
    ax.bar(METHODS,r75_m,yerr=r75_e,capsize=5,
           color=[COLORS[m] for m in METHODS],alpha=0.85)
    ax.set_ylabel('Rounds to 75% Accuracy',fontsize=13)
    ax.set_title('Convergence Speed (lower = better)',fontsize=12)
    ax.grid(axis='y',alpha=0.3)
    for i,(v,e) in enumerate(zip(r75_m,r75_e)):
        ax.text(i,v+e+0.05,f'{v:.1f}',ha='center',fontsize=10,fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'convergence.png'),dpi=150)
    plt.close()

    # Fig 4 — diversity
    dv_m = [collector.ci(m,'mean_div')[0] for m in METHODS]
    dv_e = [collector.ci(m,'mean_div')[1] for m in METHODS]
    fig,ax = plt.subplots(figsize=(7,4))
    ax.bar(METHODS,dv_m,yerr=dv_e,capsize=5,
           color=[COLORS[m] for m in METHODS],alpha=0.85)
    ax.set_ylabel('Mean Pairwise Cosine Distance',fontsize=13)
    ax.set_title('Update Diversity per Round',fontsize=12)
    ax.grid(axis='y',alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'diversity.png'),dpi=150)
    plt.close()
    print(f"  Figures saved to {RESULTS_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    T0 = time.time()
    print("="*70)
    print("  FL-EVO BCI — Paper-Exact Implementation")
    print("  Fitness : 0.6·acc + 0.3·diversity + 0.1·SNR")
    print("  PSO     : w=0.7 (fixed), c1=c2=2.0")
    print("  Model   : EEGNet-inspired CNN")
    print(f"  Config  : {CFG['n_subjects']} subjects | "
          f"{CFG['n_rounds']} rounds | "
          f"{CFG['n_epochs']} local epochs | "
          f"{CFG['n_seeds']} seeds")
    print("="*70)

    # Load real PhysioNet data
    print(f"\n[1] Loading PhysioNet EEG data ({CFG['n_subjects']} subjects) ...")
    clients = load_all_subjects(
                  n_subjects   = CFG['n_subjects'],
                  seed         = 42,
                  verbose      = True)

    # Prepare data structures
    print("\n[2] Preparing features and shared validation set ...")
    client_data, client_fisher = prepare_client_data(clients)
    shared_val = prepare_shared_val(clients, n_subjects=5, seed=42)
    print(f"  Shared val: {shared_val[0].shape}")

    # Run experiment
    collector = Collector()
    print(f"\n[3] Running {CFG['n_seeds']} seeds × {len(METHODS)} methods ...")

    for si, seed in enumerate(CFG['seeds']):
        print(f"\n{'─'*65}")
        print(f"  SEED {seed}  ({si+1}/{CFG['n_seeds']})")
        print(f"{'─'*65}")

        for method in METHODS:
            t0 = time.time()
            print(f"\n  [{method}]")
            hist = run_method(method, seed, client_data, client_fisher,
                              shared_val, CFG['n_rounds'], CFG['n_epochs'])

            strat = hist['strategy']
            fp    = (strat.current_parameters
                     if hasattr(strat,'current_parameters')
                     else strat.initial_parameters)

            rob = evaluate_robustness(fp, client_data,
                                      CFG['noise_frac'], seed)
            collector.add(method, hist, rob)

            fa  = hist['round_acc'][-1] if hist['round_acc'] else 0.0
            r75 = hist['rounds_to_75']
            print(f"    acc={fa:.4f}  rob={rob:.4f}  "
                  f"rounds→75={r75}  ({time.time()-t0:.1f}s)")

        print(f"\n  ── Seed {seed} summary ──")
        for m in METHODS:
            fa  = collector.data[m]['final_acc'][-1]
            rob = collector.data[m]['robustness'][-1]
            r75 = collector.data[m]['rounds_to_75'][-1]
            print(f"    {m:<18}: acc={fa:.4f}  "
                  f"rob={rob:.4f}  rounds→75={r75}")

        elapsed   = (time.time()-T0)/60
        remaining = elapsed/(si+1)*(CFG['n_seeds']-si-1)
        print(f"\n  Elapsed: {elapsed:.1f} min | "
              f"Est. remaining: {remaining:.1f} min")

    # Final results table
    print("\n\n" + "="*95)
    print(f"{'Method':<18} {'Accuracy':>18} {'Robustness':>18} "
          f"{'Rounds→75%':>12} {'Diversity':>14}")
    print("─"*95)
    summary = []
    for m in METHODS:
        am,as_   = collector.ci(m,'final_acc')
        rm,rs    = collector.ci(m,'robustness')
        r75m,r75s= collector.ci(m,'rounds_to_75')
        dvm,dvs  = collector.ci(m,'mean_div')
        summary.append(dict(Method=m,
                             acc_mean=am, acc_std=as_,
                             rob_mean=rm, rob_std=rs,
                             r75_mean=r75m, r75_std=r75s,
                             div_mean=dvm, div_std=dvs))
        print(f"  {m:<16} {am:.4f} ± {as_:.4f}    "
              f"{rm:.4f} ± {rs:.4f}  "
              f"{r75m:.1f} ± {r75s:.1f}    "
              f"{dvm:.4f} ± {dvs:.4f}")
    print("="*95)

    # Statistical significance (paper: paired t-test, p<0.05, 5 seeds)
    print("\nStatistical significance — paired t-test (FL-EVO vs others):")
    pso_accs = np.array(collector.data['FL-EVO (PSO)']['final_acc'])
    for m in ['Baseline','FedAvg','FedProx']:
        other = np.array(collector.data[m]['final_acc'])
        diff  = pso_accs - other
        if diff.std() < 1e-10:
            print(f"  PSO vs {m}: no variance"); continue
        t,p = stats.ttest_rel(pso_accs, other)
        sig = "✓ p<0.05 SIGNIFICANT" if p<0.05 else "✗ not significant"
        print(f"  PSO vs {m:<10}: t={t:+.3f}, p={p:.4f}  {sig}")

    print("\nImprovement of FL-EVO:")
    pso_mean = np.mean(pso_accs)
    for m in ['Baseline','FedAvg','FedProx']:
        oth = np.mean(collector.data[m]['final_acc'])
        print(f"  vs {m:<10}: +{(pso_mean-oth)*100:.2f} pp")

    # Save results
    csv_path = os.path.join(RESULTS_DIR,'results_summary.csv')
    with open(csv_path,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader(); w.writerows(summary)

    arrs = {m:np.array(collector.data[m]['round_accs']) for m in METHODS}
    rc_path = os.path.join(RESULTS_DIR,'round_accuracy.csv')
    with open(rc_path,'w',newline='') as f:
        w = csv.writer(f)
        header = ['Round']
        for m in METHODS: header += [f'{m}_mean',f'{m}_std']
        w.writerow(header)
        for r in range(CFG['n_rounds']):
            row = [r+1]
            for m in METHODS:
                row += [f"{arrs[m][:,r].mean():.6f}",
                        f"{arrs[m][:,r].std(ddof=1):.6f}"]
            w.writerow(row)

    json_data = {
        'config' : CFG,
        'summary': [{k:(float(v) if isinstance(v,float) else v)
                     for k,v in r.items()} for r in summary],
        'round_curves': {
            m:{'mean':arrs[m].mean(0).tolist(),
               'std' :arrs[m].std(0,ddof=1).tolist()}
            for m in METHODS},
    }
    with open(os.path.join(RESULTS_DIR,'full_results.json'),'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")
    plot_results(collector, CFG['n_rounds'])

    total = (time.time()-T0)/60
    print(f"\nTotal time: {total:.1f} min")
    print(f"Results in: {RESULTS_DIR}/")
