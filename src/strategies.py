"""
Flower Strategies — paper-exact implementation.

4 strategies:
  1. Baseline  — centralised CNN (pooled data, no federation)
  2. FedAvg    — uniform delta-based aggregation
  3. FedProx   — FedAvg + proximal term (μ=0.01)
  4. FL-EVO    — PSO-optimised delta-based aggregation

Aggregation formula (paper Algorithm 1, line 23):
    θ_{t+1} = θ_t + Σ_k w_k · Δθ_k
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes,
    Parameters, NDArrays,
    ndarrays_to_parameters, parameters_to_ndarrays,
)

from model import EEGMLP
from pso   import (pso_aggregate, fisher_score,
                   mean_pairwise_cosine, compute_deltas,
                   delta_aggregate)


def _sample_all(cm):
    return cm.sample(num_clients=cm.num_available())


def _fedavg_delta(global_params, results):
    """
    FedAvg delta-based aggregation:
    θ_new = θ_old + Σ_k (n_k/N) · Δθ_k
    """
    total   = sum(r.num_examples for _, r in results)
    weights = [r.num_examples/total for _, r in results]
    local_p = [parameters_to_ndarrays(r.parameters) for _, r in results]
    deltas  = compute_deltas(global_params, local_p)
    return delta_aggregate(global_params, deltas, weights)


# ── 1. Baseline ───────────────────────────────────────────────────────────────

class BaselineStrategy(fl.server.strategy.Strategy):
    """
    Centralised baseline — no federation.
    Server broadcasts same initial weights every round.
    Clients train locally only (lower-bound reference).
    """

    def __init__(self, initial_parameters, n_epochs=5):
        self.initial_parameters = initial_parameters
        self.n_epochs            = n_epochs
        self.current_parameters  = initial_parameters
        self.round_metrics       = []

    def initialize_parameters(self, client_manager=None, **kwargs):
        return self.initial_parameters

    def configure_fit(self, server_round=None, parameters=None,
                      client_manager=None, **kwargs):
        cfg = {"method":"baseline","n_epochs":self.n_epochs,
               "round": server_round or 0}
        return [(c, FitIns(self.initial_parameters, cfg))
                for c in _sample_all(client_manager)]

    def aggregate_fit(self, server_round=None, results=None,
                      failures=None, **kwargs):
        if not results: return None, {}
        accs = [r.metrics.get("val_acc",0.0) for _,r in results]
        self.round_metrics.append({"round":server_round,
                                   "mean_val_acc":float(np.mean(accs))})
        return self.initial_parameters, {}   # never updates

    def configure_evaluate(self, server_round=None, parameters=None,
                           client_manager=None, **kwargs):
        return [(c, EvaluateIns(self.initial_parameters, {}))
                for c in _sample_all(client_manager)]

    def aggregate_evaluate(self, server_round=None, results=None,
                           failures=None, **kwargs):
        if not results: return None, {}
        accs = [r.metrics.get("accuracy",0.0) for _,r in results]
        return float(np.mean(accs)), {"accuracy":float(np.mean(accs))}

    def evaluate(self, server_round=None, parameters=None, **kwargs):
        return None


# ── 2. FedAvg ─────────────────────────────────────────────────────────────────

class FedAvgStrategy(fl.server.strategy.Strategy):
    """FedAvg with delta-based aggregation (paper Algorithm 1)."""

    def __init__(self, initial_parameters, n_epochs=5):
        self.initial_parameters = initial_parameters
        self.n_epochs            = n_epochs
        self.global_params       = parameters_to_ndarrays(initial_parameters)
        self.current_parameters  = initial_parameters
        self.round_metrics       = []

    def initialize_parameters(self, client_manager=None, **kwargs):
        return self.initial_parameters

    def configure_fit(self, server_round=None, parameters=None,
                      client_manager=None, **kwargs):
        cfg = {"method":"fedavg","n_epochs":self.n_epochs,
               "round": server_round or 0}
        return [(c, FitIns(parameters, cfg))
                for c in _sample_all(client_manager)]

    def aggregate_fit(self, server_round=None, results=None,
                      failures=None, **kwargs):
        if not results: return None, {}
        new_params = _fedavg_delta(self.global_params, results)
        self.global_params      = new_params
        self.current_parameters = ndarrays_to_parameters(new_params)
        accs = [r.metrics.get("val_acc",0.0) for _,r in results]
        self.round_metrics.append({"round":server_round,
                                   "mean_val_acc":float(np.mean(accs))})
        return self.current_parameters, {}

    def configure_evaluate(self, server_round=None, parameters=None,
                           client_manager=None, **kwargs):
        return [(c, EvaluateIns(parameters, {}))
                for c in _sample_all(client_manager)]

    def aggregate_evaluate(self, server_round=None, results=None,
                           failures=None, **kwargs):
        if not results: return None, {}
        accs = [r.metrics.get("accuracy",0.0) for _,r in results]
        return float(np.mean(accs)), {"accuracy":float(np.mean(accs))}

    def evaluate(self, server_round=None, parameters=None, **kwargs):
        return None


# ── 3. FedProx ────────────────────────────────────────────────────────────────

class FedProxStrategy(fl.server.strategy.Strategy):
    """FedProx — delta aggregation same as FedAvg, proximal term client-side."""

    def __init__(self, initial_parameters, n_epochs=5, prox_mu=0.01):
        self.initial_parameters = initial_parameters
        self.n_epochs            = n_epochs
        self.prox_mu             = prox_mu
        self.global_params       = parameters_to_ndarrays(initial_parameters)
        self.current_parameters  = initial_parameters
        self.round_metrics       = []

    def initialize_parameters(self, client_manager=None, **kwargs):
        return self.initial_parameters

    def configure_fit(self, server_round=None, parameters=None,
                      client_manager=None, **kwargs):
        cfg = {"method":"fedprox","n_epochs":self.n_epochs,
               "prox_mu":self.prox_mu,"round": server_round or 0}
        return [(c, FitIns(parameters, cfg))
                for c in _sample_all(client_manager)]

    def aggregate_fit(self, server_round=None, results=None,
                      failures=None, **kwargs):
        if not results: return None, {}
        new_params = _fedavg_delta(self.global_params, results)
        self.global_params      = new_params
        self.current_parameters = ndarrays_to_parameters(new_params)
        accs = [r.metrics.get("val_acc",0.0) for _,r in results]
        self.round_metrics.append({"round":server_round,
                                   "mean_val_acc":float(np.mean(accs))})
        return self.current_parameters, {}

    def configure_evaluate(self, server_round=None, parameters=None,
                           client_manager=None, **kwargs):
        return [(c, EvaluateIns(parameters, {}))
                for c in _sample_all(client_manager)]

    def aggregate_evaluate(self, server_round=None, results=None,
                           failures=None, **kwargs):
        if not results: return None, {}
        accs = [r.metrics.get("accuracy",0.0) for _,r in results]
        return float(np.mean(accs)), {"accuracy":float(np.mean(accs))}

    def evaluate(self, server_round=None, parameters=None, **kwargs):
        return None


# ── 4. FL-EVO (PSO) ──────────────────────────────────────────────────────────

class FLEvoStrategy(fl.server.strategy.Strategy):
    """
    FL-EVO — PSO-optimised delta-based aggregation.
    Paper Algorithm 1 + Algorithm 2.

    Fitness: f(w) = 0.6·acc + 0.3·diversity + 0.1·SNR
    PSO:     w=0.7 (fixed), c1=c2=2.0, 30 particles, 50 iter
    Aggregation: θ_{t+1} = θ_t + Σ_k w_k·Δθ_k
    """

    def __init__(self, initial_parameters, client_features,
                 shared_val, n_epochs=5,
                 n_particles=30, n_iter=50, seed=42):
        self.initial_parameters = initial_parameters
        self.client_features    = client_features
        self.Xfv, self.yv       = shared_val
        self.n_epochs           = n_epochs
        self.n_particles        = n_particles
        self.n_iter             = n_iter
        self.seed               = seed
        self.global_params      = parameters_to_ndarrays(initial_parameters)
        self.current_parameters = initial_parameters
        self.eval_model         = EEGMLP(seed=seed)
        self.round_metrics      = []

        # Pre-compute Fisher scores (fixed across rounds)
        self._fisher = [
            fisher_score(cf['X_flat'], cf['y_train'])
            for cf in client_features
        ]

    def initialize_parameters(self, client_manager=None, **kwargs):
        return self.initial_parameters

    def configure_fit(self, server_round=None, parameters=None,
                      client_manager=None, **kwargs):
        cfg = {"method":"flevo","n_epochs":self.n_epochs,
               "round": server_round or 0}
        return [(c, FitIns(parameters, cfg))
                for c in _sample_all(client_manager)]

    def aggregate_fit(self, server_round=None, results=None,
                      failures=None, **kwargs):
        if not results: return None, {}

        # Sort by client ID
        ordered      = sorted(results,
                               key=lambda x: int(x[1].metrics.get("cid",0)))
        local_params = [parameters_to_ndarrays(r.parameters)
                        for _,r in ordered]
        val_accs     = [r.metrics.get("val_acc",0.5) for _,r in ordered]

        # PSO search (paper Algorithm 2)
        best_w, best_fit, pso_acc = pso_aggregate(
            global_params      = self.global_params,
            local_params_list  = local_params,
            val_accs           = val_accs,
            fisher_scores      = self._fisher,
            model              = self.eval_model,
            Xfv                = self.Xfv,
            yv                 = self.yv,
            seed               = self.seed + (server_round or 0)*7,
            n_particles        = self.n_particles,
            n_iter             = self.n_iter,
        )

        # Delta-based update (paper line 23)
        deltas     = compute_deltas(self.global_params, local_params)
        new_params = delta_aggregate(self.global_params, deltas, best_w)
        self.global_params      = new_params
        self.current_parameters = ndarrays_to_parameters(new_params)

        # Diversity
        dvecs = [np.concatenate([d.ravel() for d in dl]) for dl in deltas]
        div   = mean_pairwise_cosine(dvecs)

        self.round_metrics.append({
            "round"       : server_round,
            "pso_weights" : best_w.tolist(),
            "pso_fitness" : float(best_fit),
            "pso_val_acc" : float(pso_acc),
            "diversity"   : float(div),
        })
        print(f"    [FL-EVO] R{(server_round or 0):02d}: "
              f"fit={best_fit:.4f}  val={pso_acc:.4f}  "
              f"div={div:.4f}  w={best_w.round(2)}")

        return self.current_parameters, {"pso_fitness":float(best_fit)}

    def configure_evaluate(self, server_round=None, parameters=None,
                           client_manager=None, **kwargs):
        return [(c, EvaluateIns(parameters, {}))
                for c in _sample_all(client_manager)]

    def aggregate_evaluate(self, server_round=None, results=None,
                           failures=None, **kwargs):
        if not results: return None, {}
        accs = [r.metrics.get("accuracy",0.0) for _,r in results]
        return float(np.mean(accs)), {"accuracy":float(np.mean(accs))}

    def evaluate(self, server_round=None, parameters=None, **kwargs):
        return None
