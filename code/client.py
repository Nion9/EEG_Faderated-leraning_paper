"""
Flower Client — FL-EVO BCI.
Paper spec:
  - CNN local training (5 epochs per round)
  - Returns local weights to server
  - Local integration: w_new = α·w_global + (1-α)·w_local, α=0.5
"""

import numpy as np
import flwr as fl
from flwr.common import (
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    GetParametersIns, GetParametersRes, Status, Code,
    ndarrays_to_parameters, parameters_to_ndarrays,
)

from model import EEGMLP

BATCH_SIZE = 32
ALPHA      = 0.5   # local integration: paper specifies α=0.5


class EEGClient(fl.client.Client):
    """
    Flower client — one EEG subject.
    Trains CNN locally and returns updated weights.
    """

    def __init__(self, cid, X_train, y_train,
                 X_val, y_val, X_test, y_test, seed=42):
        self.cid     = cid
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val
        self.X_test  = X_test
        self.y_test  = y_test
        self.seed    = seed
        self.model   = EEGMLP(seed=seed, lr=0.001)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(
            status     = Status(code=Code.OK, message=""),
            parameters = ndarrays_to_parameters(self.model.get_weights()),
        )

    def fit(self, ins: FitIns) -> FitRes:
        # Receive global weights
        global_weights = parameters_to_ndarrays(ins.parameters)
        self.model.set_weights(global_weights)

        config   = ins.config
        method   = config.get("method", "fedavg")
        n_epochs = int(config.get("n_epochs", 5))
        prox_mu  = float(config.get("prox_mu", 0.0))
        rnd      = int(config.get("round", 0))

        np.random.seed(self.seed + rnd * 97)

        # Store global for FedProx / local integration
        global_W = {k: getattr(self.model, k).copy()
                    for k in self.model.PARAM_KEYS}

        # Local training
        for _ in range(n_epochs):
            self.model.train_epoch(self.X_train, self.y_train, BATCH_SIZE)

            # FedProx proximal term
            if method == "fedprox" and prox_mu > 0:
                for k in ['W_temp','W_spat','W_dep','W_fc1','W_fc2']:
                    cur  = getattr(self.model, k)
                    cur -= self.model.lr * prox_mu * (cur - global_W[k])

        # Local integration (paper Step 5): w_new = α·w_global + (1-α)·w_local
        if method in ("fedavg", "fedprox", "flevo"):
            for k in self.model.PARAM_KEYS:
                local_w  = getattr(self.model, k)
                global_w = global_W[k]
                setattr(self.model, k,
                        ALPHA*global_w + (1-ALPHA)*local_w)

        # Validation accuracy for PSO fitness
        val_acc = self.model.evaluate(self.X_val, self.y_val)

        return FitRes(
            status       = Status(code=Code.OK, message=""),
            parameters   = ndarrays_to_parameters(self.model.get_weights()),
            num_examples = len(self.y_train),
            metrics      = {"val_acc": float(val_acc),
                            "cid"    : int(self.cid)},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        weights = parameters_to_ndarrays(ins.parameters)
        self.model.set_weights(weights)
        self.model.training = False
        acc = self.model.evaluate(self.X_test, self.y_test)
        return EvaluateRes(
            status       = Status(code=Code.OK, message=""),
            loss         = float(1.0 - acc),
            num_examples = len(self.y_test),
            metrics      = {"accuracy": float(acc)},
        )


def make_client_fn(client_data_list, seed=42):
    def client_fn(cid: str) -> fl.client.Client:
        idx  = int(cid)
        data = client_data_list[idx]
        return EEGClient(
            cid     = cid,
            X_train = data['X_train'],
            y_train = data['y_train'],
            X_val   = data['X_val'],
            y_val   = data['y_val'],
            X_test  = data['X_test'],
            y_test  = data['y_test'],
            seed    = seed + idx * 13,
        )
    return client_fn
