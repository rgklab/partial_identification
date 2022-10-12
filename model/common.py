import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback, EarlyStopping

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar

import torch.nn.functional as F
from load_dag import DAG

####################
# Collection of common building blocks for the models/training
####################


def block(n_hidden: int, n_layers: int):
    """Creates a fully-connected ``n_layers`` linear layers with ``n_hidden`` units"""

    layers = []
    for _ in range(n_layers):
        layers += [nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(0.2, inplace=True)]
    return layers


class GumbelMaxBinary(nn.Module):
    """Gumbel-max trick for binary variables"""
    # TODO temperature annealing
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, x):
        return F.gumbel_softmax(x, tau=self.tau, hard=True)


class Generator(nn.Module):
    """The G-constrained generative model. For observational data, we sample from the noise and generate each variable
    according to the topological order of variables in the DAG. For interventional data, we assign the intervened
    variable to the given value and sample the rest of the variables according to the topological order.

    Args:
        dag (DAG): The causal graph.
        n_hidden (int): The number of hidden units in each layer of the generator.
        n_layers (int): The number of layers in the generator.
    """
    def __init__(self, dag: DAG, upper_bound: Optional[Dict[int, torch.Tensor]],
                 lower_bound: Optional[Dict[int, torch.Tensor]], n_hidden: int, n_layers: int):
        super().__init__()
        self.latent_dim = dag.latent_dim
        self.graph = dag.graph
        self.var_dims = dag.var_dims
        self.binary_keys = dag.binary_keys

        self.upper = upper_bound
        self.lower = lower_bound

        self.model_dict = {}

        # Create the model for each variable w.r.t. the topological order
        for key, value in self.graph.items():
            inputs = np.array(value).astype(int)
            if len(inputs) == 0:
                continue
            observed = inputs[inputs >= 0]
            latent = abs(inputs[inputs < 0])
            input_dim = self.latent_dim * len(latent) + np.sum(self.var_dims[observed])
            if n_layers == 0:
                if self.binary_keys is None or key not in self.binary_keys:
                    last_layer = [nn.Linear(input_dim,  self.var_dims[key])]

                else:
                    last_layer = [nn.Linear(input_dim, self.var_dims[key]),
                                  nn.Sigmoid(),
                                  GumbelMaxBinary(0.1)]  # Binary variables

                self.model_dict[key] = nn.Sequential(*last_layer)
            else:
                if self.binary_keys is None or key not in self.binary_keys:
                    last_layer = [nn.Linear(n_hidden, self.var_dims[key])]

                else:
                    last_layer = [nn.Linear(n_hidden, self.var_dims[key]),
                                  nn.Sigmoid(),
                                  GumbelMaxBinary(0.1)]  # Binary variables

                self.model_dict[key] = nn.Sequential(nn.Linear(input_dim, n_hidden), nn.LeakyReLU(0.2, inplace=True),
                                                     *block(n_hidden, n_layers - 1),
                                                     *last_layer, )
        self.models = nn.ModuleList([model for key, model in self.model_dict.items()])

    def _helper_forward(self, z: torch.Tensor, data: torch.Tensor, x: torch.Tensor = None, do_key: int = None):
        """Handles sampling from both observational and interventional data.
        Args:
            z (torch.Tensor): The latent (noise) samples.
            data (torch.Tensor): The empirical observed data from the true distribution. We will not generate the data
                     for root nodes and use the empirical samples instead.
            x (torch.Tensor): The intervention value.
            do_key (int): The intervened variable.
        """
        var = {}
        for key, value in self.graph.items():
            if (do_key is not None) and (key == do_key):
                if x.shape[0] == self.var_dims[do_key]:  # Single intervention for all samples (noise)
                    var[key] = x.reshape((self.var_dims[do_key], -1)).repeat((1, z.shape[0])).t()
                elif x.shape[0] == z.shape[0]:  # Different intervention for each sample (noise)
                    var[key] = x.reshape(-1, x.shape[1])
                else:
                    raise Exception(f'wrong do-var dim. z: {z.shape}, x: {x.shape}')
            else:
                inputs = np.array(value).astype(int)
                if len(inputs) == 0:
                    start = np.sum(self.var_dims[: key])
                    end = np.sum(self.var_dims[: (key + 1)])
                    var[key] = data[:, start:end]   # Empirical data for root nodes
                else:
                    latent = tuple(z[:, (i - 1) * self.latent_dim:i * self.latent_dim] for i in abs(inputs[inputs < 0]))
                    observed = tuple(var[i] for i in inputs[inputs >= 0])
                    var[key] = self.model_dict[key](torch.cat(latent + observed, dim=1))
                    if self.lower is not None and self.upper is not None and key in self.lower:
                        with torch.no_grad():   # clip the values to lower/upper bound for more stable results
                            lower, upper = self.lower[key].type_as(var[key]), self.upper[key].type_as(var[key])
                            var[key].copy_(var[key].data.clamp(min=lower, max=upper))
        observed = tuple(var[i] for i in range(len(self.var_dims)))
        return torch.cat(observed, dim=1)

    def forward(self, z, data):
        return self._helper_forward(z, data)

    def do(self, z, x, do_key, data):
        return self._helper_forward(z, data, x=x, do_key=do_key)


class MetricsCallback(Callback):
    """Callback to log metrics"""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        metric = dict([(k, each_me[k].cpu().numpy()) for k in each_me])
        self.metrics.append(metric)


class StartEstimandOpt(EarlyStopping):
    """EarlyStopping to finish the pretraining phase and set the value of alpha."""

    def on_validation_end(self, trainer, pl_module) -> None:
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not pl_module.pre_train:
            return
        done, _ = self._run_early_stopping_check(trainer)
        if done:
            pl_module.pre_train = False
            metrics = trainer.callback_metrics
            dist_min = metrics['distance_min_network']
            dist_max = metrics['distance_max_network']
            best1 = max(dist_max, dist_min)
            best2 = max(pl_module.best_dist_min, pl_module.best_dist_max)
            if pl_module.alpha == 0.:
                pl_module.alpha = min(best1, best2) * pl_module.tol_coeff

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> tuple:
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return False, None

        current = logs.get(self.monitor)

        should_stop, reason = self._evaluate_stopping_criteria(current)

        if self.verbose and should_stop:
            self._log_info(trainer, "Estimand optimization starts")

        return should_stop, current


class LitProgressBar(TQDMProgressBar):
    """A simple progress bar for Lightning."""
    def init_validation_tqdm(self):
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
        import sys
        bar = Tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=True,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar
