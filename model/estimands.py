from typing import Callable
import torch
import numpy as np
from common import Generator
from load_dag import DAG

####################
# Collection of causal estimands. We will assume dim(do_var) = 1.
####################


class Estimand:
    r"""Describes the estimand of interest (interventional quantity) and calculates it given the noise and generator.
    Args:
        param_fn (Callable): Function that calculates the estimand given the noise and generator
        name (str): Name of the estimand
    """
    def __init__(self, param_fn: Callable, name: str):
        self.param_fn = param_fn
        self.name = name

        self.interval = None  # The interval of interest, e.g. [t0, t1] for ATE = E[Y(t1) - Y(t0)]

    def __call__(self, z: torch.Tensor, generator: Generator, device: torch.device, data: torch.Tensor):
        r"""Calculate the estimand given the noise and generator
        Args:
            z (torch.Tensor): Noise samples
            generator (Generator)
            device (torch.device): 'cpu' or 'cuda:i' where i is the GPU index
            data (torch.Tensor): Data samples from the batch
        """
        return self.param_fn(z, generator, device, data)

    def set_interval(self, interval: tuple):
        self.interval = interval

    def __str__(self):
        return self.name


def create_atd(dag: DAG, delta: float = 0.1):
    """Average treatment derivative (ATD) estimand.

    Args:
        dag (DAG): The causal graph.
        delta (float): ATD ≈ E[Y(T + delta) - Y(T)] / delta.
    """
    target_start = np.sum(dag.var_dims[: dag.target_var])
    target_end = np.sum(dag.var_dims[: (dag.target_var + 1)])

    do_start = np.sum(dag.var_dims[: dag.do_var])
    do_end = np.sum(dag.var_dims[: (dag.do_var + 1)])

    def param_fn(z, generator, device, data):
        d = torch.ones(dag.var_dims[dag.do_var]) * torch.Tensor([delta])

        zero = data[:, do_start:do_end]
        one = data[:, do_start:do_end] + d.to(device)
        cntf0 = generator.do(z, zero.to(device), dag.do_var, data=data)[:, target_start:target_end]
        cntf1 = generator.do(z, one.to(device), dag.do_var, data=data)[:, target_start:target_end]
        return (cntf1 - cntf0) / delta

    return Estimand(param_fn, 'average_treatment_derivative')


def create_uatd_gauss(dag: DAG, interval: tuple, delta: float = 0.1, std: float = None):
    """Uniform average treatment derivative (UATD) with Gaussian tails.

    Args:
        dag (DAG): The causal graph.
        interval (tuple): [t0, t1] if ATE = E[Y(t1) - Y(t0)]
        delta (float): UATD = E[Y(U + delta) - Y(U)] / delta, where U is uniform in [t0, t1] and has Gaussian tails
                             outside [t0, t1].
        std (float): Standard deviation of the Gaussian tails. If None, then std = (t1 - t0) / 2.

    """
    target_start = np.sum(dag.var_dims[: dag.target_var])
    target_end = np.sum(dag.var_dims[: (dag.target_var + 1)])

    def param_fn(z, generator, device, data):
        mean = (interval[1] + interval[0]) / 2
        if std is None:
            std2 = (interval[1] - interval[0]) / 2
        else:
            std2 = std

        uniform_noises = torch.rand(size=(z.shape[0], 1)) * (interval[1] - interval[0]) + interval[0]
        if std == 0:    # only uniform intervention
            treatments = uniform_noises
        else:
            treatments = torch.randn(size=(z.shape[0], 1)) * std2 + mean
            within_interval = (treatments[:, 0] <= interval[1]) & (treatments[:, 0] >= interval[0])
            treatments[within_interval, :] = uniform_noises[within_interval, :]
        one = treatments + delta
        zero = treatments
        cntf1 = generator.do(z, one.to(device), dag.do_var, data=data)[:, target_start:target_end]
        cntf0 = generator.do(z, zero.to(device), dag.do_var, data=data)[:, target_start:target_end]
        return (cntf1 - cntf0) / delta

    estimand = Estimand(param_fn, 'uniform_average_treatment_derivative')
    estimand.set_interval(interval)
    return estimand


def create_uniform_ate(dag: DAG, interval: tuple = None):
    """Uniform average treatment effect (UATE).

    Args:
        dag (DAG): The causal graph.
        interval (tuple): [t0, t1] if UATE = E[Y(U)] for U ~ Uniform[t0, t1].
                          If None, then interval = [min(treatment), max(treatment)].
    """
    target_start = np.sum(dag.var_dims[: dag.target_var])
    target_end = np.sum(dag.var_dims[: (dag.target_var + 1)])

    do_start = np.sum(dag.var_dims[: dag.do_var])
    do_end = np.sum(dag.var_dims[: (dag.do_var + 1)])

    def param_fn(z, generator, device, data):
        if interval is None:
            rng = (torch.min(data[:, do_start:do_end]).to('cpu'), torch.max(data[:, do_start:do_end]).to('cpu'))
        else:
            rng = interval
        treatments = torch.rand(size=(z.shape[0], 1)) * (rng[1] - rng[0]) + rng[0]
        cntf = generator.do(z, treatments.to(device), dag.do_var, data=data)[:, target_start:target_end]
        return cntf

    estimand = Estimand(param_fn, 'uniform_average_treatment_effect')
    estimand.set_interval(interval)
    return estimand


def create_ate(dag: DAG, interval: tuple):
    """Average treatment effect (ATE).

       Args:
           dag (DAG): The causal graph.
           interval (tuple): [t0, t1] if ATE = E[Y(t1) - Y(t0)]

       """
    target_start = np.sum(dag.var_dims[: dag.target_var])
    target_end = np.sum(dag.var_dims[: (dag.target_var + 1)])

    def param_fn(z, generator, device, data):
        cntf1 = generator.do(z,
                             torch.Tensor([interval[1]]).to(device),
                             dag.do_var, data=data)[:, target_start:target_end]
        cntf0 = generator.do(z,
                             torch.Tensor([interval[0]]).to(device),
                             dag.do_var, data=data)[:, target_start:target_end]
        return cntf1 - cntf0

    estimand = Estimand(param_fn, 'average_treatment_effect')
    estimand.set_interval(interval)
    return estimand


def create_discrete_ate(dag: DAG):
    """Average treatment effect (ATE) with binary treatment.

       Args:
           dag (DAG): The causal graph.
       """
    target_start = np.sum(dag.var_dims[: dag.target_var])
    target_end = np.sum(dag.var_dims[: (dag.target_var + 1)])

    def param_fn(z, generator, device, data):
        cntf0 = generator.do(z, torch.Tensor([1., 0.]).to(device), dag.do_var, data)[:, target_start:target_end]
        cntf1 = generator.do(z, torch.Tensor([0., 1.]).to(device), dag.do_var, data)[:, target_start:target_end]
        return cntf1 - cntf0

    return Estimand(param_fn, 'average_treatment_effect_discrete')
