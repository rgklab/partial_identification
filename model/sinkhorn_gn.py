import warnings
from abc import ABC
from argparse import ArgumentParser
from typing import Optional, Union, Callable, Dict

from pytorch_lightning import LightningModule
from torch.optim import Adam, Optimizer
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from common import Generator

from geomloss import SamplesLoss

from itertools import chain

from estimands import Estimand
from load_dag import DAG


class SinkhornGN(LightningModule, ABC):
    r"""Creates a Sinkhorn Generative Network.
    Args:
        estimand (Estimand): The estimand of interest, e.g., ATE, ATD, UATD, etc.

        dag (DAG): The causal graph.

        loss (Callable): The differentiable loss function to compare generated samples to real ones. We will primarily
        use the Sinkhorn divergence (an approximation to Wasserstein distance) as the loss function.

        n_hidden (int): The number of hidden units in the generator.

        n_layers (int): The number of layers in the generator.

        monitor_estimand (Estimand): The estimand to monitor during training. No optimization will be performed on this.

        lr (float, default = ``0.001``): The learning rate for training the generator.

        lagrange_lr (float, default = ``0.1``): The learning rate for Lagrange multipliers.

        noise (str, default = ``normal``): The type of noise to use for the generator. Can be ``normal`` or ``uniform``.

        lower_bound (Dict[int, torch.Tensor], default = ``None``): The lower bound of the observed variables. We use
        this to constrain the generator to generate samples within the observed range.

        upper_bound (Dict[int, torch.Tensor], default = ``None``): The upper bound of the observed variables.

        alpha (float, default = ``0.``): The distribution constraint radius. If ``0.``, the radius will be chosen
        based on the best distance between the generator and the data distribution.

        tol_coeff (float, default = ``1``): if alpha = ``0``, then we set alpha = tol_coeff * best_dist.

    """

    def __init__(self, estimand: Estimand, dag: DAG, loss: Optional[Callable], n_hidden: int, n_layers: int,
                 monitor_estimand: Optional[Estimand] = None, lr: float = 0.001, lagrange_lr: float = 0.1,
                 noise: Optional[str] = "normal", lower_bound: Dict[int, torch.Tensor] = None,
                 upper_bound: Dict[int, torch.Tensor] = None, alpha: float = 0., tol_coeff: float = 1, **kwargs):

        super().__init__()
        self.save_hyperparameters(ignore=['estimand', 'dag', 'monitor_estimand'])
        self.estimand = estimand
        self.noise = noise
        self.dag = dag
        self.lr = lr
        self.lagrange_lr = lagrange_lr
        self.monitor_estimand = monitor_estimand

        # create min and max generative networks
        Gen = Generator
        generator_params = {'dag': self.dag, 'n_hidden': n_hidden, 'n_layers': n_layers, 'lower_bound': lower_bound,
                            'upper_bound': upper_bound}

        self.generator_min = Gen(**generator_params)
        self.generator_max = Gen(**generator_params)

        if loss is None:
            self.loss = SamplesLoss(loss="sinkhorn", p=1, blur=0.01, scaling=0.9, backend='tensorized')
        else:
            self.loss = loss

        self.lagrangian_min, self.lagrangian_max = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))

        self.best_dist_min = np.inf
        self.best_dist_max = np.inf
        self.pre_train = True  # pre-train the generator to match the data distribution before
        # minimizing/maximizing the estimand of interest

        self.optimizer_flag = True  # flag to reset the optimizer after pre-training
        self.alpha = alpha  # The distribution constraint radius
        self.tol_coeff = tol_coeff  # alpha = tol_coeff * best_distance

    def forward(self, z: torch.Tensor, data: torch.Tensor):
        return torch.cat([self.generator_min(z, data), self.generator_max(z, data)], dim=1)

    def _sample_noise(self, imgs: torch.Tensor):
        if self.noise == 'normal':
            z = torch.randn(imgs.shape[0], self.dag.latent_dim * self.dag.n_latent)
        elif self.noise == 'uniform':
            z = torch.rand(imgs.shape[0], self.dag.latent_dim * self.dag.n_latent)
        else:
            raise NotImplementedError

        z = z.type_as(imgs)
        return z

    def _calculate_loss(self, fake: torch.Tensor, real: torch.Tensor):
        # impute missing values with zeros
        fake[real.isnan()] = 0.
        real = real.nan_to_num(0.)
        total_loss = self.loss(fake, real)
        return total_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, = batch

        z = self._sample_noise(imgs)

        # Lagrangians should be non-negative, so we project them on positive orthant
        with torch.no_grad():
            self.lagrangian_min.copy_(self.lagrangian_min.data.clamp(min=0))
            self.lagrangian_max.copy_(self.lagrangian_max.data.clamp(min=0))

        # minimize the distribution loss
        if optimizer_idx == 0:
            fake_min = self.generator_min(z, imgs)
            min_dist = self._calculate_loss(fake_min, imgs)
            g_min = self.lagrangian_min[0] * min_dist

            fake_max = self.generator_max(z, imgs)
            max_dist = self._calculate_loss(fake_max, imgs)

            g_max = self.lagrangian_max[0] * max_dist

            if self.monitor_estimand is not None:
                min_estimand, max_estimand = (
                    torch.mean(self.monitor_estimand(z, self.generator_min, self.device, data=imgs)),
                    torch.mean(self.monitor_estimand(z, self.generator_max, self.device, data=imgs))
                )
                self.log('min_' + str(self.monitor_estimand), min_estimand, on_step=False, on_epoch=True, prog_bar=True)
                self.log('max_' + str(self.monitor_estimand), max_estimand, on_step=False, on_epoch=True, prog_bar=True)

            self.log('distance_min_network', min_dist, on_step=False, on_epoch=True, prog_bar=True)
            self.log('distance_max_network', max_dist, on_step=False, on_epoch=True, prog_bar=True)
            self.log('alpha', self.alpha, on_step=False, on_epoch=True, prog_bar=True)
            return {'loss': g_min + g_max, 'distance_min_network': min_dist.detach(),
                    'distance_max_network': max_dist.detach()}

        # minimize the lagrange multipliers loss such that |distance| < alpha
        elif optimizer_idx == 1:
            loss = 0
            for mode, generator, lagrange in [
                ('min', self.generator_min, self.lagrangian_min[0]),
                ('max', self.generator_max, self.lagrangian_max[0])
            ]:
                fake = generator(z, imgs)
                distance = self._calculate_loss(fake, imgs)
                constraint = lagrange * (distance - self.alpha)
                loss += - constraint

                self.log('lagrangian' + mode, lagrange, on_step=False, on_epoch=True, prog_bar=True)
            return {'loss': loss}

        # minimize/maximize the estimand of interest
        elif optimizer_idx == 2 and not self.pre_train:
            estimand_min = torch.mean(self.estimand(z, self.generator_min, self.device, data=imgs))
            estimand_max = torch.mean(self.estimand(z, self.generator_max, self.device, data=imgs))

            self.log('estimand_min', estimand_min, on_step=False, on_epoch=True, prog_bar=True)
            self.log('estimand_max', estimand_max, on_step=False, on_epoch=True, prog_bar=True)

            return {'loss': estimand_min - estimand_max}

    def training_epoch_end(self, outputs):
        # find the best alpha, i.e., the smallest one
        metrics = self.trainer.callback_metrics
        try:
            dist_min = metrics['distance_min_network']
            dist_max = metrics['distance_max_network']
            self.best_dist_min = min(dist_min, self.best_dist_min)
            self.best_dist_max = min(dist_max, self.best_dist_max)
            self.log('best_distance', max(self.best_dist_max, self.best_dist_max), on_step=False, on_epoch=True,
                     prog_bar=True)
        except KeyError:
            warnings.warn('Could not find the distance metric in the callback metrics. ')

    def configure_gradient_clipping(
            self,
            optimizer: Optimizer,
            optimizer_idx: int,
            gradient_clip_val: Optional[Union[int, float]] = None,
            gradient_clip_algorithm: Optional[str] = None,
    ):
        if optimizer_idx == 2 and not self.pre_train:  # only clip the gradient of estimand
            self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val,
                                gradient_clip_algorithm=gradient_clip_algorithm)

    def configure_optimizers(self):

        lagrange_lr = 0.01 if self.pre_train else self.lagrange_lr  # fix the lagrange multiplier lr during pre-training
        opt_l = Adam([self.lagrangian_min, self.lagrangian_max], lr=lagrange_lr)

        if self.pre_train:
            opt_g = Adam(chain(self.generator_max.parameters(), self.generator_min.parameters()), lr=self.lr)

            # decrease the learning rate during pre-training for faster convergence
            lr_scheduler = ReduceLROnPlateau(opt_g, mode='min', factor=0.5,
                                             patience=20, threshold=0.0, threshold_mode='abs',
                                             cooldown=0, min_lr=0,
                                             eps=1e-08, verbose=False)

            return (
                {'optimizer': opt_g,
                 'lr_scheduler': {
                     'scheduler': lr_scheduler,
                     'interval': 'epoch',
                     'frequency': 1,
                     'monitor': 'distance_min_network',
                     'strict': True,
                 }
                 },
                {'optimizer': opt_l}
            )

        else:
            opt_g = Adam(chain(self.generator_max.parameters(), self.generator_min.parameters()), lr=self.lr)
            opt_estimand = Adam(chain(self.generator_max.parameters(), self.generator_min.parameters()), lr=self.lr / 2)
            return (
                {'optimizer': opt_g},
                {'optimizer': opt_l},
                {'optimizer': opt_estimand}
            )

    def on_epoch_start(self):
        # reset the optimizer after the pre-training phase
        if not self.pre_train and self.optimizer_flag:
            self.trainer.accelerator.setup_optimizers(self.trainer)
            self.optimizer_flag = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_hidden', type=int, default=64)
        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
        parser.add_argument('--lagrange_lr', type=float, default=1e-1, help="lagrange learning rate")

        return parser
