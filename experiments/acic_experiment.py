import os
import sys
import fire

import numpy as np
import torch
from geomloss import SamplesLoss
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../model'))
sys.path.append(os.path.join(dir_path, '../data'))

from utils import ToyDataModule, get_results, save_results
from common import MetricsCallback, StartEstimandOpt, LitProgressBar
from estimands import create_discrete_ate
from load_scm import gen_scm
from sinkhorn_gn import SinkhornGN


def acic_noisy(noise_level: int, n_samples: int, max_epochs: int, seed: int = None, gpus: int = 1, num_workers: int = 4,
               n_hidden: int = 64, n_layers: int = 3, lr: float = 0.001, lagrange_lr: float = 0.5):
    """Run ACIC experiment with noisy measurement.
    noise_level (int): 0-5, choose the noise from [no noise, N(0.1, 0.5), N(0.2, 0.5), N(0.3, 1), N(0.4, 1), N(0.5, 1)]
    """
    print(f'Running ACIC experiment with noise level {noise_level} and seed {seed}')
    seed_everything(seed, workers=True)
    batch_size = min(2048, n_samples)
    noise_means = [0.1, 0.2, 0.3, 0.4, 0.5]
    noise_stds = [0.5, 0.5, 1, 1, 1]

    # SCM and data
    scm = gen_scm('acic_2019')
    data = scm.generate(n_samples)
    tol = 1.1
    if noise_level > 0:
        noise = np.random.normal(noise_means[noise_level - 1], noise_stds[noise_level - 1], size=scm.dag.var_dims[0])
        noisy_cov = data[:, :scm.dag.var_dims[0]] + noise
        data = np.concatenate([noisy_cov, data[:, scm.dag.var_dims[0]:]], axis=1)
        tol = 1 + (noise_means[noise_level - 1] + 2 * noise_stds[noise_level - 1] / 10)

    dm = ToyDataModule(torch.tensor(data).float(), batch_size, num_workers)
    #
    # Estimand
    estimand = create_discrete_ate(scm.dag)

    # Model
    loss = SamplesLoss(loss="sinkhorn", p=1, blur=0.01, scaling=0.9, backend='tensorized', diameter=scm.diam)
    model = SinkhornGN(estimand=estimand, dag=scm.dag, loss=loss, n_hidden=n_hidden, n_layers=n_layers,
                       lr=lr, lagrange_lr=lagrange_lr, monitor_estimand=create_discrete_ate(scm.dag), tol=tol)

    # Trainer
    metrics_callback = MetricsCallback()
    lagrange_start = StartEstimandOpt(monitor='distance_min_network', min_delta=0.001,
                                      patience=max_epochs // 10, verbose=True, mode='min')

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/acic_noisy', every_n_epochs=max_epochs//5)
    prog_bar = LitProgressBar(refresh_rate=20)
    callbacks = [metrics_callback, prog_bar, lagrange_start, checkpoint_callback]
    trainer = Trainer(gpus=gpus, max_epochs=max_epochs, log_every_n_steps=5, callbacks=callbacks)

    trainer.fit(model, dm)
    distances, estimands = get_results(metrics_callback.metrics, str(estimand),
                                       model.alpha.to('cpu').item(), coeff=1.05)

    result = {'name': ['acic_noisy'], 'lower_estimand': [estimands[0]], 'upper_estimand': [estimands[1]],
              'seed': [seed], 'noise_level': noise_level, 'min_distance': [distances[0]],
              'max_distance': [distances[1]], 'true_estimand': scm.estimands['ATE']}
    save_results(result, 'results/acic_noisy.csv')


if __name__ == '__main__':
    fire.Fire(acic_noisy)
