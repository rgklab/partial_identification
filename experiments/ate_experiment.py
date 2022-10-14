import os
import sys
import fire

import torch
from geomloss import SamplesLoss
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../model'))
sys.path.append(os.path.join(dir_path, '../data'))

from utils import ToyDataModule, get_results, save_results
from common import MetricsCallback, StartEstimandOpt, LitProgressBar
from estimands import create_uatd_gauss, create_ate
from load_scm import gen_scm
from sinkhorn_gn import SinkhornGN


def ATE(setting: str, d0: float, d1: float, n_samples: int, max_epochs: int, seed: int, gpus: int = 1, num_workers: int = 4,
        n_hidden: int = 64, n_layers: int = 3, lr: float = 0.001, lagrange_lr: float = 0.5):
    """Run the Average Treatment Effect experiment.
    Args:
        setting (str): Data-generating process. See ``data.load_scm.gen_scm`` function for options.
        d0 (float): Interval (d0, d1) to calculate the ATE, i.e., ATE = E[Y|do(T=d1)] - E[Y|do(T=d0)].
        d1 (float): Interval (d0, d1) to calculate the ATE, i.e., ATE = E[Y|do(T=d1)] - E[Y|do(T=d0)].
    """
    print(f'Running the ATE experiment with setting {setting} and seed {seed}')
    seed_everything(seed, workers=True)
    batch_size = min(2048, n_samples)

    interval = (d0, d1)

    # SCM and data
    scm = gen_scm(setting)
    data = scm.generate(n_samples)
    tol = 1.1

    dm = ToyDataModule(torch.tensor(data).float(), batch_size, num_workers)
    #
    # Estimand
    estimand = create_uatd_gauss(scm.dag, interval=interval, delta=0.1, std=0.5)  # choose std wisely

    # Model
    monitor = create_ate(scm.dag, interval)
    loss = SamplesLoss(loss="sinkhorn", p=1, blur=0.01, scaling=0.9, backend='tensorized', diameter=scm.diam)
    model = SinkhornGN(estimand=estimand, dag=scm.dag, loss=loss, n_hidden=n_hidden, n_layers=n_layers,
                       lr=lr, lagrange_lr=lagrange_lr, monitor_estimand=monitor, tol=tol)

    # Trainer
    metrics_callback = MetricsCallback()
    lagrange_start = StartEstimandOpt(monitor='distance_min_network', min_delta=0.001,
                                      patience=max_epochs // 10, verbose=True, mode='min')

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/ATD', every_n_epochs=max_epochs//5)
    prog_bar = LitProgressBar(refresh_rate=20)
    callbacks = [metrics_callback, prog_bar, lagrange_start, checkpoint_callback]
    trainer = Trainer(gpus=gpus, max_epochs=max_epochs, log_every_n_steps=5, callbacks=callbacks, gradient_clip_val=0.5)

    trainer.fit(model, dm)
    try:
        alpha = model.alpha.to('cpu').item()
    except AttributeError:
        alpha = model.alpha
    distances, estimands = get_results(metrics_callback.metrics, str(monitor),
                                       alpha, coeff=1.05)

    result = {'name': ['ATE'], 'setting': [setting], 'lower_estimand': [estimands[0]], 'upper_estimand': [estimands[1]],
              'seed': [seed], 'min_distance': [distances[0]],
              'max_distance': [distances[1]]}
    save_results(result, 'results/ATE.csv')


if __name__ == '__main__':
    fire.Fire(ATE)
