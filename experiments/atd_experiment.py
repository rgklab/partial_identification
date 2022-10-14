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
from estimands import create_atd
from load_scm import gen_scm
from sinkhorn_gn import SinkhornGN


def ATD(setting: str, n_samples: int, max_epochs: int, seed: int, gpus: int = 1, num_workers: int = 4,
        n_hidden: int = 64, n_layers: int = 3, lr: float = 0.001, lagrange_lr: float = 0.5):
    """Run the Average Treatment Derivative experiment.
    Args:
        setting (str): Data-generating process. See ``data.load_scm.gen_scm`` function for options.
    """
    print(f'Running the ATD experiment with setting {setting} and seed {seed}')
    seed_everything(seed, workers=True)
    batch_size = min(2048, n_samples)

    # SCM and data
    scm = gen_scm(setting)
    data = scm.generate(n_samples)
    tol = 1.1

    dm = ToyDataModule(torch.tensor(data).float(), batch_size, num_workers)
    #
    # Estimand
    estimand = create_atd(scm.dag, delta=0.1)

    # Model
    loss = SamplesLoss(loss="sinkhorn", p=1, blur=0.01, scaling=0.9, backend='tensorized', diameter=scm.diam)
    model = SinkhornGN(estimand=estimand, dag=scm.dag, loss=loss, n_hidden=n_hidden, n_layers=n_layers,
                       lr=lr, lagrange_lr=lagrange_lr, monitor_estimand=create_atd(scm.dag, delta=0.1), tol=tol)

    # Trainer
    metrics_callback = MetricsCallback()
    lagrange_start = StartEstimandOpt(monitor='distance_min_network', min_delta=0.001,
                                      patience=max_epochs // 10, verbose=True, mode='min')

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/ATD', every_n_epochs=max_epochs//5)
    prog_bar = LitProgressBar(refresh_rate=20)
    callbacks = [metrics_callback, prog_bar, lagrange_start, checkpoint_callback]
    trainer = Trainer(gpus=gpus, max_epochs=max_epochs, log_every_n_steps=5, callbacks=callbacks)

    trainer.fit(model, dm)
    distances, estimands = get_results(metrics_callback.metrics, str(estimand),
                                       model.alpha.to('cpu').item(), coeff=1.05)

    result = {'name': ['ATD'], 'setting': [setting], 'lower_estimand': [estimands[0]], 'upper_estimand': [estimands[1]],
              'seed': [seed], 'min_distance': [distances[0]],
              'max_distance': [distances[1]], 'true_estimand': scm.estimands['ATD']}
    save_results(result, 'results/ATD.csv')


if __name__ == '__main__':
    fire.Fire(ATD)
