import logging
from typing import Type

import hydra
import numpy as np
import pytorch_lightning as pl
import seisbench.data as sbd
import seisbench.generate as sbg
import torch
import typer
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader
from metrics.callbacks import EvaluationMetrics
from .utils.model_utils import SeisBenchLit, phase_dict

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def get_augmentation_configs(cfg):
    return [
        sbg.OneOf(
            [
                cfg.augmentations.window_default(
                    list(phase_dict.keys()),
                    samples_before=cfg.augmentations.window_default.samples_before,
                    windowlen=cfg.augmentations.window_default.windowlen,
                    selection=cfg.augmentations.window_default.selection,
                    strategy=cfg.augmentations.window_default.strategy,
                ),
                cfg.augmentations.null_augmentation(),
            ],
            probabilities=[2, 1],
        ),
        cfg.augmentations.random_window_default._target_(
            windowlen=cfg.augmentations.random_window_default.windowlen,
            strategy=cfg.augmentations.random_window_default.strategy,
        ),
        cfg.augmentations.normalize_default._target_(
            demean_axis=cfg.augmentations.normalize_default.demean_axis,
            amp_norm_axis=cfg.augmentations.normalize_default.amp_norm_axis,
            amp_norm_type=cfg.augmentations.normalize_default.amp_norm_type,
        ),
        cfg.augmentations.changeDtype._target_(
            dtype=cfg.augmentations.changeDtype.dtype,
        ),
        cfg.augmentations.prob_labeller_default._target_(
            label_columns=phase_dict,
            sigma=cfg.augmentations.prob_labeller_default.sigma,
            dim=cfg.augmentations.prob_labeller_default.dim,
        ),
    ]


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def train_seisbench(cfg):
    augmentations = [
        sbg.OneOf(
            [
                sbg.WindowAroundSample(
                    list(phase_dict.keys()),
                    samples_before=cfg.augmentations.window_default.samples_before,
                    windowlen=cfg.augmentations.window_default.windowlen,
                    selection=cfg.augmentations.window_default.selection,
                    strategy=cfg.augmentations.window_default.strategy,
                ),
                sbg.NullAugmentation(),
            ],
            probabilities=[2, 1],
        ),
        sbg.RandomWindow(
            windowlen=cfg.augmentations.random_window_default.windowlen,
            strategy=cfg.augmentations.random_window_default.strategy,
        ),
        sbg.Normalize(
            demean_axis=cfg.augmentations.normalize_default.demean_axis,
            amp_norm_axis=cfg.augmentations.normalize_default.amp_norm_axis,
            amp_norm_type=cfg.augmentations.normalize_default.amp_norm_type,
        ),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(
            label_columns=phase_dict,
            sigma=cfg.augmentations.prob_labeller_default.sigma,
            dim=cfg.augmentations.prob_labeller_default.dim,
            shape=cfg.augmentations.prob_labeller_default.shape,
        ),
    ]

    log.info(cfg)
    log.info(f"Starting experiment: {cfg.experiment_name}")
    dataset = cfg.dataset
    log.info(f"Loading dataset: {dataset.name}")

    try:
        DatasetClass: Type[sbd.BenchmarkDataset] | None = getattr(sbd, dataset.name)
    except AttributeError as exc:
        raise ValueError(f"Unknown dataset: {dataset.name}") from exc
    if not issubclass(DatasetClass, sbd.BenchmarkDataset):
        raise ValueError(f"Dataset {dataset.name} is not a BenchmarkDataset subclass")

    data = DatasetClass(
        component_order=dataset.component_orders,
        sampling_rate=dataset.sampling_rate,
    )
    log.info("Dataset loaded successfully.")
    train, dev, test = data.train_dev_test()

    log.info("Setting up generators...")
    train_gen = sbg.GenericGenerator(train)
    dev_gen = sbg.GenericGenerator(dev)
    test_gen = sbg.GenericGenerator(test)

    log.info("Setting up Lightning model...")
    pl_model = SeisBenchLit(
        lr=cfg.training.lr,
        sigma=cfg.augmentations.prob_labeller_default.sigma,
        pretrained_model_name=cfg.training.pretrained_model_name,
        transfer_learning =cfg.training.transfer_learning,
    )
    if cfg.training.transfer_learning == True:
            filename=f"best_model_{cfg.augmentations.prob_labeller_default.shape}_transfer_learning_{cfg.training.pretrained_model_name}"
    else:
            filename=f"best_model_{cfg.augmentations.prob_labeller_default.shape}_no_transfer_learning"
    log.info("Adding Augmentation...")
    train_gen.add_augmentations(augmentations)
    dev_gen.add_augmentations(augmentations)
    test_gen.add_augmentations(augmentations)

    log.info("Preparing data loaders...")

    train_loader = DataLoader(
        train_gen,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        worker_init_fn=worker_seeding,
    )

    test_loader = DataLoader(
        test_gen,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        worker_init_fn=worker_seeding,
    )

    dev_loader = DataLoader(
        dev_gen,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )


    mlf_logger = MLFlowLogger(experiment_name=cfg.experiment_name, log_model=True)

    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        monitor="val_loss",
        mode="min",
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=3)

    callbacks = [
        checkpoint_callback,
        early_stopping_callback,
        EvaluationMetrics(mlf_logger),
    ]

    log.info(f"Beginning training for {cfg.training.epochs} epochs...")

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        min_epochs=cfg.training.epochs,
        logger=mlf_logger,
        callbacks=callbacks,
        accelerator="gpu",
        log_every_n_steps=1,
        devices=2,
        strategy="ddp",
    )

    trainer.fit(pl_model, dev_loader, test_loader)
    mlf_logger.experiment.log_dict(
        run_id=mlf_logger.run_id,
        dictionary=OmegaConf.to_container(cfg, resolve=True),
        artifact_file=f"config_{cfg.training.epochs}.yaml",
    )

    log.info("Training complete!")


# @app.command()
def train():
    train_seisbench()


def main() -> None:
    train()


# if __name__ == "__main__":
#     train_seisbench()
