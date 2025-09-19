import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import seisbench.data as sbd
import seisbench.generate as sbg
import torch
import typer
from omegaconf import DictConfig
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader

from .utils.model_utils import SeisBenchLit

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train_seisbench(cfg: DictConfig) -> None:
    log.info(f"Starting experiment: {cfg.experiment_name}")
    dataset_name = cfg.dataset.name
    log.info(f"Loading dataset: {dataset_name}")

    dataset_class = getattr(sbd, dataset_name, None)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data = dataset_class()
    log.info("Dataset loaded successfully.")

    if cfg.dataset.component_orders:
        data.component_order = cfg.dataset.component_orders
        log.info(f"Applied component order: {cfg.dataset.component_orders}")

    if cfg.dataset.dimension_orders:
        data.dimension_order = cfg.dataset.dimension_orders
        log.info(f"Applied dimension order: {cfg.dataset.dimension_orders}")

    if cfg.dataset.sampling_rates:
        data.sampling_rate = cfg.dataset.sampling_rates
        log.info(f"Applied sampling rate: {cfg.dataset.sampling_rates}")

    log.info("Setting up generators...")
    train_gen = sbg.GenericGenerator(data)
    val_gen = sbg.GenericGenerator(data)

    log.info("Setting up Lightning model...")
    pl_model = SeisBenchLit(
        lr=cfg.training.optimizer.params.lr,
        optimizer_params=cfg.training.optimizer.params,
    )

    log.info("Adding Augmentation...")
    train_gen.add_augmentations(pl_model.get_augmentations())
    val_gen.add_augmentations(pl_model.get_augmentations())
    log.info("Preparing data loaders...")
    train_loader = DataLoader(
        train_gen,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        worker_init_fn=worker_seeding,
    )

    val_loader = DataLoader(
        val_gen,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    log.info(f"Beginning training for {cfg.training.epochs} epochs...")
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
    )
    trainer.fit(pl_model, train_loader, val_loader)

    log.info("Training complete!")


@app.command()
def train(config_file: Path) -> None:
    def run_with_config() -> None:
        @hydra.main(
            config_path=str(config_file), config_name="config", version_base="1.3"
        )
        def inner(cfg: DictConfig) -> None:
            train_seisbench(cfg)

        inner()

    run_with_config()


def main() -> None:
    app()
