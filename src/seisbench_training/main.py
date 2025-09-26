import logging
from pathlib import Path
from typing import Type

import hydra
import pytorch_lightning as pl
import seisbench.data as sbd
import seisbench.generate as sbg
import torch
import typer
from omegaconf import DictConfig
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

from .utils.model_utils import SeisBenchLit
from .utils.model_utils import build_callbacks
app = typer.Typer()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def train_seisbench(cfg):

    print(cfg)
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
    val_gen = sbg.GenericGenerator(dev)

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
    callbacks = build_callbacks(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    trainer.fit(pl_model, train_loader, val_loader)

    log.info("Training complete!")


@app.command()
def train(config_file: Path) -> None:
    train_seisbench()

def main() -> None:
    app()


# if __name__ == "__main__":
#     train_seisbench()
