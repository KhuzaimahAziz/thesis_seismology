import logging
from pathlib import Path
from typing import Type

import os
import hydra
import pytorch_lightning as pl
import torch
import typer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


# The following imports require the seisbench package to be installed and available in your environment.
import seisbench.data as sbd
import seisbench.generate as sbg
from seisbench.util import worker_seeding

from .utils.model_utils import SeisBenchLit
from .utils.model_utils import build_callbacks

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


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

    csv_logger = CSVLogger("weights", cfg.experiment_name)
    # csv_logger.log_hyperparams(cfg)
    loggers = [csv_logger]
    # tensorboard_logger = TensorBoardLogger("weights", cfg.experiment_name)
    # tensorboard_logger.log_hyperparams(cfg) 
    loggers = [csv_logger]
    log.info(f"Beginning training for {cfg.training.epochs} epochs...")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, filename="{epoch}-{step}", monitor="val_loss", mode="min"
    ) 
    callbacks = [checkpoint_callback]
    root_dir = os.path.join("weights")


    trainer = pl.Trainer(
        default_root_dir=root_dir,
        max_epochs=cfg.training.epochs,
        min_epochs=cfg.training.epochs,
        logger=loggers,
        callbacks=callbacks,
        accelerator="gpu",
        log_every_n_steps=1,
        devices= 2,
        strategy="ddp",
        num_nodes = 1
    )
    trainer.fit(pl_model, val_loader, val_loader)

    log.info("Training complete!")


@app.command()
def train(config_file: Path) -> None:
    train_seisbench()

def main() -> None:
    app()


# if __name__ == "__main__":
#     train_seisbench()
