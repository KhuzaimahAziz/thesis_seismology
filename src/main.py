import hydra
from omegaconf import DictConfig
import logging
import seisbench.data as sbd
import seisbench.generate as sbg

from torch.utils.data import DataLoader
from utils.model_utils import setup_model, setup_optimizer, train_loop, test_loop


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)
 
def apply_augmentations(generator, config):
    for name, aug_cfg in config.items():
        if aug_cfg.get("enabled", False):
            aug = hydra.utils.instantiate(aug_cfg)
            generator.add_augmentations([aug])
            log.info(f"Applied augmentation: {name} with config: {aug_cfg}")

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log.info(f"🚀 Starting experiment: {cfg.experiment_name}")
    dataset_name = cfg.dataset.name
    log.info(f"📦 Loading dataset: {dataset_name}")
    
    dataset_class = getattr(sbd, dataset_name, None)
    if dataset_class is None:
        raise ValueError(f"❌ Unknown dataset: {dataset_name}")

    data = dataset_class()
    log.info("✅ Dataset loaded successfully.")

    if cfg.dataset.component_orders:
        data.component_order = cfg.dataset.component_orders
        log.info(f"🔀 Applied component order: {cfg.dataset.component_orders}")
    
    if cfg.dataset.dimension_orders:
        data.dimension_order = cfg.dataset.dimension_orders
        log.info(f"📐 Applied dimension order: {cfg.dataset.dimension_orders}")

    if cfg.dataset.sampling_rates:
        data.sampling_rate = cfg.dataset.sampling_rates
        log.info(f"🎚️ Applied sampling rate: {cfg.dataset.sampling_rates}")

    log.info("📊 Setting up generators...")
    train_gen = sbg.GenericGenerator(data)
    val_gen = sbg.GenericGenerator(data)

    if cfg.train_augmentations:
        log.info("🧪 Applying training augmentations...")
        apply_augmentations(train_gen, cfg.augmentations)
    
    if cfg.val_augmentations:
        log.info("🧪 Applying validation augmentations...")
        apply_augmentations(val_gen, cfg.augmentations)

    log.info("🧠 Setting up model...")
    model = setup_model(cfg)

    log.info("⚙️ Setting up optimizer...")
    optimizer = setup_optimizer(model, cfg.training.optimizer)

    log.info("📦 Preparing data loaders...")
    train_loader = DataLoader(
        train_gen,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )

    val_loader = DataLoader(
        val_gen,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    log.info(f"🏋️ Beginning training for {cfg.training.epochs} epochs")
    for epoch in range(cfg.training.epochs):
        log.info(f"📁 Epoch {epoch + 1}/{cfg.training.epochs}")
        train_loop(train_loader, model, optimizer)
        test_loop(val_loader, model)

    log.info("✅ Training complete!")

if __name__ == "__main__":
    main()
