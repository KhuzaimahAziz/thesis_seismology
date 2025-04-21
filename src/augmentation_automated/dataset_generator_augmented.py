from omegaconf import DictConfig
import seisbench.data as sbd
import hydra
import seisbench.generate as sbg
import logging
import numpy as np
log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    train_data, dev_data = sbd.DummyDataset(), sbd.DummyDataset()
    
    train_generator = sbg.GenericGenerator(train_data)
    dev_generator = sbg.GenericGenerator(dev_data)
    sample = train_generator[np.random.randint(len(train_generator))]
    print(sample)
    def apply_augmentations(generator, config):
        for name, aug_cfg in config.items():
            if aug_cfg.get("enabled", False):
                aug = hydra.utils.instantiate(aug_cfg)
                generator.add_augmentations([aug])
                log.info(f"Applied {name} augmentation with config: {aug_cfg}")

    if cfg.train_augmentations:
        apply_augmentations(train_generator, cfg.augmentations)
    if cfg.val_augmentations:
        apply_augmentations(dev_generator, cfg.augmentations)

    log.info("Augmentation setup complete.")
    log.info("Sample data plot after augmentation")
    sample = train_generator[np.random.randint(len(train_generator))]
    print(sample)

if __name__ == "__main__":
    main()