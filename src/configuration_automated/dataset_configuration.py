import hydra
from omegaconf import DictConfig
import seisbench.data as sbd
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log.info(f"Running experiment: {cfg.experiment_name}")

    dataset_name = cfg.dataset.name
    log.info(f"Using dataset: {dataset_name}")

    dataset_class = getattr(sbd, dataset_name, None)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data = dataset_class()

    if cfg.dataset.component_orders:
        component_order = cfg.dataset.component_orders  
        log.info(f"Applying component order: {component_order}")
        data.component_order = component_order
        waveforms = data.get_waveforms(0)
        log.info(f"Component Order: {component_order} | Sample Data:\n{waveforms[:, :5]}")
    else:
        log.info("No component orders provided. Skipping this step.")

    if cfg.dataset.dimension_orders:
        dimension_order = cfg.dataset.dimension_orders  
        log.info(f"Applying dimension order: {dimension_order}")
        data.dimension_order = dimension_order
        waveforms = data.get_waveforms([3, 20, 45, 70])
        log.info(f"Dimension Order: {dimension_order} | Waveforms Shape: {waveforms.shape}")
    else:
        log.info("No dimension orders provided. Skipping this step.")

    if cfg.dataset.sampling_rates:
        sampling_rate = cfg.dataset.sampling_rates  #
        log.info(f"Applying sampling rate: {sampling_rate} Hz")
        data.sampling_rate = sampling_rate
        waveforms = data.get_waveforms(0)
        log.info(f"Sampling Rate: {sampling_rate} Hz | Waveforms Shape: {waveforms.shape}")
    else:
        log.info("No sampling rates provided. Skipping this step.")
    final_waveform = data.get_waveforms(0)
    log.info(f"Final Waveform after all configurations:\n{final_waveform[:, :5]}")

if __name__ == "__main__":
    main()