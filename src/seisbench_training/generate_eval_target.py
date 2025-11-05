from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm

# Try to import seisbench.data and provide a clear error if it's missing.
try:
    import seisbench.data as sbd
except Exception as e:
    raise ImportError(
        "seisbench.data could not be imported. Please install seisbench (e.g. `pip install seisbench`) "
        "or ensure it is available on PYTHONPATH."
    ) from e

from .utils.model_utils import phase_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    dataset_args = {
        "sampling_rate": cfg.dataset.sampling_rate,
        "component_order": cfg.dataset.component_orders,
    }

    try:
        dataset = sbd.__getattribute__(cfg.dataset.name)(**dataset_args)
    except AttributeError:
        dataset = sbd.WaveformDataset(cfg.dataset.name, **dataset_args)

    output = Path("output/")
    output.mkdir(parents=True, exist_ok=False)

    if "split" in dataset.metadata.columns:
        dataset.filter(dataset["split"].isin(["dev", "test"]), inplace=True)

    dataset.preload_waveforms(pbar=True)

    generate_eval_labels(dataset, output, cfg.dataset.sampling_rate)


def generate_eval_labels(
    dataset: sbd.BenchmarkDataset,
    output: Path,
    sampling_rate: int,
) -> None:
    np.random.seed(42)
    windowlen = 10 * sampling_rate  # 30 s windows
    labels = []

    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        waveforms, metadata = dataset.get_sample(idx)

        trace_split = metadata.get("split", "")

        def checkphase(metadata, phase, npts) -> bool:
            return (
                phase in metadata
                and not np.isnan(metadata[phase])
                and 0 <= metadata[phase] < npts
            )

        # Example entry: (1031, "P", "Pg")
        arrivals = sorted(
            [
                (metadata[phase], phase_label, phase.split("_")[1])
                for phase, phase_label in phase_dict.items()
                if checkphase(metadata, phase, waveforms.shape[-1])
            ]
        )

        if len(arrivals) == 0:
            # Trace has no arrivals
            continue

        for i, (onset, phase, full_phase) in enumerate(arrivals):
            if i == 0:
                onset_before = 0
            else:
                onset_before = int(arrivals[i - 1][0]) + int(
                    0.5 * sampling_rate
                )  # 0.5 s minimum spacing

            if i == len(arrivals) - 1:
                onset_after = np.inf
            else:
                onset_after = int(arrivals[i + 1][0]) - int(
                    0.5 * sampling_rate
                )  # 0.5 s minimum spacing

            if (
                onset_after - onset_before < windowlen
                or onset_before > onset
                or onset_after < onset
            ):
                # Impossible to isolate pick
                continue

            else:
                onset_after = min(onset_after, waveforms.shape[-1])
                # Shift everything to a "virtual" start at onset_before
                start_sample, end_sample = select_window_containing(
                    onset_after - onset_before,
                    windowlen=windowlen,
                    containing=onset - onset_before,
                    bounds=(50, 50),
                )
                start_sample += onset_before
                end_sample += onset_before
                if end_sample - start_sample <= windowlen:
                    sample = {
                        "trace_name": metadata["trace_name"],
                        "trace_idx": idx,
                        "trace_split": trace_split,
                        "sampling_rate": sampling_rate,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "phase_label": phase,
                        "full_phase_label": full_phase,
                        "phase_onset": onset,
                    }

                    labels += [sample]

    labels = pd.DataFrame(labels)
    diff = labels["end_sample"] - labels["start_sample"]
    labels = labels[diff > 100]
    labels.to_csv(output / "generated_labels.csv", index=False)


def select_window_containing(
    npts: int,
    windowlen: int,
    containing: int | None = None,
    bounds: tuple[int, int] = (100, 100),
):
    """
    Selects a window from a larger trace.

    :param npts: Number of points of the full trace
    :param windowlen: Desired windowlen
    :param containing: Sample number that should be contained. If None, any window within the trace is valid.
    :param bounds: The containing sample may not be in the first/last samples indicated here.
    :return: Start sample, end_sample
    """
    if npts <= windowlen:
        # If npts is smaller than the window length, always return the full window
        return 0, npts

    else:
        if containing is None:
            start_sample = np.random.randint(0, npts - windowlen + 1)
            return start_sample, start_sample + windowlen

        else:
            earliest_start = max(0, containing - windowlen + bounds[1])
            latest_start = min(npts - windowlen, containing - bounds[0])
            if latest_start <= earliest_start:
                # Again, return full window
                return 0, npts

            else:
                start_sample = np.random.randint(earliest_start, latest_start + 1)
                return start_sample, start_sample + windowlen
