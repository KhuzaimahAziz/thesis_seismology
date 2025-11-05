from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import ndimage

if TYPE_CHECKING:
    from matplotlib.figure import Figure

ComponentOrder = Literal["NPS", "PSN"]

ORDER_MAP: dict[ComponentOrder, tuple[int, int]] = {
    "NPS": (1, 2),
    "PSN": (0, 1),
}


def calculate_pick_differences(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    order: ComponentOrder = "PSN",
    min_pick_height: float = 0.3,
    window_width: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get predicted and labeled pick sample indices for P and S waves.

    Args:
        predictions (torch.Tensor): Predicted label probabilities of shape
            (batch, components, samples).
        labels (torch.Tensor): True label probabilities of shape
            (batch, components, samples).
        order (ComponentOrder, optional): Order of components in the predictions/labels.
            Defaults to "PSN".
        min_pick_height (float, optional): Minimum probability height to consider
            a pick valid. Defaults to 0.3.
        window_width (int, optional): Width of the window to expand the label picks.
            Defaults to 500.
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Predicted and
            labeled pick sample indices for P and S waves.
    """
    label_max, label_pick_sample = labels.max(dim=2)

    # Mask out predictions outside of labeled pick regions
    if window_width:
        epsilon = 1e-10
        labels_numpy = labels.detach().numpy()
        labels_numpy[labels_numpy < epsilon] = 0.0

        window = np.ones(window_width)
        # Expand mask to include neighboring samples
        mask = ndimage.convolve1d(labels_numpy, window, axis=2, mode="constant")
        mask[mask < epsilon] = 0.0
        predictions_masked = predictions * torch.from_numpy(mask.astype(bool))
    else:
        predictions_masked = predictions

    # predictions_masked = predictions
    prediction_max, prediction_pick_sample = predictions_masked.max(dim=2)
    p_idx, s_idx = ORDER_MAP[order]

    # Some samples may not have a pick for P or S wave
    p_mask = label_max[:, p_idx].to(bool)
    s_mask = label_max[:, s_idx].to(bool)

    p_mask &= prediction_max[:, p_idx] >= min_pick_height
    s_mask &= prediction_max[:, s_idx] >= min_pick_height

    p_predicted_sample = prediction_pick_sample[:, p_idx][p_mask]
    p_probability = prediction_max[:, p_idx][p_mask]
    p_labeled_sample = label_pick_sample[:, p_idx][p_mask]

    s_predicted_sample = prediction_pick_sample[:, s_idx][s_mask]
    s_probability = prediction_max[:, s_idx][s_mask]
    s_labeled_sample = label_pick_sample[:, s_idx][s_mask]

    return (
        p_predicted_sample.detach().numpy(),
        p_labeled_sample.detach().numpy(),
        s_predicted_sample.detach().numpy(),
        s_labeled_sample.detach().numpy(),
    )


def plot_histogram(
    predicted: np.ndarray,
    label: np.ndarray,
    time_window_limit: float = 1.0,
    sampling_rate: float = 100.0,
    title: str = "",
    show_figure: bool = False,
) -> Figure | None:
    fig = plt.figure()
    ax = fig.gca()
    offsets = (predicted - label) / sampling_rate  # in seconds
    ax.hist(
        offsets,
        bins=100,
        range=(-time_window_limit, time_window_limit),
        alpha=0.7,
        color="blue",
    )
    ax.set_xlabel("Pick time difference (seconds)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

    stats = compute_evaluation_metrics(
        predicted=predicted,
        label=label,
        sampling_rate=sampling_rate,
    )

    fraction_outside_window = (
        np.sum(np.abs(offsets) > time_window_limit) / offsets.size * 100.0
    )  # percentage

    ax.axvline(
        stats.mean_difference,
        color="green",
        linestyle="dashed",
        label="Mean difference",
    )
    ax.axvline(
        stats.median_difference,
        color="orange",
        linestyle="dashed",
        label="Median difference",
    )
    ax.set_title(title)

    ax.text(
        0.02,
        0.98,
        f"Median difference: {stats.median_difference:.3f} s\n"
        f"Mean difference: {stats.mean_difference:.3f} s\n"
        f"MAE: {stats.mean_abs_error:.3f} s\n"
        f"RMS error: {stats.rms_error:.3f} s\n"
        f"Total picks: {offsets.size}\n"
        f"Outside ±{time_window_limit}s: {fraction_outside_window:.0f}%",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize="small",
    )
    ax.legend(loc="upper right")
    ax.set_xlim(-time_window_limit, time_window_limit)

    if show_figure:
        plt.show()
    return fig


class EvaluationMetrics(NamedTuple):
    mean_difference: float
    median_difference: float
    mean_abs_error: float
    rms_error: float


def compute_evaluation_metrics(
    predicted: np.ndarray,
    label: np.ndarray,
    sampling_rate: float = 100.0,
) -> EvaluationMetrics:
    """Compute evaluation metrics for pick time differences.

    Args:
        predicted (np.ndarray): Predicted pick sample indices.
        label (np.ndarray): Labeled pick sample indices.
        sampling_rate (float, optional): Sampling rate of the data. Defaults to 100.0.

    Returns:
        EvaluationMetrics: Named tuple containing evaluation metrics.
    """
    offsets = (predicted - label) / sampling_rate  # in seconds

    return EvaluationMetrics(
        mean_difference=float(np.mean(offsets)),
        median_difference=float(np.median(offsets)),
        mean_abs_error=float(np.mean(np.abs(offsets))),
        rms_error=float(np.sqrt(np.mean(offsets**2))),
    )
