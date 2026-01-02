from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics

if TYPE_CHECKING:
    from matplotlib.figure import Figure

ComponentOrder = Literal["NPS", "PSN"]

ORDER_MAP: dict[ComponentOrder, tuple[int, int]] = {
    "NPS": (1, 2),
    "PSN": (0, 1),
}


class PickStats(NamedTuple):
    predicted_samples: np.ndarray
    labeled_samples: np.ndarray
    predicted_certainty: np.ndarray
    noise_max: np.ndarray

    @property
    def n_samples(self) -> int:
        return self.predicted_samples.size

    @property
    def offset_samples(self) -> np.ndarray:
        return self.predicted_samples - self.labeled_samples

    @property
    def mean_difference(self) -> float:
        return float(np.nanmean(self.offset_samples))

    @property
    def median_difference(self) -> float:
        return float(np.nanmedian(self.offset_samples))

    @property
    def mean_abs_error(self) -> float:
        return float(np.nanmean(np.abs(self.offset_samples)))

    @property
    def rms_error(self) -> float:
        return float(np.sqrt(np.nanmean(self.offset_samples**2)))

    @property
    def true_labels(self) -> np.ndarray:
        pick_examples = np.ones((~np.isnan(self.labeled_samples)).sum(), dtype=bool)
        noise_example = np.zeros(self.noise_max.size, dtype=bool)
        return np.concatenate([pick_examples, noise_example])

    @property
    def predicted_scores(self) -> np.ndarray:
        pick_score = self.predicted_certainty[~np.isnan(self.labeled_samples)]
        noise_score = self.noise_max
        return np.concatenate([pick_score, noise_score])

    @property
    def roc_curve(self) -> tuple[np.ndarray, np.ndarray]:
        fpr, tpr, _ = metrics.roc_curve(self.true_labels, self.predicted_scores)
        return fpr, tpr

    @property
    def auc(self) -> float:
        fpr, tpr = self.roc_curve
        return metrics.auc(fpr, tpr)


class DetectionMetrics(NamedTuple):
    threshold: float
    precision: float
    recall: float
    f1_score: float


def calculate_pick_differences(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    label_order: ComponentOrder = "PSN",
    window_width: int = 500,
    edge_mask: int = 100,
) -> dict[str, PickStats]:
    """Get predicted and labeled pick sample indices for P and S waves.

    Args:
        predictions (torch.Tensor): Predicted label probabilities of shape
            (batch, components, samples).
        labels (torch.Tensor): True label probabilities of shape
            (batch, components, samples).
        order (ComponentOrder, optional): Order of components in the predictions/labels.
            Defaults to "PSN".
        window_width (int, optional): Width of the window to expand the label picks.
            Defaults to 500.
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray ]: Predicted and
            labeled pick sample indices for P and S waves, Probabilities of PSN and Binary True Labels of PSN.
    """
    label_max, label_pick_sample = labels.max(dim=2)

    # Mask out predictions outside of labeled pick regions
    if window_width:
        # TODO: Use signal.argrelmax to find multiple picks if needed
        _, label_pick_sample = labels.max(dim=2)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        # Set all pick sampple locations to True
        for batch_idx in range(labels.shape[0]):
            for comp_idx in range(labels.shape[1]):
                pick_sample = label_pick_sample[batch_idx, comp_idx]
                window_start = max(pick_sample - window_width, 0)
                window_end = min(pick_sample + window_width, labels.shape[2])
                mask[
                    batch_idx,
                    comp_idx,
                    window_start:window_end,
                ] = True

        predictions_masked = predictions * mask
    else:
        predictions_masked = predictions

    # predictions_masked = predictions
    # This assumes there is only one pick per component per trace
    # TODO: Use signal.argrelmax to find multiple picks if needed
    prediction_max, prediction_pick_sample = predictions_masked.max(dim=2)
    p_idx, s_idx = ORDER_MAP[label_order]

    p_mask = label_max[:, p_idx].to(bool)
    s_mask = label_max[:, s_idx].to(bool)

    p_mask_noise = ~p_mask
    s_mask_noise = ~s_mask
    p_mask_noise[:edge_mask] = False
    p_mask_noise[-edge_mask - 1 :] = False
    s_mask_noise[:edge_mask] = False
    s_mask_noise[-edge_mask - 1 :] = False

    p_noise_max, _ = predictions[:, p_idx][p_mask_noise].max(dim=-1)
    s_noise_max, _ = predictions[:, s_idx][s_mask_noise].max(dim=-1)

    # p_mask &= prediction_max[:, p_idx] >= min_pick_height
    # s_mask &= prediction_max[:, s_idx] >= min_pick_height

    p_predicted_sample = prediction_pick_sample[:, p_idx][p_mask].type(torch.float32)
    p_labeled_sample = label_pick_sample[:, p_idx][p_mask].type(torch.float32)
    s_predicted_sample = prediction_pick_sample[:, s_idx][s_mask].type(torch.float32)
    s_labeled_sample = label_pick_sample[:, s_idx][s_mask].type(torch.float32)

    # If there is no pick or at the edges, set to NaN
    s_labeled_sample[s_labeled_sample == 0.0] = torch.nan
    p_labeled_sample[p_labeled_sample == 0.0] = torch.nan
    s_labeled_sample[s_labeled_sample == labels.shape[2] - 1] = torch.nan
    p_labeled_sample[p_labeled_sample == labels.shape[2] - 1] = torch.nan

    p_prob = prediction_max[:, p_idx][p_mask]
    s_prob = prediction_max[:, s_idx][s_mask]

    return {
        "P": PickStats(
            predicted_samples=p_predicted_sample.detach().numpy(),
            labeled_samples=p_labeled_sample.detach().numpy(),
            predicted_certainty=p_prob.detach().numpy(),
            noise_max=p_noise_max.detach().numpy(),
        ),
        "S": PickStats(
            predicted_samples=s_predicted_sample.detach().numpy(),
            labeled_samples=s_labeled_sample.detach().numpy(),
            predicted_certainty=s_prob.detach().numpy(),
            noise_max=s_noise_max.detach().numpy(),
        ),
    }


def plot_histogram(
    stats: PickStats,
    time_window_limit: float = 1.0,
    sampling_rate: float = 100.0,
    title: str = "",
    show_figure: bool = False,
) -> Figure | None:
    fig = plt.figure()
    ax = fig.gca()
    offsets = stats.offset_samples / sampling_rate
    offsets = offsets[~np.isnan(offsets)]
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

    # percentage
    fraction_outside_window = (
        np.sum(np.abs(offsets) > time_window_limit) / offsets.size * 100.0
    )

    ax.axvline(
        stats.mean_difference / sampling_rate,
        color="green",
        linestyle="dashed",
        label="Mean difference",
    )
    ax.axvline(
        stats.median_difference / sampling_rate,
        color="orange",
        linestyle="dashed",
        label="Median difference",
    )
    ax.set_title(title)

    sr = sampling_rate
    ax.text(
        0.02,
        0.98,
        f"Median difference: {stats.median_difference / sr:.3f} s\n"
        f"Mean difference: {stats.mean_difference / sr:.3f} s\n"
        f"MAE: {stats.mean_abs_error / sr:.3f} s\n"
        f"RMS error: {stats.rms_error / sr:.3f} s\n"
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


def calculate_precision_recall_f1(
    stats: PickStats,
    thresholds: np.ndarray | None = None,
) -> list[DetectionMetrics]:
    """Get offset of P and S waves and their corresponding predicted probabilities.

    Args:
        offset (torch.Tensor): offset value of P and S waves.
        final_prob (torch.Tensor): Predicted Masked probabilities around window length.
        time_tolerance (float): Time tolerance mask for offset. Defaults to 0.1.
    Returns:
        dict: Dictionary containing precision, recall, and f1 scores at different thresholds.

    """
    thresholds = np.linspace(0.0, 1.0, 41)[1:] if thresholds is None else thresholds
    results = []

    for thres in thresholds:
        pred_mask = stats.predicted_scores >= thres
        TP = pred_mask[stats.true_labels].sum()
        FP = pred_mask[~stats.true_labels].sum()
        FN = (~pred_mask[stats.true_labels]).sum()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (precision + recall)

        res = DetectionMetrics(
            threshold=thres,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        )

        results.append(res)

    return results


def get_f1_optimal_metrics(
    detection_metrics: list[DetectionMetrics],
) -> DetectionMetrics:
    """Get the detection metrics at the optimal F1 score.

    Args:
        detection_metrics (list[DetectionMetrics]): List of DetectionMetrics.
    Returns:
        DetectionMetrics: DetectionMetrics at the optimal F1 score.
    """
    f1_scores = [d.f1_score for d in detection_metrics]
    max_index = np.nanargmax(f1_scores)
    return detection_metrics[max_index]


def plot_precision_recall_f1(detection_metrics: list[DetectionMetrics], title: str):
    """Takes the Metrics dict containing Precision, Recall and F1_score.

    Args:
        metrics_dict (dict): dict containing Precision, Recall and F1_score.
    Returns:
        fig: Matplotlib figure object for further use.
    """
    thresholds = [d.threshold for d in detection_metrics]
    precision = [d.precision for d in detection_metrics]
    recall = [d.recall for d in detection_metrics]
    f1_score = [d.f1_score for d in detection_metrics]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        thresholds,
        precision,
        label="Precision",
        linewidth=2,
        color="blue",
    )
    ax.plot(
        thresholds,
        recall,
        label="Recall",
        linewidth=2,
        color="orange",
    )
    ax.plot(
        thresholds,
        f1_score,
        label="F1 Score",
        linewidth=2,
        color="green",
    )

    ax.grid(alpha=0.3)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric Score")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()

    return fig


def plot_roc_curve(
    stats: PickStats,
    title: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    fpr, tpr = stats.roc_curve
    auc = stats.auc

    ax.plot(fpr, tpr, linewidth=3, label=f"AUC = {auc:.3f}")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.grid(alpha=0.3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")

    fig.tight_layout()
    return fig
