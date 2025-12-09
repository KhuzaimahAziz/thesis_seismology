from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import ndimage

from sklearn import metrics

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
            a pick valid. Defaults to 0.2.
        window_width (int, optional): Width of the window to expand the label picks.
            Defaults to 500.
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray ]: Predicted and
            labeled pick sample indices for P and S waves, Probabilities of PSN and Binary True Labels of PSN.
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

    p_mask = label_max[:, p_idx].to(bool)
    s_mask = label_max[:, s_idx].to(bool)

    p_mask &= prediction_max[:, p_idx] >= min_pick_height
    s_mask &= prediction_max[:, s_idx] >= min_pick_height

    p_predicted_sample = prediction_pick_sample[:, p_idx][p_mask]
    p_labeled_sample = label_pick_sample[:, p_idx][p_mask]

    s_predicted_sample = prediction_pick_sample[:, s_idx][s_mask]
    s_labeled_sample = label_pick_sample[:, s_idx][s_mask]

    p_prob = prediction_max[:, p_idx][p_mask]
    s_prob = prediction_max[:, s_idx][s_mask]

    label_pick_class = label_max.argmax(dim=1) 
    p_labels_binary = label_pick_class[p_mask]
    s_labels_binary = label_pick_class[s_mask]  

    
    return (
        p_predicted_sample.detach().numpy(),
        p_labeled_sample.detach().numpy(),
        s_predicted_sample.detach().numpy(),
        s_labeled_sample.detach().numpy(),
        p_prob.detach().numpy(),
        s_prob.detach().numpy(),
        p_labels_binary.detach().numpy(),
        s_labels_binary.detach().numpy(),   
    )


def plot_histogram(
    predicted: np.ndarray,
    label: np.ndarray,
    stats: tuple,
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

def calculate_precision_recall_f1(offset: torch.Tensor, final_prob: torch.Tensor, time_tolerance=0.1) -> dict:
    """Get offset of P and S waves and their corresponding predicted probabilities.

    Args:
        offset (torch.Tensor): offset value of P and S waves.
        final_prob (torch.Tensor): Predicted Masked probabilities around window length.
        time_tolerance (float): Time tolerance mask for offset. Defaults to 0.1.
    Returns:
        dict: Dictionary containing precision, recall, and f1 scores at different thresholds.
        
    """
    thresholds = np.linspace(0, 1, 6)
    thresholds = np.round(thresholds, 2)
    results = {}

    for t in thresholds:
        pred_mask = (final_prob >= t)
        TP = (pred_mask & (offset <= time_tolerance)).sum().item()
        FP = (pred_mask & (offset > time_tolerance)).sum().item()
        FN = (~pred_mask).sum().item()

        precision = TP / (TP + FP + 1e-9)
        recall    = TP / (TP + FN + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        results[t] = {"precision": precision, "recall": recall, "f1": f1}

    return results

def plot_precision_recall_f1(metrics_dict: dict, title):
    """Takes the Metrics dict containing Precision, Recall and F1_score.

    Args:
        metrics_dict (dict): dict containing Precision, Recall and F1_score.
    Returns:
        fig: Matplotlib figure object for further use.
    """
    thresholds = list(metrics_dict.keys())
    precision_list = [metrics_dict[t]["precision"] for t in thresholds]
    recall_list    = [metrics_dict[t]["recall"] for t in thresholds]
    f1_list        = [metrics_dict[t]["f1"] for t in thresholds]

    fig, ax = plt.subplots(figsize=(8,6))

    # Get values at threshold 0.60 for legend
    if 0.60 in thresholds:
        idx = thresholds.index(0.60)
        prec_val = precision_list[idx]
        rec_val  = recall_list[idx]
        f1_val   = f1_list[idx]
    else:
        prec_val = rec_val = f1_val = None

    ax.plot(thresholds, precision_list, label=f"Precision ({prec_val:.2f})" if prec_val else "Precision", linewidth=2, color='blue')
    ax.plot(thresholds, recall_list, label=f"Recall ({rec_val:.2f})" if rec_val else "Recall", linewidth=2, color='orange')
    ax.plot(thresholds, f1_list, label=f"F1 Score ({f1_val:.2f})" if f1_val else "F1 Score", linewidth=2, color='green')
    
    # Mark threshold = 0.60 with a red star
    if 0.60 in thresholds:
        ax.scatter(thresholds[idx], precision_list[idx], color='blue', s=100, marker='*')
        ax.scatter(thresholds[idx], recall_list[idx], color='orange', s=100, marker='*')
        ax.scatter(thresholds[idx], f1_list[idx], color='green', s=100, marker='*')

    ax.set_xlabel("Threshold", fontsize=14)
    ax.set_ylabel("Metric Score", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    fig.tight_layout()
    plt.show()

    return fig


def calculate_roc_auc(offset: np.ndarray, final_prob: np.ndarray, label_binary:np.ndarray, time_tolerance=0.1):
    """Calculate ROC AUC score based on offset and predicted probabilities.

    Args:
        offset (torch.Tensor): Offset values of picks.
        final_prob (torch.Tensor): Predicted probabilities.
        time_tolerance (float): Time tolerance for considering a pick as true positive.

    Returns:
        float: ROC AUC score.
    """
    
    y_true_mask = (offset <= time_tolerance)
    phase_label = (label_binary[y_true_mask]==0)
    final_prob_masked = final_prob[y_true_mask]
    
    print(phase_label)
    tpr, fpr, thresholds = metrics.roc_curve(phase_label,
                                 final_prob_masked)
    
    roc_auc = metrics.auc(fpr, tpr)

    return tpr, fpr, roc_auc

def plot_roc_curve(
    tpr: np.ndarray,
    fpr: np.ndarray,
    auc: float,
    title: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, linewidth=3, label=f"AUC = {auc:.3f}")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)

    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)

    ax.legend(
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderpad=1,
    )

    fig.tight_layout()
    return fig

class EvaluationMetrics(NamedTuple):
    mean_difference: float
    median_difference: float
    mean_abs_error: float
    rms_error: float
    offsets: np.ndarray 


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
    offsets = (predicted - label) / sampling_rate  

    return EvaluationMetrics(
        mean_difference=float(np.mean(offsets)),
        median_difference=float(np.median(offsets)),
        mean_abs_error=float(np.mean(np.abs(offsets))),
        rms_error=float(np.sqrt(np.mean(offsets**2))),
        offsets=offsets,
    ) 
