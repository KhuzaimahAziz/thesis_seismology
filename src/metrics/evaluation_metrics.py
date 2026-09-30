from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from scipy.signal import find_peaks

PLOT_STYLE = {
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2,
}
C_BLUE, C_ORANGE, C_GREEN, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_GLOBAL = "#6f6f6f"
TOLERANCE_S = 0.1


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
        idx = torch.arange(labels.shape[2])
        start = (label_pick_sample - window_width).clamp(min=0)[..., None]
        end = (label_pick_sample + window_width).clamp(max=labels.shape[2])[..., None]
        mask = (idx >= start) & (idx < end)
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
    sr = sampling_rate
    offsets = stats.offset_samples / sr
    offsets = offsets[~np.isnan(offsets)]
    fraction_outside_window = (
        np.sum(np.abs(offsets) > time_window_limit) / offsets.size * 100.0
    )

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.axvspan(-TOLERANCE_S, TOLERANCE_S, color="0.93", lw=0, zorder=0,
                   label=f"±{TOLERANCE_S:g} s")
        kw = dict(bins=100, range=(-time_window_limit, time_window_limit), color=C_BLUE)
        ax.hist(offsets, histtype="stepfilled", alpha=0.25, **kw)
        ax.hist(offsets, histtype="step", linewidth=1.6, **kw)
        ax.axvline(stats.mean_difference / sr, color=C_ORANGE, linestyle="--",
                   linewidth=1.5, label="Mean difference")
        ax.axvline(stats.median_difference / sr, color=C_GREEN, linestyle="--",
                   linewidth=1.5, label="Median difference")
        ax.text(
            0.02, 0.97,
            f"{'Median':<9}{stats.median_difference / sr:+.3f} s\n"
            f"{'Mean':<9}{stats.mean_difference / sr:+.3f} s\n"
            f"{'MAE':<9}{stats.mean_abs_error / sr:.3f} s\n"
            f"{'RMS':<9}{stats.rms_error / sr:.3f} s\n"
            f"{'Picks':<9}{offsets.size}\n"
            f"{'Outside':<9}{fraction_outside_window:.0f}% (>±{time_window_limit:g} s)",
            transform=ax.transAxes, va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.85"),
        )
        ax.set(
            title=title,
            xlabel=r"Pick time difference $t_\mathrm{pred} - t_\mathrm{true}$ (s)",
            ylabel="Count",
            xlim=(-time_window_limit, time_window_limit),
        )
        ax.legend(loc="upper right")

    if show_figure:
        plt.show()
    return fig

def plot_comparison(
    models: dict[str, tuple[PickStats, list[DetectionMetrics]]],
    title: str,
    sampling_rate: float = 100.0,
    time_window_limit: float = 1.0,
) -> Figure:
    """Overlay residual histogram, F1-vs-threshold and ROC for several models."""
    palette = iter([C_BLUE, C_ORANGE, C_GREEN, C_YELLOW])
    colors = {n: C_GLOBAL if n.lower().startswith("global") else next(palette) for n in models}

    with plt.rc_context(PLOT_STYLE):
        fig, (ax_h, ax_f1, ax_roc) = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
        ax_h.axvspan(-TOLERANCE_S, TOLERANCE_S, color="0.93", lw=0, zorder=0,
                     label=f"±{TOLERANCE_S:g} s")
        for name, (stats, detection) in models.items():
            c = colors[name]
            offsets = stats.offset_samples / sampling_rate
            offsets = offsets[~np.isnan(offsets)]
            kw = dict(bins=100, range=(-time_window_limit, time_window_limit), color=c)
            ax_h.hist(offsets, histtype="stepfilled", alpha=0.15, **kw)
            ax_h.hist(offsets, histtype="step", linewidth=1.6,
                      label=f"{name} (MAE {stats.mean_abs_error / sampling_rate:.3f} s)", **kw)
            ax_f1.plot([d.threshold for d in detection], [d.f1_score for d in detection],
                       color=c, label=name)
            fpr, tpr = stats.roc_curve
            ax_roc.plot(fpr, tpr, color=c, label=f"{name} (AUC {stats.auc:.3f})")

        ax_roc.plot([0, 1], [0, 1], linestyle=":", color="0.6", linewidth=1)
        ax_h.set(title="Pick time difference",
                 xlabel=r"$t_\mathrm{pred} - t_\mathrm{true}$ (s)", ylabel="Count",
                 xlim=(-time_window_limit, time_window_limit))
        ax_f1.set(title="F1 vs threshold", xlabel="Threshold", ylabel="F1 score", xlim=(0, 1))
        ax_roc.set(title="ROC", xlabel="False positive rate", ylabel="True positive rate",
                   xlim=(0, 1), ylim=(0, 1.01))
        ax_roc.set_aspect("equal")
        ax_h.legend(loc="upper left")
        ax_f1.legend(loc="lower left")
        ax_roc.legend(loc="lower right")
        fig.suptitle(title, fontsize=12, fontweight="bold")
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
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.plot(thresholds, [d.precision for d in detection_metrics], color=C_BLUE, label="Precision")
        ax.plot(thresholds, [d.recall for d in detection_metrics], color=C_ORANGE, label="Recall")
        ax.plot(thresholds, [d.f1_score for d in detection_metrics], color=C_GREEN, label="F1 score")
        ax.set(title=title, xlabel="Threshold", ylabel="Metric score", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower left")
    return fig


def plot_roc_curve(
    stats: PickStats,
    title: str,
) -> Figure:
    fpr, tpr = stats.roc_curve
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
        ax.fill_between(fpr, tpr, color=C_BLUE, alpha=0.1)
        ax.plot(fpr, tpr, color=C_BLUE, label=f"AUC = {stats.auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle=":", color="0.6", linewidth=1)
        ax.set(title=title, xlabel="False positive rate", ylabel="True positive rate",
               xlim=(0, 1), ylim=(0, 1.01))
        ax.set_aspect("equal")
        ax.legend(loc="lower right")
    return fig

def extract_picks(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    label_order: ComponentOrder = "PSN",
    min_height: float = 0.05,
    min_distance: int = 100,
    tolerance: int = 50,
) -> dict[str, PickStats]:
    pred = predictions.detach().cpu().numpy()
    lab = labels.detach().cpu().numpy()
    p_idx, s_idx = ORDER_MAP[label_order]
    out = {}
    for phase, c in (("P", p_idx), ("S", s_idx)):
        pred_s, lab_s, cert, noise = [], [], [], []
        for i in range(pred.shape[0]):
            peaks, props = find_peaks(pred[i, c], height=min_height, distance=min_distance)
            heights = props["peak_heights"]
            manual, _ = find_peaks(lab[i, c], height=0.5, distance=min_distance)
            used = np.zeros(peaks.size, dtype=bool)
            for m in manual:
                near = np.flatnonzero((np.abs(peaks - m) <= tolerance) & ~used)
                lab_s.append(m)
                if near.size:
                    j = near[np.argmax(heights[near])]
                    used[j] = True
                    pred_s.append(peaks[j])
                    cert.append(heights[j])
                else:
                    pred_s.append(np.nan)
                    cert.append(0.0)
            noise.extend(heights[~used])
        out[phase] = PickStats(
            predicted_samples=np.asarray(pred_s, dtype=float),
            labeled_samples=np.asarray(lab_s, dtype=float),
            predicted_certainty=np.asarray(cert, dtype=float),
            noise_max=np.asarray(noise, dtype=float),
        )
    return out

def at_threshold(stats: PickStats, threshold: float) -> PickStats:
    """Keep only picks whose probability reaches the threshold; the rest count as missed."""
    keep = stats.predicted_certainty >= threshold
    return stats._replace(predicted_samples=np.where(keep, stats.predicted_samples, np.nan))