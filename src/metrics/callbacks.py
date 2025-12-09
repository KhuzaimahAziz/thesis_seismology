from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
from mlflow.metrics import MetricValue
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import MLFlowLogger
from torch import Tensor

from metrics.evaluation_metrics import (
    calculate_pick_differences,
    compute_evaluation_metrics,
    plot_histogram,
    calculate_precision_recall_f1,
    plot_precision_recall_f1,
    calculate_roc_auc,
    plot_roc_curve
)

if TYPE_CHECKING:
    from mlflow.tracking.client import MlflowClient


class EvaluationMetrics(Callback):
    scores: list[float]

    p_picks_labels: list[np.ndarray]
    p_picks_predictions: list[np.ndarray]
    s_picks_labels: list[np.ndarray]
    s_pick_predictions: list[np.ndarray]
    p_prob: list[np.ndarray]
    s_prob: list[np.ndarray]
    p_labels_binary: list[np.ndarray]
    s_labels_binary: list[np.ndarray]

    mlflow_logger: MLFlowLogger
    experiment: MlflowClient

    def __init__(self, mlflow: MLFlowLogger) -> None:
        self.scores = []

        self.p_picks_labels = []
        self.p_picks_predictions = []
        self.s_picks_labels = []
        self.s_pick_predictions = []
        self.p_prob = []
        self.s_prob = []
        self.p_labels_binary = []
        self.s_labels_binary = []
        self.mlflow_logger = mlflow
        self.experiment = mlflow.experiment
        super().__init__()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Save the model
        ...

    def get_picks(self, phase: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if phase == "P":
            return self.p_picks_predictions, self.p_picks_labels, self.p_prob, self.p_labels_binary
        else:
            return self.s_pick_predictions, self.s_picks_labels, self.s_prob, self.s_labels_binary

    def on_validation_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.scores.clear()
        print("Validation started")

    def on_validation_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        for phase in ("P", "S"):
            predictions, labels, probs, label_binary = self.get_picks(phase)
            if not predictions or not labels:
                print(f"No {phase} picks to log.")
                continue

            stats = compute_evaluation_metrics(
                predicted=np.concatenate(predictions),
                label=np.concatenate(labels),
                sampling_rate=100.0,
            )

            figure_hist = plot_histogram(
                predicted=np.concatenate(predictions).ravel(),
                label=np.concatenate(labels).ravel(),
                stats=stats,
                title=f"{phase}-Pick Differences - Epoch {trainer.current_epoch}",
            )

            self.experiment.log_figure(
                self.mlflow_logger.run_id,
                figure_hist,
                f"histograms/{phase}-phase/epoch-{trainer.current_epoch:03d}.png",
            )
            print("Logged histogram for phase", phase)

            metric_results = calculate_precision_recall_f1(offset= stats.offsets, final_prob=np.concatenate(probs).ravel())
            figure_precision_recall_f1 = plot_precision_recall_f1(metric_results,
                                     title=f"{phase}-Precision, Recall and F1 Score - Epoch {trainer.current_epoch}")
            self.experiment.log_figure(
                self.mlflow_logger.run_id,
                figure_precision_recall_f1,
                f"precision_recall_f1_plots/{phase}-phase/epoch-{trainer.current_epoch:03d}.png",
            )

            print("Logged Metrics for phase", phase)

            tpr, fpr, auc = calculate_roc_auc(offset= stats.offsets, final_prob=np.concatenate(probs).ravel(), label_binary=np.concatenate(label_binary).ravel())
            figure_roc = plot_roc_curve(tpr, fpr, auc,  title=f"ROC Curve for {phase}-wave Picks - Epoch {trainer.current_epoch}")
            self.experiment.log_figure(
                            self.mlflow_logger.run_id,
                            figure_roc,
                            f"roc_curve_plot/{phase}-phase/epoch-{trainer.current_epoch:03d}.png",
                        )
            print("Logged ROC Curve for phase", phase)
            self.mlflow_logger.log_metrics(
                {
                    f"{phase}_mean_difference": stats.mean_difference,
                    f"{phase}_median_difference": stats.median_difference,
                    f"{phase}_mean_abs_error": stats.mean_abs_error,
                    f"{phase}_rms_error": stats.rms_error,
                    f"{phase}_precision": metric_results[0.6]["precision"],
                    f"{phase}_recall": metric_results[0.6]["recall"],
                    f"{phase}_f1_score": metric_results[0.6]["f1"],
                    f"{phase}_auc": auc

                },
                step=trainer.global_step,
            )

        self.p_picks_labels.clear()
        self.p_picks_predictions.clear()
        self.s_picks_labels.clear()
        self.s_pick_predictions.clear()
        self.p_prob.clear()
        self.s_prob.clear()
        self.p_labels_binary.clear()
        self.s_labels_binary.clear()


        
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: dict[Literal["X", "y"], Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:


        waveform_data = batch["X"]
        label_data = batch["y"]
        label_predicted: Tensor = pl_module(waveform_data)
        # print(outputs.shape)
        # print(label_predicted.shape)
        # print(outputs)
        # print(label_predicted)

        torch.save(waveform_data.cpu(), "data/example_waveform.pt")
        torch.save(label_data.cpu(), "data/example_labels.pt")
        torch.save(label_predicted.cpu(), "data/example_predictions.pt")

        # p_differences, s_differences = get_pick_differences(label_data, label_predicted)

        # self.p_pick_differences.append(p_differences)
        # self.s_pick_differences.append(s_differences)

        # Debug below
        # for key, value in batch.items():
        #     print(key, type(value), value, value.shape)
        # torch.save(label_data, "example_labels.pt")
        # torch.save(waveform_data, "example_waveform.pt")
        # torch.save(label_predicted, "example_predictions.pt")
        p_predicted, p_labels, s_predicted, s_labels, p_prob, s_prob, p_labels_binary, s_labels_binary = calculate_pick_differences(
            label_predicted.cpu(), label_data.cpu(), window_width=500
        )

        self.p_picks_predictions.append(p_predicted)
        self.p_picks_labels.append(p_labels)

        self.s_pick_predictions.append(s_predicted)
        self.s_picks_labels.append(s_labels)

        self.p_prob.append(p_prob)
        self.s_prob.append(s_prob)


        self.p_labels_binary.append(p_labels_binary)
        self.s_labels_binary.append(s_labels_binary)

