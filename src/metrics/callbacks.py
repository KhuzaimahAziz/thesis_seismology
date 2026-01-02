from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, NamedTuple

import mlflow
import numpy as np
from matplotlib import pyplot as plt
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import MLFlowLogger
from torch import Tensor

from metrics.evaluation_metrics import (
    PickStats,
    calculate_pick_differences,
    calculate_precision_recall_f1,
    get_f1_optimal_metrics,
    plot_histogram,
    plot_precision_recall_f1,
    plot_roc_curve,
)
from seisbench_training.utils.model_utils import SeisBenchLit

if TYPE_CHECKING:
    from mlflow.tracking.client import MlflowClient

SAMPLING_RATE = 100.0

mlflow.enable_system_metrics_logging()


class CollectedStats:
    stats: dict[str, list[PickStats]] = defaultdict(list)

    def get_stats(self, phase: str) -> PickStats:
        if phase not in self.stats:
            raise ValueError(f"No stats collected for phase {phase}")
        stats = self.stats[phase]
        return PickStats(
            predicted_samples=np.concatenate([s.predicted_samples for s in stats]),
            labeled_samples=np.concatenate([s.labeled_samples for s in stats]),
            predicted_certainty=np.concatenate([s.predicted_certainty for s in stats]),
            noise_max=np.concatenate([s.noise_max for s in stats]),
        )

    def add(self, new_stats: dict[str, PickStats]) -> None:
        for phase, stat in new_stats.items():
            self.stats[phase].append(stat)

    def clear(self) -> None:
        self.stats.clear()


class BestModel(NamedTuple):
    model: SeisBenchLit
    loss: float


class EvaluationMetrics(Callback):
    scores: list[float]
    system_monitor: SystemMetricsMonitor

    stats: CollectedStats

    mlflow_logger: MLFlowLogger
    experiment: MlflowClient

    best_model: BestModel | None = None

    def __init__(self, mlflow: MLFlowLogger) -> None:
        self.scores = []

        self.stats = CollectedStats()
        self.mlflow_logger = mlflow
        self.experiment = mlflow.experiment
        super().__init__()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.system_monitor = SystemMetricsMonitor(run_id=self.mlflow_logger.run_id)
        self.system_monitor.start()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.system_monitor:
            self.system_monitor.finish()

    def on_train_end(self, trainer: Trainer, pl_module: SeisBenchLit) -> None:
        if self.best_model:
            with TemporaryDirectory() as tmpdir:
                model = self.best_model.model

                model_dir = Path(tmpdir) / "best_model"
                model_dir.mkdir()
                model.save_model(model_dir / model.model_name)
                self.experiment.log_artifact(
                    self.mlflow_logger.run_id,
                    model_dir,
                )

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
            pick_stats = self.stats.get_stats(phase)
            if not pick_stats.n_samples:
                print(f"No {phase} picks to log.")
                continue

            figure_hist = plot_histogram(
                stats=pick_stats,
                sampling_rate=SAMPLING_RATE,
                title=f"{phase}-Pick Differences - Epoch {trainer.current_epoch}",
            )

            self.experiment.log_figure(
                self.mlflow_logger.run_id,
                figure_hist,
                f"histograms/{phase}-phase/epoch-{trainer.current_epoch:03d}.png",
            )
            print("Logged histogram for phase", phase)

            metric_results = calculate_precision_recall_f1(stats=pick_stats)

            figure_precision_recall_f1 = plot_precision_recall_f1(
                metric_results,
                title=f"{phase}-Precision, Recall and F1 Score "
                f"- Epoch {trainer.current_epoch}",
            )
            self.experiment.log_figure(
                self.mlflow_logger.run_id,
                figure_precision_recall_f1,
                f"precision_recall_f1_plots/"
                f"{phase}-phase/epoch-{trainer.current_epoch:03d}.png",
            )

            print("Logged Metrics for phase", phase)

            figure_roc = plot_roc_curve(
                stats=pick_stats,
                title=f"ROC Curve for {phase}-wave Picks - Epoch {trainer.current_epoch}",
            )
            self.experiment.log_figure(
                self.mlflow_logger.run_id,
                figure_roc,
                f"roc_curve_plot/{phase}-phase/epoch-{trainer.current_epoch:03d}.png",
            )
            print("Logged ROC Curve for phase", phase)
            optimal_detection_metrics = get_f1_optimal_metrics(metric_results)

            sr = SAMPLING_RATE
            self.mlflow_logger.log_metrics(
                {
                    f"{phase}_mean_difference": pick_stats.mean_difference / sr,
                    f"{phase}_median_difference": pick_stats.median_difference / sr,
                    f"{phase}_mean_abs_error": pick_stats.mean_abs_error / sr,
                    f"{phase}_rms_error": pick_stats.rms_error / sr,
                    f"{phase}_precision": optimal_detection_metrics.precision,
                    f"{phase}_recall": optimal_detection_metrics.recall,
                    f"{phase}_f1_score": optimal_detection_metrics.f1_score,
                    f"{phase}_optimal_threshold": optimal_detection_metrics.threshold,
                    f"{phase}_auc": pick_stats.auc,
                },
                step=trainer.global_step,
            )

        self.stats.clear()
        plt.close("all")

        self.store_model(trainer, pl_module)  # type: ignore

    def store_model(self, trainer: Trainer, pl_module: SeisBenchLit) -> None:
        callback_metrics = trainer.callback_metrics

        if self.best_model is None:
            self.best_model = BestModel(
                model=pl_module,
                loss=callback_metrics["val_loss"].item(),
            )
            return

        current_loss = float(callback_metrics["val_loss"])
        if current_loss < self.best_model.loss:
            print(
                f"New best model found with val_loss: {current_loss:.4f} "
                f"(previous: {self.best_model.loss:.4f})"
            )
            self.best_model = BestModel(
                model=pl_module,
                loss=current_loss,
            )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: SeisBenchLit,
        outputs: tuple[Tensor, Tensor],
        batch: dict[Literal["X", "y"], Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        label_data = batch["y"]
        _, label_predicted = outputs

        # Debug below
        # for key, value in batch.items():
        #     print(key, type(value), value, value.shape)
        # torch.save(label_data, "example_labels.pt")
        # torch.save(waveform_data, "example_waveform.pt")
        # torch.save(label_predicted, "example_predictions.pt")
        stats = calculate_pick_differences(
            label_predicted.cpu(),
            label_data.cpu(),
            window_width=200,
            label_order=pl_module.label_order,
        )
        self.stats.add(stats)
