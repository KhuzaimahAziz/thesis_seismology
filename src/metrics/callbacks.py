from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, NamedTuple
import torch
import mlflow
import numpy as np
from matplotlib import pyplot as plt
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import MLFlowLogger
from torch import Tensor
import seisbench.models as sbm
import copy


from metrics.evaluation_metrics import (
    DetectionMetrics,
    PickStats,
    calculate_pick_differences,
    calculate_precision_recall_f1,
    get_f1_optimal_metrics,
    plot_histogram,
    plot_precision_recall_f1,
    plot_roc_curve,
    plot_comparison
)
from seisbench_training.utils.model_utils import SeisBenchLit

if TYPE_CHECKING:
    from mlflow.tracking.client import MlflowClient

SAMPLING_RATE = 100.0

mlflow.enable_system_metrics_logging()


class CollectedStats:
    def __init__(self) -> None:
        self.stats: dict[str, list[PickStats]] = defaultdict(list)

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
    state: dict
    loss: float


class EvaluationMetrics(Callback):
    scores: list[float]
    system_monitor: SystemMetricsMonitor

    stats: CollectedStats

    mlflow_logger: MLFlowLogger
    experiment: MlflowClient

    best_model: BestModel | None = None

    def __init__(self, mlflow: MLFlowLogger, baseline_model_name: str = "original") -> None:
        self.scores = []

        self.stats = CollectedStats()
        self.mlflow_logger = mlflow
        self.experiment = mlflow.experiment
        super().__init__()

        self.baseline = None
        self.baseline_stats = CollectedStats()
        if baseline_model_name:
            self.baseline = sbm.PhaseNet.from_pretrained(baseline_model_name).eval()
            self.baseline.requires_grad_(False)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.system_monitor = SystemMetricsMonitor(run_id=self.mlflow_logger.run_id)
        self.system_monitor.start()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.system_monitor:
            self.system_monitor.finish()

    def on_train_end(self, trainer: Trainer, pl_module: SeisBenchLit) -> None:
        if self.best_model:
            pl_module.model.load_state_dict(self.best_model.state)
            with TemporaryDirectory() as tmpdir:
                model_dir = Path(tmpdir) / "best_model"
                model_dir.mkdir()
                pl_module.save_model(model_dir / pl_module.model_name)
                self.experiment.log_artifact(self.mlflow_logger.run_id, model_dir)
            self.mlflow_logger.log_metrics({"best_val_loss": self.best_model.loss})

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

            self.mlflow_logger.log_metrics(
                scalar_metrics(pick_stats, metric_results, phase),
                step=trainer.global_step,
            )

            if self.baseline is not None:
                base_stats = self.baseline_stats.get_stats(phase)
                base_results = calculate_precision_recall_f1(stats=base_stats)
                self.mlflow_logger.log_metrics(
                    scalar_metrics(base_stats, base_results, f"global_{phase}"),
                    step=trainer.global_step,
                )

                local_name = (
                    "Local-fine-tuned" if pl_module.pretrained_model_name else "Local-scratch"
                )
                figure_comparison = plot_comparison(
                    {
                        "Global/original": (base_stats, base_results),
                        local_name: (pick_stats, metric_results),
                    },
                    title=f"{phase}-phase: {local_name} vs Global - Epoch {trainer.current_epoch}",
                    sampling_rate=SAMPLING_RATE,
                )
                self.experiment.log_figure(
                    self.mlflow_logger.run_id,
                    figure_comparison,
                    f"comparison/{phase}-phase/epoch-{trainer.current_epoch:03d}.png",
                )
        self.stats.clear()
        self.baseline_stats.clear()
        plt.close("all")

        self.store_model(trainer, pl_module)

    def store_model(self, trainer: Trainer, pl_module: SeisBenchLit) -> None:
        current_loss = float(trainer.callback_metrics["val_loss"])
        if self.best_model is None or current_loss < self.best_model.loss:
            if self.best_model is not None:
                print(
                    f"New best model found with val_loss: {current_loss:.4f} "
                    f"(previous: {self.best_model.loss:.4f})"
                )
            self.best_model = BestModel(
                state=copy.deepcopy(pl_module.model.state_dict()),
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
        if self.baseline is not None:
            X = batch["X"]
            self.baseline.to(X.device)
            with torch.no_grad():
                pred = self.baseline(self.baseline.annotate_batch_pre(X, {}))
            order = [self.baseline.labels.index(c) for c in pl_module.label_order]
            self.baseline_stats.add(
                calculate_pick_differences(
                    pred[:, order].cpu(), label_data.cpu(), window_width=200, label_order=pl_module.label_order))

def scalar_metrics(
    stats: PickStats, detection: list[DetectionMetrics], prefix: str
) -> dict[str, float]:
    best = get_f1_optimal_metrics(detection)
    sr = SAMPLING_RATE
    return {
        f"{prefix}_mean_difference": stats.mean_difference / sr,
        f"{prefix}_median_difference": stats.median_difference / sr,
        f"{prefix}_mean_abs_error": stats.mean_abs_error / sr,
        f"{prefix}_rms_error": stats.rms_error / sr,
        f"{prefix}_precision": best.precision,
        f"{prefix}_recall": best.recall,
        f"{prefix}_f1_score": best.f1_score,
        f"{prefix}_optimal_threshold": best.threshold,
        f"{prefix}_auc": stats.auc,
    }
