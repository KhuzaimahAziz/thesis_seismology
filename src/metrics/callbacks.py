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
import pandas as pd


from metrics.evaluation_metrics import (
    DetectionMetrics,
    PickStats,
    calculate_pick_differences,
    calculate_precision_recall_f1,
    get_f1_optimal_metrics,
    plot_histogram,
    plot_precision_recall_f1,
    plot_roc_curve,
    plot_comparison,
    extract_picks,
    at_threshold,
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

    def __init__(self, mlflow: MLFlowLogger, baseline_model_name: str = "original",
                 eval_cfg=None, test_loader=None) -> None:
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

        # label-independent evaluation (always set, with or without a global model)
        self.eval_cfg = eval_cfg
        self.test_loader = test_loader
        self.det_stats = CollectedStats()
        self.det_baseline_stats = CollectedStats()
        self.det_thresholds: dict[str, float] = {}
        self.best_det_thresholds: dict[str, float] = {}
        self.global_det_thresholds: dict[str, float] = {}

    def _pick_kwargs(self, tolerance_s: float) -> dict:
        c = self.eval_cfg
        return dict(
            min_height=float(c.min_peak_height),
            min_distance=max(1, int(round(c.min_peak_distance_s * SAMPLING_RATE))),
            tolerance=int(round(tolerance_s * SAMPLING_RATE)),
        )

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
            if self.eval_cfg is not None and self.eval_cfg.test_at_end and self.test_loader is not None:
                self.evaluate_test(pl_module)

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
        run_id = self.mlflow_logger.run_id
        epoch = trainer.current_epoch
        local_name = "Local-fine-tuned" if pl_module.pretrained_model_name else "Local-scratch"

        for phase in ("P", "S"):
            # legacy window-based metrics: kept under the old names so screening runs stay comparable
            pick_stats = self.stats.get_stats(phase)
            if not pick_stats.n_samples:
                print(f"No {phase} picks to log.")
                continue
            metric_results = calculate_precision_recall_f1(stats=pick_stats)
            self.mlflow_logger.log_metrics(
                scalar_metrics(pick_stats, metric_results, phase),
                step=trainer.global_step,
            )

            # values used for the plots: label-independent if enabled, legacy otherwise
            plot_stats, plot_results, thr_note = pick_stats, metric_results, ""
            if self.eval_cfg is not None:
                det = self.det_stats.get_stats(phase)
                det_results = calculate_precision_recall_f1(det)
                best = get_f1_optimal_metrics(det_results)
                self.det_thresholds[phase] = float(best.threshold)
                plot_stats = at_threshold(det, best.threshold)
                plot_results = det_results
                thr_note = f" (threshold {best.threshold:.2f})"
                self.mlflow_logger.log_metrics(
                    scalar_metrics(plot_stats, det_results, f"det_{phase}"),
                    step=trainer.global_step,
                )

            self.experiment.log_figure(
                run_id,
                plot_histogram(
                    stats=plot_stats,
                    sampling_rate=SAMPLING_RATE,
                    title=f"{phase}-Pick Differences - Epoch {epoch}{thr_note}",
                ),
                f"histograms/{phase}-phase/epoch-{epoch:03d}.png",
            )
            self.experiment.log_figure(
                run_id,
                plot_precision_recall_f1(
                    plot_results,
                    title=f"{phase}-Precision, Recall and F1 Score - Epoch {epoch}",
                ),
                f"precision_recall_f1_plots/{phase}-phase/epoch-{epoch:03d}.png",
            )
            self.experiment.log_figure(
                run_id,
                plot_roc_curve(
                    stats=plot_stats,
                    title=f"ROC Curve for {phase}-wave Picks - Epoch {epoch}",
                ),
                f"roc_curve_plot/{phase}-phase/epoch-{epoch:03d}.png",
            )
            print("Logged plots for phase", phase)

            if self.baseline is not None:
                base_stats = self.baseline_stats.get_stats(phase)
                base_results = calculate_precision_recall_f1(stats=base_stats)
                self.mlflow_logger.log_metrics(
                    scalar_metrics(base_stats, base_results, f"global_{phase}"),
                    step=trainer.global_step,
                )

                g_plot_stats, g_plot_results = base_stats, base_results
                if self.eval_cfg is not None:
                    gdet = self.det_baseline_stats.get_stats(phase)
                    g_results = calculate_precision_recall_f1(gdet)
                    gbest = get_f1_optimal_metrics(g_results)
                    self.global_det_thresholds[phase] = float(gbest.threshold)
                    g_plot_stats = at_threshold(gdet, gbest.threshold)
                    g_plot_results = g_results
                    self.mlflow_logger.log_metrics(
                        scalar_metrics(g_plot_stats, g_results, f"det_global_{phase}"),
                        step=trainer.global_step,
                    )

                self.experiment.log_figure(
                    run_id,
                    plot_comparison(
                        {
                            "Global/original": (g_plot_stats, g_plot_results),
                            local_name: (plot_stats, plot_results),
                        },
                        title=f"{phase}-phase: {local_name} vs Global - Epoch {epoch}",
                        sampling_rate=SAMPLING_RATE,
                    ),
                    f"comparison/{phase}-phase/epoch-{epoch:03d}.png",
                )

        self.stats.clear()
        self.baseline_stats.clear()
        self.det_stats.clear()
        self.det_baseline_stats.clear()
        plt.close("all")

        previous_best = self.best_model
        self.store_model(trainer, pl_module)
        if self.best_model is not previous_best:
            self.best_det_thresholds = dict(self.det_thresholds)

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
        if self.eval_cfg is not None:                                           
            self.det_stats.add(extract_picks(                                   
                label_predicted.cpu(), label_data.cpu(), pl_module.label_order,
                **self._pick_kwargs(self.eval_cfg.tolerance_s)))
            
        if self.baseline is not None:
            X = batch["X"]
            self.baseline.to(X.device)
            with torch.no_grad():
                pred = self.baseline(self.baseline.annotate_batch_pre(X, {}))
            order = [self.baseline.labels.index(c) for c in pl_module.label_order]
            self.baseline_stats.add(
                calculate_pick_differences(
                    pred[:, order].cpu(), label_data.cpu(), window_width=200, label_order=pl_module.label_order))
            if self.eval_cfg is not None:                                       
                self.det_baseline_stats.add(extract_picks(                      
                    pred[:, order].cpu(), label_data.cpu(), pl_module.label_order,
                    **self._pick_kwargs(self.eval_cfg.tolerance_s)))

    @torch.no_grad()
    def evaluate_test(self, pl_module: SeisBenchLit) -> None:
        tolerances = [float(t) for t in self.eval_cfg.test_tolerances_s]
        device = pl_module.device
        pl_module.model.eval()
        local = {t: CollectedStats() for t in tolerances}
        glob = {t: CollectedStats() for t in tolerances}

        for batch in self.test_loader:
            X, y = batch["X"].to(device), batch["y"]
            pred = pl_module.model(pl_module.model.annotate_batch_pre(X, {})).cpu()
            base_pred = None
            if self.baseline is not None:
                self.baseline.to(device)
                order = [self.baseline.labels.index(c) for c in pl_module.label_order]
                base_pred = self.baseline(self.baseline.annotate_batch_pre(X, {}))[:, order].cpu()
            for t in tolerances:
                kw = self._pick_kwargs(t)
                local[t].add(extract_picks(pred, y, pl_module.label_order, **kw))
                if base_pred is not None:
                    glob[t].add(extract_picks(base_pred, y, pl_module.label_order, **kw))

        local_name = "Local-fine-tuned" if pl_module.pretrained_model_name else "Local-scratch"
        run_id = self.mlflow_logger.run_id
        sr = SAMPLING_RATE
        rows = []
        for t in tolerances:
            for phase in ("P", "S"):
                folder = f"final_evaluation/tolerance_{t:g}s/{phase}_phase"
                models = {local_name: (local[t], "local", self.best_det_thresholds.get(phase, 0.5))}
                if self.baseline is not None:
                    models["Global/original"] = (glob[t], "global", self.global_det_thresholds.get(phase, 0.5))
                comparison = {}
                for name, (collected, slug, thr) in models.items():
                    stats = collected.get_stats(phase)
                    curve = calculate_precision_recall_f1(stats)
                    at_thr = calculate_precision_recall_f1(stats, thresholds=np.array([thr]))[0]
                    picked = at_threshold(stats, thr)
                    row = {
                        "model": name,
                        "phase": phase,
                        "tolerance_s": t,
                        "threshold": float(thr),
                        "precision": float(at_thr.precision),
                        "recall": float(at_thr.recall),
                        "f1": float(at_thr.f1_score),
                        "mae_s": picked.mean_abs_error / sr,
                        "median_s": picked.median_difference / sr,
                        "rms_s": picked.rms_error / sr,
                        "n_manual_picks": int(stats.labeled_samples.size),
                        "n_false_picks": int((stats.noise_max >= thr).sum()),
                    }
                    rows.append(row)
                    self.mlflow_logger.log_metrics({
                        f"test_{slug}_{phase}_tol{t:g}s_{k}": v
                        for k, v in row.items() if k not in ("model", "phase", "tolerance_s")
                    })
                    self.experiment.log_figure(
                        run_id,
                        plot_precision_recall_f1(curve, title=f"{name}: {phase}-phase, test set, ±{t:g} s"),
                        f"{folder}/{slug}_model_precision_recall_f1.png",
                    )
                    self.experiment.log_figure(
                        run_id,
                        plot_histogram(
                            picked,
                            sampling_rate=sr,
                            title=f"{name}: {phase}-phase residuals, test set, ±{t:g} s (threshold {thr:.2f})",
                        ),
                        f"{folder}/{slug}_model_residual_histogram.png",
                    )
                    comparison[name] = (picked, curve)
                if len(comparison) > 1:
                    self.experiment.log_figure(
                        run_id,
                        plot_comparison(
                            comparison,
                            sampling_rate=sr,
                            title=f"{phase}-phase: {local_name} vs Global, test set, ±{t:g} s",
                        ),
                        f"{folder}/comparison_local_vs_global.png",
                    )
                plt.close("all")

        self.experiment.log_text(
            run_id,
            pd.DataFrame(rows).to_csv(index=False),
            "final_evaluation/summary_all_models.csv",
        )


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


