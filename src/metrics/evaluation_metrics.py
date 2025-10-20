from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from mlflow.metrics import MetricValue
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import MLFlowLogger
from torch import Tensor


class EvaluationMetrics(Callback):
    scores: list[float]
    p_pick_differences: list[Tensor]
    s_pick_differences: list[Tensor]
    mlflow: MLFlowLogger

    def __init__(self, mlflow: MLFlowLogger) -> None:
        self.scores = []
        self.p_pick_differences = []
        self.s_pick_differences = []
        self.mlflow = mlflow
        super().__init__()

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
        print("Validation ended")

        # Concatenate tensors?
        # Dump s_pick_differences to a pandas DataFrame?

        # Plot histogram of difference
        # Add as mlflow artifact

        self.mlflow.log_metrics({"evaluation_metric": np.random.uniform(0, 1)})

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: dict[Literal["X", "y"], Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        print(type(outputs))
        print(outputs, outputs.shape)
        print(len(batch))

        waveform_data = batch["X"]
        label_data = batch["y"]
        label_predicted: Tensor = pl_module.forward(waveform_data)

        # p_differences, s_differences = get_pick_differences(label_data, label_predicted)

        # self.p_pick_differences.append(p_differences)
        # self.s_pick_differences.append(s_differences)

        # Debug below
        for key, value in batch.items():
            print(key, type(value), value, value.shape)
        torch.save(label_data, "example_labels.pt")
        torch.save(waveform_data, "example_waveform.pt")
        torch.save(label_predicted, "example_predictions.pt")

        example_data = waveform_data[0]
        example_label = label_data[0]
        label_predicted = label_predicted[0]

        plt.plot(example_data.cpu().numpy().T)
        print("saving example waveform plot")
        plt.savefig("example_waveform.png")
        plt.close()

        plt.plot(example_label.cpu().numpy().T)
        print("saving example label plot")
        plt.savefig("example_label.png")
        plt.close()

        plt.plot(label_predicted.cpu().numpy().T)
        print("saving example predicted label plot")
        plt.savefig("example_predictions.png")
        plt.close()

        self.scores.append(0.5)


def test(
    predictions: pd.Series,
    targets: pd.Series,
    metrics: dict[str, MetricValue],
    **kwargs,
) -> float | MetricValue:
    print("Predictions:", predictions)
    print("Targets:", targets)
    print("Metrics:", metrics)
    scores = np.random.uniform(0, 1, size=1)

    return MetricValue(scores=scores.tolist())
