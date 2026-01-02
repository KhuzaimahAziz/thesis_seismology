from pathlib import Path

import pytorch_lightning as pl
import seisbench.models as sbm
import torch

phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}


def loss_fn(y_pred, y_true, eps=1e-5):
    """
    Cross entropy loss

    :param y_true: True label probabilities
    :param y_pred: Predicted label probabilities
    :param eps: Epsilon to clip values for stability
    :return: Average loss across batch
    """
    h = y_true * torch.log(y_pred + eps)
    if y_pred.ndim == 3:
        h = h.mean(-1).sum(-1)
    else:
        h = h.sum(-1)  # Sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h


class SeisBenchLit(pl.LightningModule):
    model: sbm.PhaseNet

    def __init__(
        self,
        dataset_name: str,
        pretrained_model_name: str = "",
        sample_boundaries=(None, None),
        optimizer_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.sample_boundaries = sample_boundaries
        self.optimizer_params = optimizer_params or {}

        self.pretrained_model_name = pretrained_model_name
        if self.pretrained_model_name:
            print(f"Loading pretrained model: {self.pretrained_model_name}")
            self.model = sbm.PhaseNet.from_pretrained(self.pretrained_model_name)
        else:
            self.model = sbm.PhaseNet()

        self.model_name = f"{self.model.__class__.__name__}-{dataset_name}"
        if self.pretrained_model_name:
            self.model_name += f"-from-{self.pretrained_model_name}"

    @property
    def label_order(self) -> str:
        return self.model.labels

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x = self.model.annotate_batch_pre(batch["X"], {})
        y_true = batch["y"]
        y_pred = self.model(x)
        return loss_fn(y_pred, y_true), y_pred

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        return optimizer

    def save_model(self, path: str | Path):
        return self.model.save(path)

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch["X"]
        window_borders = batch["window_borders"]

        pred = self.model(x)

        score_detection = torch.zeros(pred.shape[0])
        score_p_or_s = torch.zeros(pred.shape[0])
        p_sample = torch.zeros(pred.shape[0], dtype=int)
        s_sample = torch.zeros(pred.shape[0], dtype=int)

        for i in range(pred.shape[0]):
            start_sample, end_sample = window_borders[i]
            local_pred = pred[i, :, start_sample:end_sample]

            score_detection[i] = torch.max(1 - local_pred[-1])  # 1 - noise
            score_p_or_s[i] = torch.max(local_pred[0]) / torch.max(local_pred[1])

            p_sample[i] = torch.argmax(local_pred[0])
            s_sample[i] = torch.argmax(local_pred[1])

        return score_detection, score_p_or_s, p_sample, s_sample
