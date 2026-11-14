"""Lab 6 - configurable PyTorch Lightning pipeline for the Forest Cover dataset.

The defaults reproduce the final (best) model. Every experiment in the lab report can be
re-created from the command line, e.g.

    python lab6.py --name part3_arch_256x256 --hidden 256 256 --epochs 5 --batch-size 1024 \
        --lr 1e-3 --scheduler none --no-shuffle --dropout 0 --weight-decay 0

Each run appends a JSON line with its configuration and results to results.jsonl.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import loggers as pl_loggers
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

TARGET = "Cover_Type"
N_CONTINUOUS = 10  # The first 10 columns are continuous; the remaining 44 are one-hot flags.
N_CLASSES = 7


@dataclass(frozen=True)
class Config:
    name: str
    data: str
    hidden: tuple[int, ...]
    batch_size: int
    epochs: int
    lr: float
    scheduler: str
    step_size: int
    gamma: float
    scale: bool
    shuffle: bool
    dropout: float
    batch_norm: bool
    weight_decay: float
    seed: int
    device: str


class ForestCoverDataset(Dataset):
    """Holds the whole split as tensors so batches are gathered without per-row conversion."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def load_splits(cfg: Config) -> tuple[ForestCoverDataset, ForestCoverDataset, ForestCoverDataset]:
    """Fixed-seed 70/10/20 split; the scaler is fit on the training split only to avoid leakage."""
    df = pd.read_csv(cfg.data)
    features = df.drop(columns=[TARGET]).to_numpy(dtype=np.float64, copy=True)
    labels = df[TARGET].to_numpy() - 1  # Zero-index the 1..7 labels.

    n_total = len(df)
    n_train, n_val = int(0.7 * n_total), int(0.1 * n_total)
    order = np.random.default_rng(cfg.seed).permutation(n_total)
    train_idx, val_idx, test_idx = np.split(order, [n_train, n_train + n_val])

    if cfg.scale:
        scaler = StandardScaler().fit(features[train_idx, :N_CONTINUOUS])
        features[:, :N_CONTINUOUS] = scaler.transform(features[:, :N_CONTINUOUS])

    return tuple(ForestCoverDataset(features[idx], labels[idx]) for idx in (train_idx, val_idx, test_idx))


def build_mlp(input_ch: int, hidden: tuple[int, ...], n_classes: int, dropout: float, batch_norm: bool):
    layers: list[nn.Module] = []
    in_features = input_ch
    for width in hidden:
        layers.append(nn.Linear(in_features, width))
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_features = width
    layers.append(nn.Linear(in_features, n_classes))
    return nn.Sequential(*layers)


class LitNetwork(pl.LightningModule):
    def __init__(self, cfg: Config, input_ch: int, n_classes: int, steps_per_epoch: int):
        super().__init__()
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        self.network = build_mlp(input_ch, cfg.hidden, n_classes, cfg.dropout, cfg.batch_norm)
        self.loss_func = nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average="micro")
        self.test_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average="micro")

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        cfg = self.cfg
        if cfg.weight_decay > 0:
            optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        if cfg.scheduler == "none":
            return optimizer
        if cfg.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        if cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        if cfg.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=self.steps_per_epoch
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    def training_step(self, data, batch_idx):
        features, target = data
        loss = self.loss_func(self(features), target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_data, batch_idx):
        features, target = val_data
        self.val_acc(self(features), target)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, test_data, batch_idx):
        features, target = test_data
        self.test_acc(self(features), target)
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)


def run(cfg: Config) -> dict:
    pl.seed_everything(cfg.seed, workers=True)
    train_ds, val_ds, test_ds = load_splits(cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    val_loader = DataLoader(val_ds, batch_size=8192)
    test_loader = DataLoader(test_ds, batch_size=8192)

    model = LitNetwork(cfg, train_ds.features.shape[1], N_CLASSES, len(train_loader))
    checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc", save_top_k=1, mode="max")
    callbacks = [checkpoint, pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    logger = pl_loggers.TensorBoardLogger(save_dir="my_logs", name=cfg.name)

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.device,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=max(1, min(50, len(train_loader) // 4)),
        enable_progress_bar=False,
    )
    start = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    train_time = time.perf_counter() - start

    test_acc = trainer.test(ckpt_path="best", dataloaders=test_loader, verbose=False)[0]["test_acc"]
    result = {
        **asdict(cfg),
        "test_acc": test_acc,
        "best_val_acc": checkpoint.best_model_score.item(),
        "best_checkpoint": checkpoint.best_model_path,
        "train_time_s": round(train_time, 1),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"RESULT {cfg.name}: test_acc={test_acc:.4f} train_time={train_time:.1f}s")
    return result


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="part7_final")
    p.add_argument("--data", default="./datasets/forest_cover.csv")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024, 512, 256])
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--scheduler", choices=["none", "step", "cosine", "onecycle"], default="onecycle")
    p.add_argument("--step-size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--no-scale", dest="scale", action="store_false")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch-norm", action="store_true")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="'cpu', 'mps' (Apple silicon) or 'gpu' (NVIDIA)")
    args = vars(p.parse_args())
    args["hidden"] = tuple(args["hidden"])
    return Config(**args)


if __name__ == "__main__":
    run(parse_args())
