"""Build Lab6-FinalModel.ipynb: the final (best) Lab 6 model as a self-contained notebook."""

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

INTRO = """\
# Lab 6 – Final Model (Forest Cover, PyTorch Lightning)

This notebook trains the configuration that achieved the best test accuracy in the lab report
(**96.96%**):

| Setting | Value |
| --- | --- |
| Architecture | MLP 54 → [1024, 1024, 1024, 1024] → 7, ReLU, dropout 0.1 after each hidden layer |
| Batch size | 1024 (training data reshuffled every epoch) |
| Epochs | 100 (best checkpoint by validation accuracy) |
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | OneCycleLR, warm-up to 2e-3 then anneal to ~0, stepped every batch |
| Preprocessing | StandardScaler on the 10 continuous features, fit on the training split only |
| Split | 70 / 10 / 20 train / validation / test, fixed seed 42 |

Expects the dataset at `./datasets/forest_cover.csv`. Training takes about 40 minutes on a 4-core CPU.
"""

IMPORTS = """\
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import loggers as pl_loggers
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset"""

CONFIG = """\
DATA_PATH = "./datasets/forest_cover.csv"
TARGET = "Cover_Type"
N_CONTINUOUS = 10  # The first 10 columns are continuous; the remaining 44 are one-hot flags.
N_CLASSES = 7

HIDDEN = (1024, 1024, 1024, 1024)
DROPOUT = 0.1
BATCH_SIZE = 1024
EPOCHS = 100
MAX_LR = 2e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = "cpu"  # 'cpu', 'mps' for Apple silicon, or 'gpu' for an NVIDIA GPU"""

DATA = """\
class ForestCoverDataset(Dataset):
    \"\"\"Holds the whole split as tensors so batches are gathered without per-row conversion.\"\"\"

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def load_splits(path: str, seed: int):
    \"\"\"Fixed-seed 70/10/20 split; the scaler is fit on the training split only to avoid leakage.\"\"\"
    df = pd.read_csv(path)
    features = df.drop(columns=[TARGET]).to_numpy(dtype=np.float64, copy=True)
    labels = df[TARGET].to_numpy() - 1  # Zero-index the 1..7 labels.

    n_total = len(df)
    n_train, n_val = int(0.7 * n_total), int(0.1 * n_total)
    order = np.random.default_rng(seed).permutation(n_total)
    train_idx, val_idx, test_idx = np.split(order, [n_train, n_train + n_val])

    scaler = StandardScaler().fit(features[train_idx, :N_CONTINUOUS])
    features[:, :N_CONTINUOUS] = scaler.transform(features[:, :N_CONTINUOUS])

    return tuple(ForestCoverDataset(features[idx], labels[idx]) for idx in (train_idx, val_idx, test_idx))


pl.seed_everything(SEED, workers=True)
train_dataset, validation_dataset, test_dataset = load_splits(DATA_PATH, SEED)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=8192)
test_loader = DataLoader(test_dataset, batch_size=8192)
print(f"train={len(train_dataset)}  val={len(validation_dataset)}  test={len(test_dataset)}")"""

MODEL = """\
def build_mlp(input_ch: int, hidden: tuple[int, ...], n_classes: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_ch
    for width in hidden:
        layers += [nn.Linear(in_features, width), nn.ReLU(), nn.Dropout(dropout)]
        in_features = width
    layers.append(nn.Linear(in_features, n_classes))
    return nn.Sequential(*layers)


class LitNetwork(pl.LightningModule):
    def __init__(self, input_ch: int, n_classes: int, steps_per_epoch: int):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.network = build_mlp(input_ch, HIDDEN, n_classes, DROPOUT)
        self.loss_func = nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average="micro")
        self.test_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average="micro")

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=self.steps_per_epoch
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

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


model = LitNetwork(train_dataset.features.shape[1], N_CLASSES, len(train_loader))
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")"""

TRAIN = """\
checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc", save_top_k=1, mode="max")
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
logger = pl_loggers.TensorBoardLogger(save_dir="my_logs", name="final_model")

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator=DEVICE,
    callbacks=[checkpoint, lr_monitor],
    logger=logger,
    log_every_n_steps=50,
)
start = time.perf_counter()
trainer.fit(model, train_loader, val_loader)
train_time = time.perf_counter() - start
print(f"Training time: {train_time:.1f} s ({train_time / 60:.1f} min)")
print(f"Best validation accuracy: {checkpoint.best_model_score.item():.4f}")"""

TEST = """\
test_result = trainer.test(ckpt_path="best", dataloaders=test_loader)
print(f"Test accuracy: {100 * test_result[0]['test_acc']:.2f}%")"""

TENSORBOARD = """\
!tensorboard --logdir my_logs"""


def main() -> None:
    nb = new_notebook(
        cells=[
            new_markdown_cell(INTRO),
            new_code_cell(IMPORTS),
            new_markdown_cell("## Configuration"),
            new_code_cell(CONFIG),
            new_markdown_cell("## Data: fixed-seed split and feature scaling"),
            new_code_cell(DATA),
            new_markdown_cell("## Model"),
            new_code_cell(MODEL),
            new_markdown_cell("## Training"),
            new_code_cell(TRAIN),
            new_markdown_cell("## Testing (best checkpoint)"),
            new_code_cell(TEST),
            new_markdown_cell("## TensorBoard\nOr run `tensorboard --logdir my_logs/` from a terminal."),
            new_code_cell(TENSORBOARD),
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    nbformat.write(nb, "Lab6-FinalModel.ipynb")
    print("Wrote Lab6-FinalModel.ipynb")


if __name__ == "__main__":
    main()
