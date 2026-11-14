import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time

class MyDataset(Dataset):
    def __init__(self,filename):
        super(MyDataset, self).__init__()
        df = pd.read_csv(filename)
        self.features = df.drop(columns=['Cover_Type']).values
        self.labels = df['Cover_Type'].values
    
    def __getitem__(self, idx):
        return self.features[idx].astype(np.float32), self.labels[idx].astype(np.int_)-1 # Subtract 1 to make labels zero-indexed
    
    def __len__(self):
        return len(self.features)
    

# Make 70/10/20 split for training, validation, and testing
full_dataset = MyDataset("./datasets/forest_cover.csv")
n_total = len(full_dataset)
n_train = int(0.7 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val, n_test])

b = 1024 # Batch size
train_loader = DataLoader(train_dataset,batch_size=b)
val_loader = DataLoader(validation_dataset,batch_size=b)
test_loader = DataLoader(test_dataset,batch_size=b)


class LitNetwork(pl.LightningModule):
    def __init__(self, n_classes=2, input_ch=10):
        super(LitNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_ch, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

        # Change n_classes to match target number, assumes binary classification by default
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')
        self.test_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, data, batch_idx):
        features, target = data[0], data[1]
        outs = self.forward(features)
        loss = self.loss_func(outs, target)
        self.log("train_loss",loss)
        return loss
    
    def validation_step(self, val_data, batch_idx):
        features, target = val_data[0], val_data[1]
        outs = self.forward(features)
        self.val_acc(outs,target)
        self.log("val_acc",self.val_acc,prog_bar=True,on_step=False,on_epoch=True)
        return None

    def test_step(self, test_data, batch_idx):
        features, target = test_data[0], test_data[1]
        outs = self.forward(features)
        self.test_acc(outs,target)
        self.log("test_acc",self.test_acc,prog_bar=True,on_step=False,on_epoch=True)
        return None

features_count = 54
targets_cats = 7
model = LitNetwork(n_classes=targets_cats, input_ch=features_count)

checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
logger = pl_loggers.TensorBoardLogger(save_dir="my_logs", name="part1_baseline")
#logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

device = "cpu" # Use 'cpu' or 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU

trainer = pl.Trainer(max_epochs=5, accelerator=device, callbacks=[checkpoint], logger=logger)
start = time.perf_counter()
trainer.fit(model,train_loader,val_loader)
print(f"TRAIN_TIME {time.perf_counter() - start:.1f}")
    
trainer.test(ckpt_path="best", dataloaders=test_loader)
