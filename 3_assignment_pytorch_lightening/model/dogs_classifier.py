
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from torchmetrics import Accuracy
from data_module.dogs_datamodule import DogsDataModule
import torchmetrics
import timm  # Make sure timm is installed: pip install timm

class DogsClassifier(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, num_classes: int = 10):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # Load pre-trained ResNet18 model, adjusting the final layer for 10 classes
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        # Check for and correct out-of-bounds labels during training
        labels = torch.clamp(labels, 0, self.num_classes - 1)

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        # Check for and correct out-of-bounds labels during validation
        labels = torch.clamp(labels, 0, self.num_classes - 1)

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        # Check for and correct out-of-bounds labels during validation
        labels = torch.clamp(labels, 0, self.num_classes - 1)

        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss
