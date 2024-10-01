
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from torchmetrics import Accuracy
from data_module.dogs_datamodule import DogsDataModule


class DogsClassifier(pl.LightningModule):
    def __init__(self, num_classes =10, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=10)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=10)
        acc = (preds == y).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        
     # class DogsClassifier(pl.LightningModule):
#     def __init__(
#         self,
#         num_classes: int = None,
#         learning_rate: float = 1e-3,
#         pretrained: bool = True,
#         weight_decay: float = 1e-5,
#         factor: float = 0.1,
#         patience: int = 10,
#         min_lr: float = 1e-6,
#     ):
#         super().__init__()
#         self.save_hyperparameters()

#         # Load pre-trained model
#         self.model = models.resnet50(pretrained=pretrained)
        
#         # Multi-class accuracy
#         self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
#         self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
#         self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.cross_entropy(logits, y)
#         preds = F.softmax(logits, dim=1)
#         self.train_acc(preds, y)
#         self.log("train/loss", loss, prog_bar=True)
#         self.log("train/acc", self.train_acc, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.cross_entropy(logits, y)
#         preds = F.softmax(logits, dim=1)
#         self.val_acc(preds, y)
#         self.log("val/loss", loss, prog_bar=True)
#         self.log("val/acc", self.val_acc, prog_bar=True)

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = F.cross_entropy(logits, y)
#         preds = F.softmax(logits, dim=1)
#         self.test_acc(preds, y)
#         self.log("test/loss", loss, prog_bar=True)
#         self.log("test/acc", self.test_acc, prog_bar=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(
#             self.parameters(),
#             lr=self.hparams.learning_rate,
#             weight_decay=self.hparams.weight_decay,
#         )

#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             factor=self.hparams.factor,
#             patience=self.hparams.patience,
#             min_lr=self.hparams.min_lr,
#         )

#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "monitor": "val/loss",
#                 "interval": "epoch",
#             },
#         }

#     # def setup(self, stage=None):
#     #     if stage == "fit":
#     #         # Calculate num_classes based on the datamodule
#     #         num_classes = self.DogsDataModule.num_classes
#     #         # Modify the model's fc layer
#     #         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            
#     #         # Update the accuracy metrics
#     #         self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
#     #         self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
#     #         self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
   
