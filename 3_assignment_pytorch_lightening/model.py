import pytorch_lightning as pl
import torch
from torch import nn

class DogBreedModel(pl.LightningModule):
    def __init__(self, num_classes=120):
        super(DogBreedModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*111*111, num_classes)  # Assuming 224x224 input
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
