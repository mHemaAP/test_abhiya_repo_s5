import pytorch_lightning as pl
from data_module.dogs_datamodule import DogsDataModule
from model.dogs_classifier import DogsClassifier
import torch


def evaluate():
    data_module = DogsDataModule('./data')  # Ensure data path is correct
    model = DogsClassifier.load_from_checkpoint('checkpoints/dogs_classifier-best_val_loss.ckpt', strict=False)

    # Use accelerator='gpu' or accelerator='auto' instead of gpus
    trainer = pl.Trainer(accelerator='auto')
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    evaluate()



