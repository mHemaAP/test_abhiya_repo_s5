import sys
sys.path.append('/app')

import os
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from data_module.dogs_datamodule import DogsDataModule
from model.dogs_classifier import DogsClassifier
import time


def main():
    # Create data module
    data_module = DogsDataModule(dl_path='./data', batch_size=32)

    # Explicitly call prepare_data and setup
    data_module.prepare_data()
    data_module.setup()

    # Get the number of classes
    num_classes = data_module.get_num_classes()
    print(f"Number of classes in the dataset: {num_classes}")

    print("#########################")

    # Create model
    model = DogsClassifier(num_classes=num_classes, lr=1e-3)

    # Setup logging
    logger = TensorBoardLogger("logs", name="dogs_classifier")

    # Ensure directory exists
    checkpoint_dir = '/app/checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True) 

    # ADDITION TO CHECK OLD CHEKPOINTS AND LOGS

    # Remove old checkpoints
    for checkpoint in glob.glob(os.path.join(checkpoint_dir, '*.ckpt')):
        os.remove(checkpoint)


    # Setup log directory cleanup
    log_dir = '/app/logs/'

    # Remove old log files
    for log_file in glob.glob(os.path.join(log_dir, '*')):
        os.remove(log_file)



    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',         # Save model with the best validation loss
        dirpath= checkpoint_dir,     # Directory where checkpoints will be saved
        filename='dogs_classifier-best_val_loss',  # Checkpoint filename format
        save_top_k=1,               # Save only the best model
        mode='min',                 # 'min' because we want the minimum validation loss
        save_last=False             # Don't save the latest, only the best
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        callbacks=[checkpoint_callback, RichProgressBar()],
        accelerator='auto',
        log_every_n_steps=5  # Log more frequently to avoid warning
    )

    print("trainer created")

    # Train the model
    trainer.fit(model, data_module)


    # Print the path of the best model checkpoint
    print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
