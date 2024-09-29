import sys
sys.path.append('/app')
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from data_module.dogs_datamodule import DogsDataModule
from model.dogs_classifier import DogsClassifier

def main():
    # Create data module
    data_module = DogsDataModule(dl_path='data', batch_size=32)

    # Create model
    model = DogsClassifier(num_classes=120, learning_rate=1e-3)
    
    # Setup logging
    logger = TensorBoardLogger("logs", name="dogs_classifier")

    # Ensure directory exists
    if not os.path.exists('checkpoints/'):
        os.makedirs('checkpoints/')
    

    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',         # Save model with the best validation loss
        dirpath='checkpoints/',     # Directory where checkpoints will be saved
        filename='dogs_classifier-{epoch:02d}-{val_loss:.2f}',  # Checkpoint filename format
        save_top_k=1,               # Save only the best model
        mode='min',                 # 'min' because we want the minimum validation loss
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        callbacks=[checkpoint_callback, RichProgressBar()],
        accelerator='auto',
    )
    
    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
