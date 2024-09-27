import sys
sys.path.append('/app')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from data_module.dogs_datamodule import DogsDataModule
from model.dogs_classifier import DogsClassifier

def main():
    # Create data module
    data_module = DogsDataModule(data_dir='data', batch_size=32)
    
    # Create model
    model = DogsClassifier(num_classes=120, learning_rate=1e-3)
    
    # Setup logging
    logger = TensorBoardLogger("logs", name="dogs_classifier")
    
    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='dogs_classifier-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss'
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